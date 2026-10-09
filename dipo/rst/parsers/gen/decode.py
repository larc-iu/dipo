"""Generic decode core for `gen`.

One greedy loop and one beam loop, each parameterized over the backbone's decode
I/O (`decode_prefix` / `seed` / `advance` / `inference_mode` / `reorder_cache`)
and the serialization's state machine (`decode_id` / `apply` / `candidate_ranges`
/ `stash_meta` / `build_tree`), plus:

  * a `state_factory`: a `() -> state` callback the loop calls to mint fresh
    decode states (pred vs gold-forced is chosen by which factory modeling passes;
    a beam mints one per beam so each carries its own forcer/ranges);
  * a `mask_source`: a `(state, vocab_size) -> BoolTensor` callback giving the
    legal scoring-vocab mask at each step (validity for pred-EDU, gold-boundary
    forcing for gold-EDU).

Decoding is ALWAYS constrained -- there is no unconstrained mode -- so the mask and
the state machine must agree at every step. Where they don't (mask admits an action
`apply` rejects, or a live state has no legal action at all), both loops raise
`DecodeInvariantError` rather than dropping the beam or falling back to a degenerate
tree: a legality bug must not score as if the model produced its output.

This single pair of loops serves every backbone x serialization combo.
"""

import torch

from dipo.rst.parsers.common.seqgen import beam_topk_step, empty_tree, select_best_beam
from dipo.rst.parsers.gen.errors import DecodeInvariantError, OverLengthError


def _masked(logits, mask):
    """Restrict `logits` to the True positions of the bool `mask`."""
    return torch.where(mask.to(logits.device), logits, torch.full_like(logits, float("-inf")))


def _step_budget(bb, cfg, prefix_len: int) -> int:
    """Decode steps this document may take: `max_output_length`, tightened by the
    backbone's positional limit (`Backbone.max_decode_steps`). Training's
    pack_example enforces the same combined-stream cap, so without this a doc could
    silently generate past max_position_embeddings at inference (RoPE
    extrapolation, no error). Exhausting the budget raises OverLengthError."""
    cap = bb.max_decode_steps(prefix_len)
    budget = cfg.max_output_length if cap is None else min(cfg.max_output_length, cap)
    if budget <= 0:  # the prefix alone reaches the positional limit
        raise OverLengthError(f"Output overflowed at inference: {_overlength_detail(budget, cfg)}")
    return budget


def _overlength_detail(step_budget: int, cfg) -> str:
    if step_budget >= cfg.max_output_length:
        return (
            f"decode hit max_output_length={cfg.max_output_length} without completing the "
            f"tree. Bump max_output_length to fit the longest document, then re-run."
        )
    return (
        f"decode hit its step budget ({step_budget} < max_output_length="
        f"{cfg.max_output_length}) without completing the tree: the model's positional "
        f"limit is the binding cap (prefix + output must fit max_position_embeddings). "
        f"Use a longer-context backbone, or drop the offending doc on purpose."
    )


def greedy_decode(parser, source_ids, state_factory, mask_source):
    """One greedy decode. `state_factory()` mints the start state; `mask_source(state,
    V) -> BoolTensor` gives the legal scoring mask per step. Returns an RstTree
    with the source-position meta stashed for the dev eval."""
    bb, ser = parser.backbone, parser.serialization
    cfg = parser.config
    if not source_ids:
        return empty_tree(cfg.relation_types)

    prefix_ids = bb.decode_prefix(source_ids)
    step_budget = _step_budget(bb, cfg, len(prefix_ids))
    st = state_factory()
    action_seq: list[int] = []
    hit_max_len = False

    with bb.inference_mode():
        logits, cache = bb.seed(prefix_ids)
        for _step in range(step_budget):
            mask = mask_source(st, logits.size(-1))
            if not bool(mask.any()):
                # Argmax over an all--inf row would silently pick index 0.
                raise DecodeInvariantError(
                    f"Greedy decode at step {_step}: the {ser.__class__.__name__} state "
                    f"machine offered an empty legal set while not done (automaton "
                    f"deadlock). source_len={len(source_ids)}, "
                    f"actions_so_far={len(action_seq)}."
                )
            step_logits = _masked(logits, mask)
            idx = int(step_logits.argmax(-1).item())
            full_id = ser.decode_id(idx)
            next_input, kind = ser.apply(st, full_id, source_ids)
            if kind in ("illegal", "copy_exhausted"):
                raise DecodeInvariantError(
                    f"Greedy decode at step {_step}: the validity mask admitted action "
                    f"{full_id}, but the {ser.__class__.__name__} state machine rejected "
                    f"it ({kind!r}): the mask and the automaton disagree. "
                    f"source_len={len(source_ids)}, actions_so_far={len(action_seq)}."
                )
            action_seq.append(full_id)
            if next_input is None:  # eos -> stop
                break
            logits, cache = bb.advance(cache, next_input)
        else:
            # ran to max_output_length without an EOS break
            hit_max_len = not st.done

    if hit_max_len:
        raise OverLengthError(f"Output overflowed at inference (greedy): {_overlength_detail(step_budget, cfg)}")
    ranges = ser.candidate_ranges(st, finished=st.done)
    tree = ser.build_tree(action_seq, source_ids)
    ser.stash_meta(tree, ranges, source_ids)
    return tree


def greedy_decode_batch(parser, docs):
    """Greedy-decode B DIFFERENT documents concurrently: one shared forward per step
    over the whole batch instead of one batch-1 forward per document per step.

    `docs` is a list of `(source_ids, state_factory, mask_source)` triples -- the same
    three things `greedy_decode` takes, one per document -- and the return is one tree
    per doc, in the order given. Both the pred-EDU and the gold-EDU-forced conditions
    reach this through their own factory/mask_source, exactly as they do serially.

    This is `greedy_decode` run B-at-a-time, not an approximation of it. The rows never
    interact: each drives its own automaton, builds its own legal mask, and takes its
    own argmax, while the backbone masks padding out of attention and sets each row's
    positions explicitly (see `Backbone.seed_rows`). A row's logits therefore do not
    depend on what shares its batch, so the tree a document gets here is the tree it
    gets alone -- `tests/test_gen_batched_greedy_equivalence.py` pins that as exact
    equality against the serial path, including on ragged batches.

    A row that emits EOS retires: it is fed a pad whose logits are ignored, which
    leaves the other rows' cache and state untouched, and the loop stops once every
    row has retired. Per-row step budgets are honored individually (`OverLengthError`
    fires for exactly the documents that would raise serially).
    """
    bb, ser = parser.backbone, parser.serialization
    cfg = parser.config
    trees: list = [None] * len(docs)
    # An empty source has nothing to decode and no row in the batch (its prefix would
    # be pad-only); `greedy_decode` returns the same degenerate tree for it.
    rows = [i for i, (source_ids, _, _) in enumerate(docs) if source_ids]
    for i, (source_ids, _, _) in enumerate(docs):
        if not source_ids:
            trees[i] = empty_tree(cfg.relation_types)
    if not rows:
        return trees

    pad_id = int(bb.tokenizer.pad_token_id)
    prefixes = [bb.decode_prefix(docs[i][0]) for i in rows]
    budgets = [_step_budget(bb, cfg, len(p)) for p in prefixes]
    states = [docs[i][1]() for i in rows]
    action_seqs: list[list[int]] = [[] for _ in rows]
    live = [True] * len(rows)  # still being stepped
    stopped_on_eos = [False] * len(rows)  # retired by EOS, not by exhausting its budget

    with bb.inference_mode():
        logits, cache = bb.seed_rows(prefixes)  # [B, V]
        for step in range(max(budgets)):
            next_inputs = [pad_id] * len(rows)
            for j, i in enumerate(rows):
                if not live[j]:
                    continue
                if step >= budgets[j]:
                    # This row's serial loop would have ended here; the post-loop
                    # `st.done` check below decides whether that is an overflow.
                    live[j] = False
                    continue
                st, mask_source = states[j], docs[i][2]
                mask = mask_source(st, logits.size(-1))
                if not bool(mask.any()):
                    # Argmax over an all--inf row would silently pick index 0.
                    raise DecodeInvariantError(
                        f"Batched greedy decode (row {j}) at step {step}: the "
                        f"{ser.__class__.__name__} state machine offered an empty legal set "
                        f"while not done (automaton deadlock). source_len={len(docs[i][0])}, "
                        f"actions_so_far={len(action_seqs[j])}."
                    )
                idx = int(_masked(logits[j], mask).argmax(-1).item())
                full_id = ser.decode_id(idx)
                next_input, kind = ser.apply(st, full_id, docs[i][0])
                if kind in ("illegal", "copy_exhausted"):
                    raise DecodeInvariantError(
                        f"Batched greedy decode (row {j}) at step {step}: the validity mask "
                        f"admitted action {full_id}, but the {ser.__class__.__name__} state "
                        f"machine rejected it ({kind!r}): the mask and the automaton disagree. "
                        f"source_len={len(docs[i][0])}, actions_so_far={len(action_seqs[j])}."
                    )
                action_seqs[j].append(full_id)
                if next_input is None:  # eos -> this row retires
                    live[j], stopped_on_eos[j] = False, True
                    continue
                next_inputs[j] = next_input
            if not any(live):
                break  # nothing left to extend; skip a pointless advance_rows()
            logits, cache = bb.advance_rows(cache, next_inputs)

    for j, i in enumerate(rows):
        st = states[j]
        # Mirrors greedy_decode's `hit_max_len = not st.done` on the for-else: a row
        # that ran out its budget instead of breaking on EOS overflowed, unless its
        # automaton is nonetheless done.
        if not stopped_on_eos[j] and not st.done:
            raise OverLengthError(
                f"Output overflowed at inference (batched greedy): {_overlength_detail(budgets[j], cfg)}"
            )
        ranges = ser.candidate_ranges(st, finished=st.done)
        tree = ser.build_tree(action_seqs[j], docs[i][0])
        ser.stash_meta(tree, ranges, docs[i][0])
        trees[i] = tree
    return trees


def beam_decode(parser, source_ids, num_beams, state_factory, mask_source):
    """K-beam decode. Replicates the prefix across the batch dim so the prefix
    forward seeds K identical caches; only beam 0 is alive at step 0, so beams
    diverge after the first step. `mask_source(state, V) -> BoolTensor` gives each
    beam's legal scoring mask (beam always constrains). Returns an RstTree."""
    bb, ser = parser.backbone, parser.serialization
    cfg = parser.config
    K = int(num_beams)
    if not source_ids:
        return empty_tree(cfg.relation_types)
    device = parser.device
    pad_id = int(bb.tokenizer.pad_token_id)
    prefix_ids = bb.decode_prefix(source_ids)
    step_budget = _step_budget(bb, cfg, len(prefix_ids))

    with bb.inference_mode():
        logits, cache = bb.seed(prefix_ids, num_rows=K)  # [K, V]
        states = [state_factory() for _ in range(K)]
        action_seqs: list[list[int]] = [[] for _ in range(K)]
        finished_beams: list[dict] = []

        beam_scores = torch.full((K,), float("-inf"), device=device)
        beam_scores[0] = 0.0

        for step in range(step_budget):
            # A beam is live if it can still be extended: not finished, and not a
            # dead row (-inf, either a finished beam retired below or a topk
            # backfill). Dead rows are never advanced, so their states never reach
            # `done` -- liveness, not `done`, is what terminates the loop.
            live = [j for j, st in enumerate(states) if not st.done and torch.isfinite(beam_scores[j])]
            if not live:
                break

            legal = torch.zeros_like(logits, dtype=torch.bool)  # [K, V]
            for j in live:
                mask = mask_source(states[j], logits.size(-1)).to(device)
                if not bool(mask.any()):
                    raise DecodeInvariantError(
                        f"Beam {j} at step {step}: the {ser.__class__.__name__} state "
                        f"machine offered an empty legal set while not done (automaton "
                        f"deadlock). source_len={len(source_ids)}, "
                        f"actions_so_far={len(action_seqs[j])}."
                    )
                legal[j] = mask
            top_scores, parent_of_new, action_of_new = beam_topk_step(beam_scores, logits, legal, K)

            parent_tensor = torch.tensor(parent_of_new, device=device, dtype=torch.long)
            if bb.beam_reorder_needed(step, parent_of_new, K, cache):
                cache = bb.reorder_cache(cache, parent_tensor)

            # Each child gets its OWN cloned parent state (siblings must not share).
            new_states = [states[p].clone() for p in parent_of_new]
            new_action_seqs = [list(action_seqs[p]) for p in parent_of_new]

            next_inputs = [pad_id] * K
            for j in range(K):
                st = new_states[j]
                if st.done:
                    continue
                if not torch.isfinite(top_scores[j]):
                    # -inf topk backfill: fewer than K legal continuations existed, so
                    # this row's action came from a masked-out column. It is not a real
                    # hypothesis -- never apply it to the state.
                    continue
                full_id = ser.decode_id(action_of_new[j])
                nxt, kind = ser.apply(st, full_id, source_ids)
                if kind in ("illegal", "copy_exhausted"):
                    raise DecodeInvariantError(
                        f"Beam {j} at step {step}: the validity mask admitted action "
                        f"{full_id}, but the {ser.__class__.__name__} state machine "
                        f"rejected it ({kind!r}). Score was finite ({float(top_scores[j]):.4f}), "
                        f"so this was a mask-legal column, not -inf backfill: the mask and "
                        f"the automaton disagree. source_len={len(source_ids)}, "
                        f"actions_so_far={len(new_action_seqs[j])}."
                    )
                new_action_seqs[j].append(full_id)
                if nxt is not None:
                    next_inputs[j] = nxt

            states, action_seqs = new_states, new_action_seqs
            beam_scores = top_scores

            for j, st in enumerate(states):
                if st.done and torch.isfinite(beam_scores[j]):
                    finished_beams.append({
                        "action_seq": list(action_seqs[j]),
                        "pred_edu_ranges": ser.candidate_ranges(st, finished=True),
                        "score": float(beam_scores[j].item()),
                        "length": len(action_seqs[j]),
                        "finished": True,
                    })
                    beam_scores[j] = float("-inf")

            if not any(not st.done and torch.isfinite(beam_scores[j]) for j, st in enumerate(states)):
                break  # nothing left to extend; skip a pointless advance()
            logits, cache = bb.advance(cache, next_inputs)

    candidates: list[dict] = list(finished_beams)
    for j, st in enumerate(states):
        if not st.done and torch.isfinite(beam_scores[j]):
            candidates.append({
                "action_seq": list(action_seqs[j]),
                "pred_edu_ranges": ser.candidate_ranges(st, finished=False),
                "score": float(beam_scores[j].item()),
                "length": len(action_seqs[j]),
                "finished": False,
            })

    if not candidates:
        # Every beam is dead (-inf) without any having finished: the live states'
        # legal sets all came back empty, i.e. the automaton deadlocked while not
        # done. Under a correct mask this is unreachable.
        raise DecodeInvariantError(
            f"Beam decode ended with no candidate hypothesis over {K} beams: every live "
            f"state offered an empty legal set (automaton deadlock). "
            f"source_len={len(source_ids)}, actions_per_beam={[len(a) for a in action_seqs]}."
        )
    best = select_best_beam(candidates)
    if not best.get("finished", False):
        raise OverLengthError(f"Output overflowed at inference (beam, no beam finished): {_overlength_detail(step_budget, cfg)}")
    tree = ser.build_tree(best["action_seq"], source_ids)
    ser.stash_meta(tree, best["pred_edu_ranges"], source_ids)
    return tree
