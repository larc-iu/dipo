"""Soundness and admission tests for the budget-range `GoldEduForcer`.

The forcer's contract: gold segmentation by construction, tree shape by the
model. Two failure modes are pre-specified and pinned here:

  * UNSOUNDNESS (under-bracketing): a naive "force leaf extents, defer the
    rest" forcer lets the model start the root as a leaf on a multi-EDU doc
    (the PDA allows a 1-EDU tree), and the forced early close then breaks
    gold segmentation. The soundness tests drive adversarial and random
    policies through the forcer and require exactly the gold leaves at
    exactly the gold boundaries, terminal state, bounded steps.
  * OVER-FORCING (the spine bug): a forcer that never defers fixes the tree
    shape and reduces gold-EDU Parseval to a labeling metric. The admission
    tests require that DIFFERENT model policies reach DIFFERENT shapes, and
    exhaustively that the reachable shape set over n gold leaves is ALL
    Catalan(n-1) binary shapes, nothing more, nothing fewer.

Action-id convention matches tests/test_sexp_constraints.py:
    OPEN=1 CLOSE=2 EOS=3 COPY=4   LABEL_NS=100 LABEL_SN=101 LABEL_NN=102
Source ids start at 10. Policies pick from exactly the mask the modeling
consumers would build per the `narrowed_legal` contract (None | frozenset).
"""

from __future__ import annotations

import copy
import random

import pytest

from dipo.rst.parsers.common.sexp_constraints import GoldEduForcer, SexpDecodingState


OPEN_ID = 1
CLOSE_ID = 2
EOS_ID = 3
COPY_ID = 4
LABEL_NS = 100
LABEL_SN = 101
LABEL_NN = 102
LABEL_IDS = frozenset({LABEL_NS, LABEL_SN, LABEL_NN})

# `copy` = the <copy>-sentinel content stream; `verbatim` = use_copy=False,
# content pinned to the source id at the cursor.
MODES = ["copy", "verbatim"]


def _make_state(source_len: int, traversal_order: str, mode: str) -> SexpDecodingState:
    source_ids = [10 + i for i in range(source_len)]
    return SexpDecodingState(
        source_len=source_len,
        traversal_order=traversal_order,
        use_copy=(mode == "copy"),
        open_id=OPEN_ID,
        close_id=CLOSE_ID,
        eos_id=EOS_ID,
        label_ids=LABEL_IDS,
        copy_id=COPY_ID if mode == "copy" else None,
        source_ids=() if mode == "copy" else tuple(source_ids),
        min_edu_length=1,
    )


def _ranges(widths: list[int], start: int = 0) -> tuple[list[tuple[int, int]], int]:
    """Contiguous gold ranges with the given per-leaf widths."""
    ranges, cur = [], start
    for w in widths:
        ranges.append((cur, cur + w))
        cur += w
    return ranges, cur


def _options(state: SexpDecodingState, narrowed) -> list[int]:
    """The candidate ids a consumer's mask admits, per the narrowed_legal
    contract (None = the full legal set, frozenset = a whitelist)."""
    legal = state.legal_actions()
    if narrowed is None:
        return sorted(legal)
    if len(narrowed) == 1:
        return [next(iter(narrowed))]
    return sorted(narrowed & legal)


def _drive(state: SexpDecodingState, forcer: GoldEduForcer, policy, max_steps: int):
    """Run a forced decode with `policy(options) -> id`. Returns
    (final_state, actions, leaf_spans)."""
    actions: list[int] = []
    leaf_spans: list[tuple[int, int]] = []
    leaf_start = None
    for _ in range(max_steps):
        if state.is_terminal():
            break
        options = _options(state, forcer.narrowed_legal(state))
        assert options, f"empty option set at step {len(actions)}: actions={actions}"
        chosen = policy(options)
        before = state
        state = state.step(chosen)
        forcer.observe(before, state, chosen)
        actions.append(chosen)
        if state.in_edu_leaf and not before.in_edu_leaf:
            leaf_start = before.cursor
        if chosen == CLOSE_ID and before.in_edu_leaf:
            leaf_spans.append((leaf_start, before.cursor))
            leaf_start = None
    return state, actions, leaf_spans


def _structural_policy(options: list[int]) -> int:
    """Adversarial: prefer structural tokens, so any leaf the forcer fails
    to pin becomes an internal node."""
    for fid in (OPEN_ID, LABEL_NS, LABEL_SN, LABEL_NN, CLOSE_ID, EOS_ID):
        if fid in options:
            return fid
    return options[0]


def _leafy_policy(options: list[int]) -> int:
    """Adversarial the other way: prefer content, so every deferral becomes
    a leaf as early as possible."""
    structural = {OPEN_ID, CLOSE_ID, EOS_ID} | LABEL_IDS
    for fid in options:
        if fid not in structural:
            return fid
    return options[0]


def _shape(actions: list[int]) -> str:
    """Collapse an action sequence to its bracket shape: leaves become 'L',
    labels vanish, internal nodes keep their parens."""
    structural = {OPEN_ID: "(", CLOSE_ID: ")"}
    ignore = LABEL_IDS | {EOS_ID}
    out = []
    for a in actions:
        if a in ignore:
            continue
        ch = structural.get(a, "c")
        if ch == "c" and out and out[-1] == "c":
            continue
        out.append(ch)
    s = "".join(out).replace("c", "L")
    while "(L)" in s:
        s = s.replace("(L)", "L")
    return s


def _catalan(n: int) -> int:
    c = 1
    for i in range(n):
        c = c * 2 * (2 * i + 1) // (i + 2)
    return c


def _step_bound(source_len: int, n_leaves: int) -> int:
    # 2n-1 nodes, each at most OPEN + label + CLOSE, plus one action per
    # source position, plus EOS. Double it as the runaway line.
    return 2 * (source_len + (2 * n_leaves - 1) * 3 + 1)


# ---------------------------------------------------------------------------
# Soundness gate (PDA level): exact gold leaves under adversarial and random
# policies, every mode, both traversal orders, mixed leaf widths.
# ---------------------------------------------------------------------------


def _policies():
    yield "structural", _structural_policy
    yield "leafy", _leafy_policy
    for seed in range(3):
        rng = random.Random(seed)
        yield f"random{seed}", (lambda options, rng=rng: rng.choice(options))


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n_leaves", [1, 2, 3, 5, 8])
def test_soundness_exact_gold_leaves(traversal_order, mode, n_leaves):
    widths = [(i % 3) + 1 for i in range(n_leaves)]
    gold, source_len = _ranges(widths)
    for name, policy in _policies():
        state = _make_state(source_len, traversal_order, mode)
        forcer = GoldEduForcer(n_leaves, list(gold))
        final, actions, spans = _drive(state, forcer, policy, _step_bound(source_len, n_leaves))
        ctx = f"(order={traversal_order}, mode={mode}, n={n_leaves}, policy={name})"
        assert final.is_terminal(), f"no termination {ctx}: actions={actions}"
        assert forcer.closed_leaves == n_leaves, f"leaf count {forcer.closed_leaves} != {n_leaves} {ctx}"
        assert spans == gold, f"leaf spans {spans} != gold {gold} {ctx}"
        assert final.cursor == source_len, ctx


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_soundness_trailing_tokens_absorbed_by_last_leaf(traversal_order):
    """Gold ranges that stop short of source_len (alignment slack): the last
    leaf must absorb the tail, everything else stays gold."""
    gold, _ = _ranges([2, 1, 2])
    source_len = 7  # two trailing positions past the last gold end
    for name, policy in _policies():
        state = _make_state(source_len, traversal_order, "copy")
        forcer = GoldEduForcer(3, list(gold))
        final, actions, spans = _drive(state, forcer, policy, _step_bound(source_len, 3))
        assert final.is_terminal(), f"policy={name}: actions={actions}"
        assert forcer.closed_leaves == 3
        assert spans[:-1] == gold[:-1], f"policy={name}: {spans}"
        assert spans[-1] == (gold[-1][0], source_len), f"policy={name}: {spans}"


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_soundness_gap_between_ranges(traversal_order):
    """A gap before a gold range is absorbed by the leaf that spans it (the
    leaf starts at the cursor, ends at the gold end)."""
    gold = [(0, 2), (3, 5)]  # position 2 belongs to no gold EDU
    source_len = 5
    for name, policy in _policies():
        state = _make_state(source_len, traversal_order, "copy")
        forcer = GoldEduForcer(2, list(gold))
        final, actions, spans = _drive(state, forcer, policy, _step_bound(source_len, 2))
        assert final.is_terminal(), f"policy={name}: actions={actions}"
        assert forcer.closed_leaves == 2
        assert [e for _, e in spans] == [2, 5], f"policy={name}: {spans}"


# ---------------------------------------------------------------------------
# Admission: the forcer must NOT fix the tree shape (the spine bug).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
@pytest.mark.parametrize("mode", MODES)
def test_admits_left_and_right_branching(traversal_order, mode):
    """The pre-specified admission case: over 3 gold leaves, an OPEN-greedy
    policy reaches the left chain ((e0 e1) e2) and a content-greedy policy
    reaches the right chain (e0 (e1 e2)). A shape-forcing forcer fails this."""
    gold, source_len = _ranges([2, 2, 2])
    shapes = {}
    for name, policy in (("structural", _structural_policy), ("leafy", _leafy_policy)):
        state = _make_state(source_len, traversal_order, mode)
        forcer = GoldEduForcer(3, list(gold))
        final, actions, spans = _drive(state, forcer, policy, _step_bound(source_len, 3))
        assert final.is_terminal() and spans == gold, f"policy={name}: actions={actions}"
        shapes[name] = _shape(actions)
    assert shapes["structural"] == "((LL)L)", shapes
    assert shapes["leafy"] == "(L(LL))", shapes


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
@pytest.mark.parametrize("n_leaves", [1, 2, 3, 4, 5])
def test_exhaustive_reachable_shapes_are_all_binary_trees(traversal_order, n_leaves):
    """Enumerate EVERY decode reachable under the forcer (branching over the
    full option set at each step, single label id to keep paths = shapes) and
    require the terminal shape set to be exactly the Catalan(n-1) binary
    shapes over the gold leaves, each with exact gold spans."""
    gold, source_len = _ranges([1] * n_leaves)
    label_one = frozenset({LABEL_NS})

    def initial():
        st = _make_state(source_len, traversal_order, "copy")
        import dataclasses

        return dataclasses.replace(st, label_ids=label_one)

    shapes: set[str] = set()
    n_terminals = 0
    stack = [(initial(), GoldEduForcer(n_leaves, list(gold)), [], [], None)]
    bound = _step_bound(source_len, n_leaves)
    while stack:
        state, forcer, actions, spans, leaf_start = stack.pop()
        if state.is_terminal():
            n_terminals += 1
            assert forcer.closed_leaves == n_leaves, f"actions={actions}"
            assert spans == gold, f"spans {spans} != gold {gold}: actions={actions}"
            shapes.add(_shape(actions))
            continue
        assert len(actions) < bound, f"runaway branch: actions={actions}"
        options = _options(state, forcer.narrowed_legal(state))
        assert options, f"deadlock (empty options): actions={actions}"
        for chosen in options:
            f2 = copy.deepcopy(forcer)
            before = state
            after = state.step(chosen)
            f2.observe(before, after, chosen)
            spans2, leaf_start2 = list(spans), leaf_start
            if after.in_edu_leaf and not before.in_edu_leaf:
                leaf_start2 = before.cursor
            if chosen == CLOSE_ID and before.in_edu_leaf:
                spans2.append((leaf_start2, before.cursor))
                leaf_start2 = None
            stack.append((after, f2, actions + [chosen], spans2, leaf_start2))

    assert len(shapes) == _catalan(n_leaves - 1), (
        f"reachable shapes {sorted(shapes)} != Catalan({n_leaves - 1}) = {_catalan(n_leaves - 1)}"
    )


# ---------------------------------------------------------------------------
# Model level: both sexp parsers, untrained-or-tiny backbones. The soundness
# gate at the modeling seam: predict_with_gold_edus yields exactly the gold
# EDU count on a multi-EDU toy doc regardless of weights.
# ---------------------------------------------------------------------------


def _toy_tree(n_edus: int):
    from dipo.rst.data.tree import Reduce, RstTree, Shift

    texts = ["Cats sleep.", "Dogs bark.", "Birds sing.", "Fish swim.", "Ants march."][:n_edus]
    actions = [Shift(edu_text=texts[0])]
    for t in texts[1:]:
        actions.append(Shift(edu_text=t))
        actions.append(Reduce(nuc="NS", rel="elaboration"))
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst")])


def _build_gen_sexp(backbone: str):
    transformers = pytest.importorskip("transformers")  # noqa: F841
    import os

    from dipo.rst.parsers.gen.configuration_gen import GenConfig
    from dipo.rst.parsers.gen.modeling_gen import GenParser

    if backbone == "seq2seq":
        model_name = os.environ.get("DIPO_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")
    else:
        model_name = os.environ.get("DIPO_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
    d = dict(
        backbone=backbone,
        serialization="sexp",
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=model_name,
        relation_types=[("elaboration", "rst")],
        gradient_checkpointing=False,
        amp=False,
        max_input_length=256,
        max_output_length=512,
        min_edu_length=1,
        traversal_order="postorder",
        use_copy=True,
    )
    try:
        return GenParser(GenConfig.from_dict(d))
    except Exception as e:
        pytest.skip(f"Could not load {model_name}: {e!r}")


def _build_seq2seq_sexp():
    return _build_gen_sexp("seq2seq")


def _build_decoder_only_sexp():
    return _build_gen_sexp("decoder_only")


@pytest.mark.parametrize("build", [_build_seq2seq_sexp, _build_decoder_only_sexp])
def test_model_level_gold_edu_count_and_boundaries(build):
    from dipo.rst.parsers.common.seqgen import gold_edu_source_ranges

    parser = build()
    tree = _toy_tree(4)
    pred = parser.predict_with_gold_edus(tree)
    assert len(pred.edus) == len(tree.edus), (
        f"gold-EDU forced decode produced {len(pred.edus)} EDUs, gold has {len(tree.edus)}"
    )
    gold_ranges = gold_edu_source_ranges(parser.tokenizer, tree)
    pred_ranges = getattr(pred, "_pred_edu_source_ranges", [])
    assert len(pred_ranges) == len(gold_ranges)
    # Ends must match gold exactly except the last (which absorbs any
    # trailing source positions the alignment left uncovered).
    assert [e for _, e in pred_ranges[:-1]] == [e for _, e in gold_ranges[:-1]], (
        f"pred {pred_ranges} vs gold {gold_ranges}"
    )
    assert pred_ranges[-1][1] >= gold_ranges[-1][1]
