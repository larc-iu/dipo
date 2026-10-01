"""Shared utilities for the generative (text-to-tree) RST parser `gen`, which
fine-tunes a seq2seq or causal LM (the `backbone` axis) to emit a linearized tree
(shift-reduce or s-expression, the `serialization` axis). These helpers are pure,
self-contained functions the backbone/serialization strategies compose:

- `align_edus_to_tokens`: the EDU to subword tiling that keeps train-time COPY
  substitution in lockstep with the inference copy-every-source-token
  constraint. The tiling invariant must agree across train and predict.
- `reorder_past_key_values`: beam-search KV-cache reordering, defensive
  HF-version-compat plumbing.
- `beam_topk_step` / `beam_reorder_needed` / `select_best_beam`: the
  serialization-agnostic beam-search primitives (top-K expansion with the
  dead-beam NaN guard, the reorder-is-a-no-op predicate, and GNMT
  length-normalized candidate selection), consumed by gen's decode core
  (`gen/decode.py`).
- `ShiftReduceDecodeState`: the shift-reduce decode automaton.
- `reconstruct_text` / `gold_edu_source_ranges` / `empty_tree`: text
  reconstruction, gold-range tiling, and the single-EDU fallback tree.
- `chunked_cross_entropy`: memory-bounded full-vocab CE (words mode at
  27-31B scale).

The encoder-based parsers (`dmrst`, `topdown_biaffine`, `sr_biaffine`)
do not use these; their shared token-encoding lives in `common/encoding.py`.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from iudex.common.log import warn
from iudex.rst.data.tree import RstTree, Shift, ShiftReduceAction

# GNMT length-normalization exponent for beam selection (Wu et al. 2016), the
# `select_best_beam` default.
BEAM_LENGTH_PENALTY_ALPHA = 0.6


# -----------------------------------------------------------------
# Action-head warm-init
# -----------------------------------------------------------------


def warm_init_head(new_linear: torch.nn.Linear, unembed_weight: torch.Tensor, full_id_for_head_idx: list[int]) -> None:
    """Warm-init each row of a freshly-built small action head from the matching
    row of `unembed_weight`, the model's unembedding matrix: pass the OLD
    lm_head weight when embeddings are untied (large Qwen/Gemma backbones), or
    the tied `embed_tokens` weight (identical tensor to the lm_head in that
    case, e.g. Gemma-3/T5Gemma). Row `full_id` of that matrix is the hidden
    direction the pretrained model maps to token `full_id`; copying those rows
    into the small head means the model starts already knowing which hidden
    direction maps to which token, skipping the training that would otherwise
    just relearn that alignment. For action tokens whose row was freshly
    created by `resize_token_embeddings` the row is itself random, so this is
    no worse than an N(0, 0.02) init there (and strictly better for
    pre-existing tokens like EOS). `full_id_for_head_idx[hi]` is the full-vocab
    id seeding head row `hi`. Mutates `new_linear.weight`.

    If `unembed_weight` is the wrong width to copy into the head (tied input
    embeddings on asymmetric encoder/decoder backbones, e.g. t5gemma-9b-2b:
    3584-wide encoder embeddings but a 2304-wide decoder lm_head), fall back to
    the same N(0, 0.02) init fresh rows would otherwise get rather than
    crashing on the dim mismatch.
    """
    with torch.no_grad():
        if unembed_weight.shape[-1] != new_linear.weight.shape[-1]:
            new_linear.weight.normal_(mean=0.0, std=0.02)
            return
        for hi, full_id in enumerate(full_id_for_head_idx):
            src = unembed_weight[full_id].to(dtype=new_linear.weight.dtype, device=new_linear.weight.device)
            new_linear.weight[hi].copy_(src)


def relation_to_words(rel: str) -> str:
    """Natural-language spelling of a relation label for `label_style='words'`:
    hyphens become spaces so every piece is a clean, space-prefixed pretrained
    word token (`same-unit` -> `same unit`, `topic-comment` -> `topic comment`).
    Single-word relations (`elaboration`, `joint`, ...) are unchanged."""
    return rel.replace("-", " ")


def build_word_label_vocab(tokenizer, relation_types, terminator_id=None):
    """For `label_style='words'`. Maps each (nuc, rel) merge label to the token-id
    sequence for `" <nuc> <relation words>"` (a leading space so every piece hits
    a pretrained word row, e.g. `" NS elaboration"` -> `[<space>NS, <space>elaboration]`),
    and the inverse. The nuclearity marker is the abbreviated `NS`/`SN`/`NN`; the
    relation words are ordinary pretrained-vocab tokens scored over the full
    lm_head. Returns `(label_to_ids, ids_to_label)` keyed by (nuc, rel) tuples and
    by token-id tuples respectively.

    `terminator_id` (a dedicated `<label_end>` id) is appended to every label
    sequence when set. The decode/reconstruct machinery assumes prefix-free labels;
    raw relation words are NOT prefix-free (`elaboration` is a prefix of
    `elaboration-additional`), so without a terminator the longer label is
    unreachable at decode and mis-parsed at reconstruct. The terminator delimits
    every label, making the whole set prefix-free by construction (so any unique
    label set is decodable). None reproduces the pre-terminator (prefix-fragile)
    encoding."""
    label_to_ids: dict[tuple[str, str], tuple[int, ...]] = {}
    ids_to_label: dict[tuple[int, ...], tuple[str, str]] = {}
    suffix = (int(terminator_id),) if terminator_id is not None else ()
    for rel, kind in relation_types:
        nucs = ("NN",) if kind == "multinuc" else ("NS", "SN")
        for nuc in nucs:
            ids = tuple(tokenizer(f" {nuc} {relation_to_words(rel)}", add_special_tokens=False)["input_ids"]) + suffix
            label_to_ids[(nuc, rel)] = ids
            ids_to_label[ids] = (nuc, rel)
    return label_to_ids, ids_to_label


# -----------------------------------------------------------------
# EDU <-> token alignment
# -----------------------------------------------------------------


def align_edus_to_tokens(
    tokenizer: Any,
    text: str,
    edus: Any,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize `text` (the reconstructed document) and partition its subword
    tokens among `edus` so the per-EDU token ranges TILE range(len(input_ids))
    exactly: no gaps, no overlaps, sum of lengths == len(input_ids).

    Tokenizing the whole doc once and partitioning (rather than tokenizing each
    EDU separately and concatenating) is deliberate: SentencePiece is
    whitespace-sensitive, so per-EDU tokenizations drift from the encoder's
    actual whole-doc tokenization by a few subwords per doc. Tiling keeps the
    gold EDU ranges (`encode_target`, `gold_edu_source_ranges`) in the same
    token space as the pred ranges the inference loop tracks by cursor.

    `edus` is a sequence of objects with `.text: str` and `.prefix: str | None`
    (default prefix " " for all but the first EDU), matching how `reconstruct_text`
    builds `text`. Assignment is by a single monotonic forward sweep over tokens:
    each token goes to the current EDU until its character midpoint crosses into
    the next EDU's char range, and the final EDU absorbs all trailing tokens. This
    guarantees a tiling even when a token straddles a boundary or sits in
    inter-EDU whitespace. An EDU shorter than a token may receive an empty range
    (start == end), which is allowed and still tiles.

    Returns (input_ids: list[int], edu_token_spans: list[tuple[int, int]]) where
    edu_token_spans[i] = (start, end) is a half-open token-index range into
    input_ids for EDU i.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # Exclusive char-end per EDU, walking prefixes/text exactly like reconstruct_text.
    char_ends: list[int] = []
    char_cursor = 0
    for i, edu in enumerate(edus):
        if i > 0:
            prefix = edu.prefix if edu.prefix is not None else " "
            char_cursor += len(prefix)
        char_cursor += len(edu.text)
        char_ends.append(char_cursor)

    n_edus = len(char_ends)
    counts = [0] * n_edus
    edu_idx = 0
    for tcs, tce in offsets:
        m = (tcs + tce) / 2
        while edu_idx < n_edus - 1 and m >= char_ends[edu_idx]:
            edu_idx += 1
        counts[edu_idx] += 1

    spans: list[tuple[int, int]] = []
    cursor = 0
    for c in counts:
        spans.append((cursor, cursor + c))
        cursor += c
    return input_ids, spans


# -----------------------------------------------------------------
# Beam search
# -----------------------------------------------------------------


def reorder_past_key_values(past_key_values, beam_idx: torch.Tensor, model):
    """Reorder a HF past_key_values cache along the beam dimension. Handles
    three layouts:
      1. The model exposes `_reorder_cache(pkv, beam_idx)` (T5/T5Gemma2 and most
         HF seq2seq models).
      2. `past_key_values` is a `DynamicCache`-like object with its own
         `reorder_cache` method (newer transformers).
      3. Tuple-of-tuple of Tensors (older HF), possibly with `None` entries for
         unfilled cross-attention slots.

    `model` is the underlying (PEFT-unwrapped) model that may carry the legacy
    `_reorder_cache` helper.
    """
    # Path 1: canonical HF helper on the base model. T5Gemma 2's inherited
    # `_reorder_cache` assumes the legacy tuple-of-tuple layout, and newer HF
    # versions may hand us a DynamicCache instead, which makes that call blow
    # up. Catch and fall through to the next path on type/attribute mismatches.
    reorder = getattr(model, "_reorder_cache", None)
    if callable(reorder):
        try:
            result = reorder(past_key_values, beam_idx)
            # Modern HF cache classes mutate in place and return None.
            # Blindly returning None drops the cache on the next step.
            return result if result is not None else past_key_values
        except (TypeError, AttributeError) as e:
            warn(
                f"{type(model).__name__}._reorder_cache failed on "
                f"{type(past_key_values).__name__} ({type(e).__name__}: {e}). "
                "Falling back to object/tuple cache reordering."
            )
    # Path 2: DynamicCache or similar object-style cache.
    if hasattr(past_key_values, "reorder_cache"):
        result = past_key_values.reorder_cache(beam_idx)
        return result if result is not None else past_key_values
    # Path 2b: newer transformers Cache classes renamed the beam reorder to
    # `batch_select_indices` (mutates in place, returns None). Without this
    # path a future HF bump would fall through to the tuple walk below, which
    # iterates a Cache object with version-dependent layout and could silently
    # mis-reorder every beam decode.
    if hasattr(past_key_values, "batch_select_indices"):
        result = past_key_values.batch_select_indices(beam_idx)
        return result if result is not None else past_key_values
    # Path 3: manual tuple walk, handling Nones gracefully.
    return tuple(
        tuple(t.index_select(0, beam_idx) if isinstance(t, torch.Tensor) else t for t in layer)
        for layer in past_key_values
    )


def beam_topk_step(
    beam_scores: torch.Tensor,
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """One beam-search expansion step, serialization-agnostic.

    `logits` is the [K, V] RAW (unmasked) model output; `legal_mask` is a
    [K, V] bool tensor, True where the caller's validity constraints admit the
    continuation (a done/dead beam's row should be all False). `beam_scores`
    is [K] cumulative log-prob (dead beams at -inf). Returns the top-K
    continuations as (top_scores [K], parents list[int], actions list[int]),
    where the flat top-k index `flat = parent * V + action` is decoded back
    into the parent beam and the chosen action (a column of `logits`).

    Scoring is deliberately UNrenormalized: log_softmax runs over the full raw
    vocab FIRST, and illegal entries are then dropped to -inf. Renormalizing
    over the legal set (log_softmax of pre-masked logits, the previous
    behavior) made heavily-constrained steps nearly free. Under the sexp
    constraints the in-leaf legal set is ~2 ids, so a full-vocab distribution
    renormalized to a binary choice priced skipping an EDU boundary at ~0 and
    beam search collapsed unconfident documents to a handful of giant EDUs (a
    legal, "high-scoring" parse the raw model distribution hates, e.g.
    wsj_1118: renormalized -2.5 vs raw -974 for the 3-EDU parse). Raw scoring
    matches the HF generate() default (renormalize_logits is opt-in there) and
    greedy argmax is unaffected either way. The NaN guard stays as a backstop
    for -inf raw logits. The mask-AFTER-log_softmax order is the whole point;
    it lives here so it is defined exactly once (the failure mode is silent)."""
    v = logits.size(-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_probs = torch.where(legal_mask, log_probs, torch.full_like(log_probs, float("-inf")))
    cum = beam_scores.unsqueeze(1) + log_probs
    cum = torch.where(torch.isnan(cum), torch.full_like(cum, float("-inf")), cum)
    top_scores, top_idx = cum.view(-1).topk(k)
    parents = (top_idx // v).tolist()
    actions = (top_idx % v).tolist()
    return top_scores, parents, actions


def beam_reorder_needed(step: int, parents: list[int], k: int, past_key_values) -> bool:
    """Whether the KV cache + decoder inputs need reordering by `parents` this
    step. Skips the no-op cases: step 0 expands K identical rows from the single
    seed beam (all parents 0), and an identity permutation rearranges nothing.
    `past_key_values is None` (pre-cache) also needs no reorder."""
    if past_key_values is None:
        return False
    is_step0_uniform = step == 0 and all(p == 0 for p in parents)
    is_identity = parents == list(range(k))
    return not (is_step0_uniform or is_identity)


def select_best_beam(candidates: list[dict], alpha: float = BEAM_LENGTH_PENALTY_ALPHA) -> dict:
    """Pick the length-normalized best beam from a candidate pool. Each candidate
    is a dict carrying at least `"score"` (cumulative sum log-prob) and
    `"length"` (token count); `"finished"` (bool) marks hypotheses that
    legitimately reached EOS. Finished candidates are preferred outright: an
    unfinished hypothesis is a truncated prefix (max_output_length hit), has
    paid for fewer tokens, and under length normalization can spuriously
    outrank a complete parse, so it is only eligible when NO hypothesis
    finished (the fallback that keeps truncated documents recoverable).
    Dividing by `length**alpha` mitigates the bias toward shorter beams (every
    emitted token has log-prob <= 0, so raw sum-log-prob monotonically favors
    fewer-token trajectories). `alpha=0.6` is the GNMT default. Caller must
    ensure `candidates` is non-empty."""
    finished = [c for c in candidates if c.get("finished", False)]
    pool = finished if finished else candidates
    return max(pool, key=lambda c: c["score"] / max(c["length"], 1) ** alpha)


# -----------------------------------------------------------------
# Shift-reduce decode state
# -----------------------------------------------------------------


@dataclass
class ShiftReduceDecodeState:
    """Bottom-up shift-reduce decode state for `gen`'s `sr` serialization, the
    shift-reduce analogue of the `sexp` serialization's `SexpDecodingState`.
    Vocab-agnostic: it tracks the source cursor, the constituent-stack size, and
    the current EDU's COPY count, exposing the four validity predicates and the
    four transitions that the greedy, beam, and gold-EDU loops share. The SR
    serialization maps the predicates into its scoring space and classifies
    emitted ids back into action kinds (`serializations/sr.py`), so the
    vocab-specific glue stays there while the automaton lives here.

    The state machine over actions {COPY, SHIFT, REDUCE, EOS}:
      COPY   advances the source cursor and extends the current EDU.
      SHIFT  commits the current EDU (records its `(start, cursor)` source-token
             range), pushes a leaf, and resets the EDU counter.
      REDUCE pops two constituents and pushes one.
      EOS    terminates.
    """

    source_len: int
    min_edu_length: int = 1
    cursor: int = 0
    stack_size: int = 0
    edu_length: int = 0
    edu_start: int = 0
    pred_edu_ranges: list[tuple[int, int]] = field(default_factory=list)
    done: bool = False
    # Multi-token relation-word labels (label_style='words'); empty in token mode,
    # where a REDUCE is a single action id the parser masks directly. Prefix-free.
    word_label_ids: frozenset = field(default_factory=frozenset)
    # In-progress label prefix; () when not mid-label. Words mode only. A REDUCE is
    # deferred until the label completes (see `words_step_full`).
    label_cursor: tuple = ()

    def clone(self) -> "ShiftReduceDecodeState":
        """Deep-enough copy for beam expansion (the only mutable field is the
        ranges list; `word_label_ids` is immutable and shared)."""
        return ShiftReduceDecodeState(
            source_len=self.source_len,
            min_edu_length=self.min_edu_length,
            cursor=self.cursor,
            stack_size=self.stack_size,
            edu_length=self.edu_length,
            edu_start=self.edu_start,
            pred_edu_ranges=list(self.pred_edu_ranges),
            done=self.done,
            word_label_ids=self.word_label_ids,
            label_cursor=self.label_cursor,
        )

    # ---- words mode (label_style='words'): multi-token relation-word labels ----

    def _label_conts(self) -> set[int]:
        n = len(self.label_cursor)
        return {seq[n] for seq in self.word_label_ids if len(seq) > n and seq[:n] == self.label_cursor}

    def words_step_full(self, full_id: int, copy_id: int, shift_id: int, end_id: int) -> str:
        """Advance on a full-vocab id (words mode). A multi-token label defers the
        REDUCE until the label completes; the label tokens themselves change no SR
        state. Returns the action kind so the loop can pick the next decoder input:
        'copy' | 'copy_exhausted' | 'shift' | 'label' | 'reduce' | 'eos' | 'illegal'."""
        if self.label_cursor:
            if full_id not in self._label_conts():
                self.done = True
                return "illegal"
            new = self.label_cursor + (full_id,)
            if new in self.word_label_ids:
                self.label_cursor = ()
                self.step_reduce()
                return "reduce"
            self.label_cursor = new
            return "label"
        if full_id == copy_id:
            return "copy" if self.step_copy() else "copy_exhausted"
        if full_id == shift_id:
            self.step_shift()
            return "shift"
        if full_id == end_id:
            self.step_eos()
            return "eos"
        if any(seq and seq[0] == full_id for seq in self.word_label_ids):
            self.label_cursor = (full_id,)
            return "label"
        self.done = True
        return "illegal"

    @property
    def at_end(self) -> bool:
        return self.cursor >= self.source_len

    @property
    def copy_ok(self) -> bool:
        return not self.at_end

    @property
    def shift_ok(self) -> bool:
        # At least `min_edu_length` COPYs, or end-of-source with any content so
        # the final EDU can still be committed.
        return self.edu_length >= self.min_edu_length or (self.at_end and self.edu_length >= 1)

    @property
    def reduce_ok(self) -> bool:
        return self.stack_size >= 2

    @property
    def eos_ok(self) -> bool:
        return self.at_end and self.stack_size == 1 and self.edu_length == 0

    def step_copy(self) -> bool:
        """Consume one source token. Returns False (and marks done) if the
        source is already exhausted, which the validity mask should prevent."""
        if self.cursor >= self.source_len:
            self.done = True
            return False
        self.cursor += 1
        self.edu_length += 1
        return True

    def step_shift(self) -> None:
        self.stack_size += 1
        self.pred_edu_ranges.append((self.edu_start, self.cursor))
        self.edu_start = self.cursor
        self.edu_length = 0

    def step_reduce(self) -> None:
        self.stack_size -= 1

    def step_eos(self) -> None:
        self.done = True


# -----------------------------------------------------------------
# SR tree reconstruction
# -----------------------------------------------------------------


def reconstruct_text(tree: RstTree) -> str:
    """Reverse the storage convention: join EDU strings with spaces (or each
    EDU's `prefix` field if populated, for detokenized corpora)."""
    parts: list[str] = []
    for i, edu in enumerate(tree.edus):
        if i == 0:
            parts.append(edu.text)
            continue
        prefix = edu.prefix if edu.prefix is not None else " "
        parts.append(prefix + edu.text)
    return "".join(parts)


def gold_edu_source_ranges(tokenizer, tree: RstTree) -> list[tuple[int, int]]:
    """Per-EDU `(start, end_exclusive)` token-position ranges in the source
    tokenizer's whole-doc tokenization space, tiling it exactly. Delegates to
    `align_edus_to_tokens` so train and predict agree on the tiling."""
    text = reconstruct_text(tree)
    _, spans = align_edus_to_tokens(tokenizer, text, tree.edus)
    return spans


def empty_tree(relation_types, text: str = "") -> RstTree:
    """Single-EDU fallback for empty / unrecoverable input. The text payload
    becomes one EDU so downstream callers (to_rs4_string, eval) work."""
    actions: list[ShiftReduceAction] = [Shift(edu_text=text or "")]
    return RstTree.from_shift_reduce(actions, relation_types=relation_types)


_CE_CHUNK_ROWS = 1024


class _ChunkedVocabCE(torch.autograd.Function):
    """Full-vocab cross-entropy over `logits[idx]` computed in row-chunks, so the fp32
    logits + log-softmax peak is `[chunk, V]` instead of `[len(idx), V]`. In words mode at
    27-31B the 262k-vocab CE over a long stream otherwise materializes multi-GB fp32
    tensors and OOMs (measured on gemma-4-31B sexp). Mathematically identical to
    `F.cross_entropy(logits[idx].float(), labels[idx], reduction='mean', label_smoothing=s)`:
    each chunk sums its loss, the total is divided by the row count, and the backward
    recomputes each chunk's gradient (a compute-for-memory trade). Only the scored rows get
    gradient; the rest of `logits.grad` stays zero (matching ignore_index)."""

    @staticmethod
    def forward(ctx, logits, labels, idx, label_smoothing, chunk):
        ctx.save_for_backward(logits, labels, idx)
        ctx.label_smoothing = float(label_smoothing)
        ctx.chunk = int(chunk)
        n = int(idx.numel())
        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        for s in range(0, n, ctx.chunk):
            rows = idx[s : s + ctx.chunk]
            lg = logits.index_select(0, rows).float()
            total = total + F.cross_entropy(
                lg, labels.index_select(0, rows), label_smoothing=ctx.label_smoothing, reduction="sum"
            )
        return total / max(n, 1)

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, idx = ctx.saved_tensors
        n = int(idx.numel())
        scale = grad_output.detach().to(torch.float32) / max(n, 1)
        grad = torch.zeros_like(logits)
        for s in range(0, n, ctx.chunk):
            rows = idx[s : s + ctx.chunk]
            lg = logits.index_select(0, rows).float().requires_grad_(True)
            with torch.enable_grad():
                loss = F.cross_entropy(
                    lg, labels.index_select(0, rows), label_smoothing=ctx.label_smoothing, reduction="sum"
                )
            (g,) = torch.autograd.grad(loss, lg)
            grad.index_copy_(0, rows, (g * scale).to(grad.dtype))
        return grad, None, None, None, None


def chunked_cross_entropy(logits, labels, idx, *, label_smoothing: float = 0.0, chunk: int = _CE_CHUNK_ROWS):
    """Memory-bounded mean cross-entropy over the `idx` rows of `logits` (`[M, V]`) against
    `labels` (`[M]`). See `_ChunkedVocabCE`. Empty `idx` returns 0 (no scored positions)."""
    if idx.numel() == 0:
        return logits.new_zeros((), dtype=torch.float32)
    return _ChunkedVocabCE.apply(logits, labels, idx, label_smoothing, chunk)
