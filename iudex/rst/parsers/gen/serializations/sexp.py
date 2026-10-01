"""S-expression serialization: the tree is a nested `( label child child )`
linearization. Three scoring modes: `use_copy=True` token (small action head over
`{copy, open, close, labels, eos}`), `use_copy=True` words (relation words + literal
` {` / ` }` brackets over the full lm_head), and `use_copy=False` (Hu & Wan full-vocab
content, full lm_head). Decode is driven by the shared SexpDecodingState PDA +
GoldEduForcer.

The PDA is immutable (`step()` returns a fresh state, raising on an illegal action)
and tracks no source ranges, so the decode loops drive it through a small mutable
`SexpDecodeState` wrapper that also carries the in-loop EDU-range bookkeeping and
(for gold-EDU decode) the GoldEduForcer. That wrapper is what `initial_state` /
`apply` / `clone` operate on, matching the SR path's mutable-state contract.
"""

import torch

from iudex.common.log import warn
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.common.seqgen import (
    align_edus_to_tokens,
    build_word_label_vocab,
    empty_tree,
    reconstruct_text,
)
from iudex.rst.parsers.common.sexp_constraints import GoldEduForcer, SexpDecodingState
from iudex.rst.parsers.gen.errors import DecodeInvariantError
from iudex.rst.parsers.gen.serializations.base import Serialization


class SexpDecodeState:
    """Mutable decode-loop state wrapping the immutable `SexpDecodingState` PDA.

    Holds the PDA state (`inner`, replaced on each step), the in-loop EDU-range
    tracking the PDA doesn't do (`leaf_start` / `pred_edu_ranges`), a `done` flag,
    and an optional `GoldEduForcer` for gold-EDU forced decode. `clone` deep-copies
    the mutable pieces (and the forcer) so beam siblings never alias; the inner PDA
    is immutable and shared."""

    __slots__ = ("inner", "forcer", "pred_edu_ranges", "leaf_start", "done")

    def __init__(self, inner, forcer=None):
        self.inner = inner
        self.forcer = forcer
        self.pred_edu_ranges: list[tuple[int, int]] = []
        self.leaf_start: int | None = None
        self.done = False

    def clone(self) -> "SexpDecodeState":
        new = SexpDecodeState(self.inner, self.forcer.clone() if self.forcer is not None else None)
        new.pred_edu_ranges = list(self.pred_edu_ranges)
        new.leaf_start = self.leaf_start
        new.done = self.done
        return new


def _dedup_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in ranges:
        if not out or out[-1] != r:
            out.append(r)
    return out


class SexpSerialization(Serialization):
    OPEN_TOKEN = "<sexp_open>"
    CLOSE_TOKEN = "<sexp_close>"
    COPY_TOKEN = "<copy>"
    # words mode delimits every relation-word label so the label set is prefix-free.
    LABEL_END_TOKEN = "<label_end>"

    def __init__(self, config):
        super().__init__(config)
        self._words_mode = config.label_style == "words"
        self.label_end_id: int | None = None
        # scoring vocab is the small head only for use_copy=True token mode; words
        # and use_copy=False keep the full pretrained lm_head.
        self._small_head = config.use_copy and not self._words_mode

        self.stream_end_id: int | None = None
        self.open_token_id: int | None = None
        self.close_token_id: int | None = None
        self.copy_token_id: int | None = None
        self.label_token_ids: dict[str, int] = {}
        self.label_token_map: dict[str, tuple[str, str]] = {}
        self.label_id_set: set[int] = set()
        self.label_id_to_str: dict[int, str] = {}
        self.full_id_for_head_idx: list[int] = []
        self.head_idx_for_full_id: dict[int, int] = {}
        self.head_vocab_size: int | None = None
        self.word_label_to_ids: dict[tuple[str, str], tuple[int, ...]] = {}
        self.word_ids_to_label: dict[tuple[int, ...], tuple[str, str]] = {}
        self.word_label_token_ids: set[int] = set()

    @property
    def scores_separator(self) -> bool:
        # sexp scores the SEP position (the transition out of the source prefix),
        # but only when the scoring head can represent the SEP id: the small
        # token-mode head has no SEP slot, so there its target would just remap to
        # ignore_index in loss_terms. Full-head modes (words / use_copy=False)
        # score it for real.
        return not self.uses_small_head()

    # ---- construction ----

    def uses_small_head(self) -> bool:
        return self._small_head

    def _build_label_vocab(self) -> list[str]:
        labels: list[str] = []
        self.label_token_map = {}
        for rel, kind in self.config.relation_types:
            nucs = ("NN",) if kind == "multinuc" else ("NS", "SN")
            for nuc in nucs:
                token = f"<{nuc}:{rel}>"
                labels.append(token)
                self.label_token_map[token] = (nuc, rel)
        return labels

    def action_tokens(self) -> list[str]:
        """Specials added besides the backbone's stream separator. words mode adds
        only `<copy>` (brackets are literal existing tokens, labels are relation
        words); token/use_copy modes add open/close (+copy) and the fused labels."""
        if self._words_mode:
            return [self.COPY_TOKEN, self.LABEL_END_TOKEN]
        brackets = [self.OPEN_TOKEN, self.CLOSE_TOKEN]
        labels = self._build_label_vocab()
        if not self.config.use_copy:
            return brackets + labels
        # The two hand-written sexp parsers drifted on where <copy> was added, so its
        # new-token id differs by backbone: decoder_only added it right after the
        # brackets (before the labels), seq2seq added it last (after the labels).
        # Reproduce both exactly so a legacy .pt's added-token ids line up. The
        # backbone discriminator is the only place this token-order accident shows.
        if self.config.backbone == "seq2seq":
            return brackets + labels + [self.COPY_TOKEN]
        return brackets + [self.COPY_TOKEN] + labels

    def _single_token_id(self, s: str) -> int:
        ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(
                f"label_style='words' expects {s!r} to be a single token; got {ids}. "
                f"This backbone splits the structural bracket; words mode needs a single-token ' {{' / ' }}'."
            )
        return int(ids[0])

    def build_vocab(self, stream_end_id: int) -> None:
        self.stream_end_id = int(stream_end_id)
        if self._words_mode:
            self._build_vocab_words()
        else:
            self._build_vocab_token(stream_end_id)

    def _build_vocab_token(self, stream_end_id: int) -> None:
        tok = self.tokenizer
        self.open_token_id = int(tok.convert_tokens_to_ids(self.OPEN_TOKEN))
        self.close_token_id = int(tok.convert_tokens_to_ids(self.CLOSE_TOKEN))
        if self.config.use_copy:
            self.copy_token_id = int(tok.convert_tokens_to_ids(self.COPY_TOKEN))
        self.label_token_ids = {t: int(tok.convert_tokens_to_ids(t)) for t in self.label_token_map}
        self.label_id_set = set(self.label_token_ids.values())
        self.label_id_to_str = {tid: t for t, tid in self.label_token_ids.items()}

        if self.config.use_copy:
            labels_sorted = sorted(self.label_id_set)
            # Same historical drift as action_tokens' <copy> placement, mirrored in
            # the small-head layout: decoder_only ordered it [copy, open, close,
            # labels, eos], seq2seq [open, close, labels, eos, copy]. The head rows
            # load positionally from a legacy .pt, so gen must match the layout the
            # checkpoint was trained with (else identical rows mean permuted tokens).
            if self.config.backbone == "seq2seq":
                self.full_id_for_head_idx = [
                    self.open_token_id, self.close_token_id, *labels_sorted, stream_end_id, self.copy_token_id,
                ]
            else:
                self.full_id_for_head_idx = [
                    self.copy_token_id, self.open_token_id, self.close_token_id, *labels_sorted, stream_end_id,
                ]
            self.head_idx_for_full_id = {fid: i for i, fid in enumerate(self.full_id_for_head_idx)}
            self.head_vocab_size = len(self.full_id_for_head_idx)
            self.copy_head_idx = self.head_idx_for_full_id[self.copy_token_id]
            self.open_head_idx = self.head_idx_for_full_id[self.open_token_id]
            self.close_head_idx = self.head_idx_for_full_id[self.close_token_id]
            self.eos_head_idx = self.head_idx_for_full_id[stream_end_id]
            self.label_head_indices = {self.head_idx_for_full_id[fid] for fid in self.label_id_set}

            self._register_head_buffers(self.label_head_indices | {self.open_head_idx, self.close_head_idx})
        else:
            # use_copy=False: full lm_head, structural ids in full-vocab space
            # (used by the action_loss_weight rebalance in loss_terms).
            self.head_vocab_size = int(len(tok))
            self._register_structural_full_ids(self.label_id_set | {self.open_token_id, self.close_token_id})

    def _build_vocab_words(self) -> None:
        tok = self.tokenizer
        self.copy_token_id = int(tok.convert_tokens_to_ids(self.COPY_TOKEN))
        self.label_end_id = int(tok.convert_tokens_to_ids(self.LABEL_END_TOKEN))
        # Structural brackets are literal single-token ' {' / ' }' (pretrained).
        self.open_token_id = self._single_token_id(" {")
        self.close_token_id = self._single_token_id(" }")
        self.word_label_to_ids, self.word_ids_to_label = build_word_label_vocab(
            tok, self.config.relation_types, terminator_id=self.label_end_id
        )
        self.label_id_set = set()
        self.label_token_map = {}
        self.label_id_to_str = {}
        self.head_vocab_size = int(len(tok))
        self.full_id_for_head_idx = []
        self.head_idx_for_full_id = {}
        self.word_label_token_ids = {int(t) for ids in self.word_label_to_ids.values() for t in ids}
        self._register_structural_full_ids(self.word_label_token_ids | {self.open_token_id, self.close_token_id})

    def head_spec(self) -> dict:
        # sexp warm-inits its small head from the input embedding (SR uses the old
        # lm_head); see DecoderOnlyBackbone.install_head.
        return {
            "head_vocab_size": self.head_vocab_size,
            "full_id_for_head_idx": self.full_id_for_head_idx,
            "warm_init_source": "input_embedding",
        }

    # ---- linearize (loss_terms is inherited from the base) ----

    def linearize(self, tree):
        """tree -> (source_ids, seen_tokens, label_tokens) for the sexp body,
        WITHOUT the terminator (the backbone appends it and handles the SEP). The
        SEP-scoring difference vs SR is carried by `scores_separator`."""
        source_ids, edu_subword_ids = self._edu_subword_ids(tree)
        seen, label = self._build_sexp_tokens(tree, edu_subword_ids)
        return source_ids, seen, label

    def _edu_subword_ids(self, tree):
        text = reconstruct_text(tree)
        source_ids, spans = align_edus_to_tokens(self.tokenizer, text, tree.edus)
        edu_subword_ids = [list(source_ids[s:e]) for s, e in spans]
        return source_ids, edu_subword_ids

    def _build_sexp_tokens(self, tree, edu_subword_ids):
        """(seen_ids, label_ids) for the sexp body. seen substitutes source
        subwords at COPY positions (use_copy=True); label keeps the `<copy>`
        sentinel. Nesting/label order follows `traversal_order`."""
        traversal = self.config.traversal_order
        binary = tree._build_binary_tree()
        seen: list[int] = []
        labels: list[int] = []

        def render(node, edu_idx):
            if node[0] == "edu":
                idx = edu_idx[0]
                edu_idx[0] += 1
                seen.append(self.open_token_id)
                labels.append(self.open_token_id)
                for sub in edu_subword_ids[idx]:
                    if self.config.use_copy:
                        seen.append(sub)
                        labels.append(self.copy_token_id)
                    else:
                        seen.append(sub)
                        labels.append(sub)
                seen.append(self.close_token_id)
                labels.append(self.close_token_id)
                return
            _, nuc, rel, left, right = node
            if self._words_mode:
                label_seq = self.word_label_to_ids.get((nuc, rel))
                if label_seq is None:
                    raise ValueError(f"_build_sexp_tokens: label {(nuc, rel)} not in the word-label vocab.")
            else:
                label_str = f"<{nuc}:{rel}>"
                if label_str not in self.label_token_ids:
                    raise ValueError(f"_build_sexp_tokens: label {label_str!r} not in the label vocab.")
                label_seq = (self.label_token_ids[label_str],)
            seen.append(self.open_token_id)
            labels.append(self.open_token_id)
            if traversal == "preorder":
                seen.extend(label_seq)
                labels.extend(label_seq)
                render(left, edu_idx)
                render(right, edu_idx)
            else:
                render(left, edu_idx)
                render(right, edu_idx)
                seen.extend(label_seq)
                labels.extend(label_seq)
            seen.append(self.close_token_id)
            labels.append(self.close_token_id)

        render(binary, [0])
        return seen, labels

    # ---- decode ----

    def _make_inner_state(self, source_ids, *, min_edu_length: int, blank_positions=frozenset()) -> SexpDecodingState:
        # Content is always constrained to the source cursor (via the PDA's
        # `_content_legal`): free-content generation is not a supported gen mode
        # (it is the only way a decode could emit a sexp `from_sexp` rejects,
        # see `build_tree`).
        return SexpDecodingState(
            source_len=len(source_ids),
            traversal_order=self.config.traversal_order,
            use_copy=self.config.use_copy,
            open_id=self.open_token_id,
            close_id=self.close_token_id,
            eos_id=self.stream_end_id,
            label_ids=frozenset(self.label_id_set),
            word_label_ids=frozenset(self.word_label_to_ids.values()) if self._words_mode else frozenset(),
            copy_id=self.copy_token_id if self.config.use_copy else None,
            source_ids=tuple() if self.config.use_copy else tuple(source_ids),
            min_edu_length=int(min_edu_length),
            blank_positions=blank_positions,
        )

    def initial_state(self, source_ids):
        return SexpDecodeState(
            self._make_inner_state(
                source_ids, min_edu_length=self.config.min_edu_length, blank_positions=self.blank_positions(source_ids)
            )
        )

    def gold_initial_state(self, source_ids, gold_ranges):
        # Gold forcing pins min_edu_length=1 (the forcer assumes it; see GoldEduForcer).
        inner = self._make_inner_state(source_ids, min_edu_length=1)
        forcer = GoldEduForcer(len(gold_ranges), [tuple(r) for r in gold_ranges])
        return SexpDecodeState(inner, forcer=forcer)

    def decode_id(self, idx: int) -> int:
        return self.full_id_for_head_idx[idx] if self.uses_small_head() else idx

    def _legal_bool(self, inner, vocab_size: int) -> torch.Tensor:
        """Bool mask over the scoring vocab of the PDA's legal actions at `inner`.
        Token mode maps legal full-vocab ids through the head layout; the full-head
        modes place legal ids directly."""
        legal = inner.legal_actions()
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        if self._small_head:
            for full_id in legal:
                hi = self.head_idx_for_full_id.get(int(full_id))
                if hi is not None:
                    mask[hi] = True
            return mask
        for full_id in legal:
            fid = int(full_id)
            if 0 <= fid < vocab_size:
                mask[fid] = True
        return mask

    def _narrowed_bool(self, narrowed, base_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Materialize a GoldEduForcer narrowing onto the base legal mask (see
        `GoldEduForcer.narrowed_legal`: None | frozenset). None keeps the base; a
        frozenset intersects the base with those ids (a singleton is a hard force,
        and it is always in the base)."""
        if isinstance(narrowed, frozenset):
            keep = torch.zeros(vocab_size, dtype=torch.bool)
            for fid in narrowed:
                hi = self.head_idx_for_full_id.get(int(fid)) if self._small_head else int(fid)
                if hi is not None and 0 <= int(hi) < vocab_size:
                    keep[int(hi)] = True
            return base_mask & keep
        return base_mask

    def pred_mask(self, st, vocab_size: int) -> torch.Tensor:
        return self._legal_bool(st.inner, vocab_size)

    def gold_mask_source(self, gold_ranges, source_len: int):
        # sexp gold forcing is stateful: the forcer rides on the decode state (so
        # `apply` can `observe` it and beams can clone it), and the narrowing is a
        # read of `st.forcer`. gold_ranges/source_len are baked into the forcer at
        # gold_initial_state and unused here.
        def mask(st, vocab_size):
            base = self._legal_bool(st.inner, vocab_size)
            narrowed = st.forcer.narrowed_legal(st.inner)
            return self._narrowed_bool(narrowed, base, vocab_size)

        return mask

    def apply(self, st, full_id: int, source_ids):
        """Advance the PDA on one emitted id (replacing `st.inner`), tracking the
        EDU-range bookkeeping and advancing the gold forcer. Returns
        `(next_input_or_None, kind)`; `kind == "illegal"` on a PDA-rejected action."""
        inner = st.inner
        closing_leaf = full_id == self.close_token_id and bool(inner.stack) and inner.stack[-1].kind == "leaf"
        pre_cursor = inner.cursor
        try:
            new_inner = inner.step(full_id)
        except ValueError:
            st.done = True
            return None, "illegal"
        if st.forcer is not None:
            st.forcer.observe(inner, new_inner, full_id)
        st.inner = new_inner
        if full_id == self.stream_end_id:
            st.done = True
            return None, "eos"
        # Leaf-range bookkeeping (cursor advances only on content tokens).
        if new_inner.cursor > pre_cursor and st.leaf_start is None:
            st.leaf_start = pre_cursor
        if closing_leaf and st.leaf_start is not None:
            st.pred_edu_ranges.append((st.leaf_start, new_inner.cursor))
            st.leaf_start = None
        # Next model input: COPY feeds back the source subword it stands for.
        if self.config.use_copy and full_id == self.copy_token_id:
            src_pos = new_inner.cursor - 1
            next_input = source_ids[src_pos] if 0 <= src_pos < len(source_ids) else full_id
        else:
            next_input = full_id
        return next_input, "step"

    def candidate_ranges(self, st, finished: bool) -> list:
        # sexp leaves close explicitly (each commits its range in `apply`), so there
        # is no trailing partial to add; just dedup any double-counted adjacencies.
        return _dedup_ranges(st.pred_edu_ranges)

    def stash_meta(self, tree, ranges, source_ids) -> None:
        # A degraded (pathologically-deep) tree has one EDU, so the action-tracked
        # ranges would disagree with it; null them.
        failed = getattr(tree, "_from_sexp_failed", False)
        tree._pred_edu_source_ranges = [] if failed else ranges  # type: ignore[attr-defined]
        tree._source_ids = source_ids  # type: ignore[attr-defined]

    def build_tree(self, action_ids, source_ids):
        """Stringify the emitted action ids and call `RstTree.from_sexp`. Raises
        `DecodeInvariantError` on malformed output: the PDA constrains structure and
        content is pinned to the source cursor (and `(`/`)` in it are escaped), so a
        sexp `from_sexp` rejects means the constraints are broken, not the model."""
        eos_id = self.stream_end_id
        parts: list[str] = []
        leaf_buf: list[int] = []
        label_buf: list[int] = []  # in-progress multi-token label ids (words mode)
        cursor = 0
        tok = self.tokenizer

        def flush_leaf():
            if leaf_buf:
                decoded = tok.decode(leaf_buf, skip_special_tokens=False).strip()
                if decoded:
                    parts.append(decoded.replace("(", "-LRB-").replace(")", "-RRB-"))
                leaf_buf.clear()

        for t in action_ids:
            if t == eos_id:
                break
            if t == self.open_token_id:
                flush_leaf()
                parts.append("(")
                continue
            if t == self.close_token_id:
                flush_leaf()
                parts.append(")")
                continue
            if t in self.label_id_set:
                flush_leaf()
                parts.append(self.label_id_to_str[t][1:-1])  # strip the angle brackets
                continue
            if self._words_mode and t in self.word_label_token_ids:
                label_buf.append(t)
                label = self.word_ids_to_label.get(tuple(label_buf))
                if label is not None:
                    flush_leaf()
                    parts.append(f"{label[0]}:{label[1]}")
                    label_buf.clear()
                continue
            if self.config.use_copy and t == self.copy_token_id:
                if cursor < len(source_ids):
                    leaf_buf.append(source_ids[cursor])
                    cursor += 1
                continue
            # use_copy=False source token
            leaf_buf.append(t)
        flush_leaf()

        sexp_text = " ".join(parts)
        try:
            return RstTree.from_sexp(
                sexp_text,
                traversal_order=self.config.traversal_order,
                relation_types=self.config.relation_types,
            )
        except RecursionError:
            # Legal but pathologically deep (see sr.py's build_tree): a bad parse, not
            # a bug. Degrade so it scores near zero instead of crashing the eval.
            warn(f"Pathologically deep sexp ({len(action_ids)} actions). Degrading to a single-EDU tree.")
            full_text = tok.decode(source_ids, skip_special_tokens=True)
            tree = empty_tree(self.config.relation_types, text=full_text)
            tree._from_sexp_failed = True  # type: ignore[attr-defined]
            return tree
        except Exception as e:
            raise DecodeInvariantError(
                f"from_sexp rejected the decoded action stream ({type(e).__name__}: {e}). "
                f"The PDA admits only well-formed sexps, so this means the constraints and "
                f"the reconstruction disagree. source_len={len(source_ids)}, "
                f"actions={len(action_ids)}, sexp={sexp_text[:200]!r}"
            ) from e
