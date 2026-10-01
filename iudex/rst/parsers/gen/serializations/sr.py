"""Shift-reduce serialization: a bottom-up SHIFT / REDUCE action sequence. Content
is always copied via the `<copy>` sentinel (use_copy is implicitly True). Two label
styles: `token` (one fused `<reduce_*>` id per merge over a small action head) and
`words` (relation words over the full pretrained lm_head).
"""

import torch

from iudex.common.log import warn
from iudex.rst.data.tree import Reduce, RstTree, Shift, strings_to_actions
from iudex.rst.parsers.common.seqgen import (
    ShiftReduceDecodeState,
    align_edus_to_tokens,
    build_word_label_vocab,
    empty_tree,
    reconstruct_text,
)
from iudex.rst.parsers.gen.errors import DecodeInvariantError
from iudex.rst.parsers.gen.serializations.base import Serialization


class SRSerialization(Serialization):
    COPY_TOKEN = "<copy>"
    # words mode delimits every relation-word label with this terminator so the
    # label set is prefix-free (see build_word_label_vocab); token mode never uses it.
    LABEL_END_TOKEN = "<label_end>"

    def __init__(self, config):
        super().__init__(config)
        self._words_mode = config.label_style == "words"
        self.label_end_id: int | None = None
        # Scalar vocab state (populated by build_vocab). Tensor buffers that need
        # device movement are registered on this module (see build_vocab).
        self.stream_end_id: int | None = None
        self.copy_token_id: int | None = None
        self.shift_token_id: int | None = None
        self.reduce_token_ids: set[int] = set()
        self.reduce_token_map: dict[str, tuple[str, str]] = {}
        self.reduce_token_to_id: dict[str, int] = {}
        self.full_id_for_head_idx: list[int] = []
        self.head_idx_for_full_id: dict[int, int] = {}
        self.head_vocab_size: int | None = None
        self.word_label_to_ids: dict[tuple[str, str], tuple[int, ...]] = {}
        self.word_ids_to_label: dict[tuple[int, ...], tuple[str, str]] = {}
        self.word_label_token_ids: set[int] = set()

    # ---- construction ----

    def uses_small_head(self) -> bool:
        return not self._words_mode

    def action_tokens(self) -> list[str]:
        shift = Shift().to_token()
        if self._words_mode:
            return [self.COPY_TOKEN, shift, self.LABEL_END_TOKEN]
        reduces: list[str] = []
        self.reduce_token_map = {}
        for rel, kind in self.config.relation_types:
            nucs = ("NN",) if kind == "multinuc" else ("NS", "SN")
            for nuc in nucs:
                tok = Reduce(nuc=nuc, rel=rel).to_token()
                reduces.append(tok)
                self.reduce_token_map[tok] = (nuc, rel)
        return [self.COPY_TOKEN, shift] + reduces

    def build_vocab(self, stream_end_id: int) -> None:
        tok = self.tokenizer
        self.stream_end_id = int(stream_end_id)  # backbone's terminator; used at decode
        self.copy_token_id = int(tok.convert_tokens_to_ids(self.COPY_TOKEN))
        self.shift_token_id = int(tok.convert_tokens_to_ids(Shift().to_token()))
        if self._words_mode:
            self._build_vocab_words()
        else:
            self._build_vocab_token(stream_end_id)

    def _build_vocab_token(self, stream_end_id: int) -> None:
        tok = self.tokenizer
        self.reduce_token_ids = {int(tok.convert_tokens_to_ids(t)) for t in self.reduce_token_map}
        self.reduce_token_to_id = {t: int(tok.convert_tokens_to_ids(t)) for t in self.reduce_token_map}
        # Small action head ordering: [copy, shift, sorted(reduces), stream_end].
        self.full_id_for_head_idx = [
            self.copy_token_id,
            self.shift_token_id,
            *sorted(self.reduce_token_ids),
            stream_end_id,
        ]
        self.head_idx_for_full_id = {fid: i for i, fid in enumerate(self.full_id_for_head_idx)}
        self.head_vocab_size = len(self.full_id_for_head_idx)
        self.copy_head_idx = self.head_idx_for_full_id[self.copy_token_id]
        self.shift_head_idx = self.head_idx_for_full_id[self.shift_token_id]
        self.eos_head_idx = self.head_idx_for_full_id[stream_end_id]
        self.reduce_head_indices = {self.head_idx_for_full_id[fid] for fid in self.reduce_token_ids}

        self._register_head_buffers(self.reduce_head_indices | {self.shift_head_idx})

    def _build_vocab_words(self) -> None:
        tok = self.tokenizer
        self.label_end_id = int(tok.convert_tokens_to_ids(self.LABEL_END_TOKEN))
        self.word_label_to_ids, self.word_ids_to_label = build_word_label_vocab(
            tok, self.config.relation_types, terminator_id=self.label_end_id
        )
        self.reduce_token_map = {
            Reduce(nuc=nuc, rel=rel).to_token(): (nuc, rel) for (nuc, rel) in self.word_label_to_ids
        }
        self.reduce_token_ids = set()
        self.full_id_for_head_idx = []
        self.head_idx_for_full_id = {}
        self.head_vocab_size = int(len(tok))
        self.word_label_token_ids = {int(t) for ids in self.word_label_to_ids.values() for t in ids}
        self._register_structural_full_ids(self.word_label_token_ids | {self.shift_token_id})

    def head_spec(self) -> dict:
        return {"head_vocab_size": self.head_vocab_size, "full_id_for_head_idx": self.full_id_for_head_idx}

    # ---- linearize + loss ----

    def linearize(self, tree):
        """tree -> (source_ids, seen_tokens, label_tokens), WITHOUT the stream
        terminator (the backbone appends it). `seen_tokens` substitutes the real
        source subword at every COPY position (what the model sees); `label_tokens`
        keeps the `<copy>` sentinel as the prediction target. Lengths agree."""
        if self.copy_token_id is None:
            raise RuntimeError("linearize called before build_vocab (cfg.relation_types unset?)")
        text = reconstruct_text(tree)
        source_ids, spans = align_edus_to_tokens(self.tokenizer, text, tree.edus)
        edu_subword_ids = [list(source_ids[s:e]) for s, e in spans]
        seen: list[int] = []
        label: list[int] = []
        edu_idx = 0
        for action in tree.to_shift_reduce(include_text=False):
            if isinstance(action, Shift):
                for src_id in edu_subword_ids[edu_idx]:
                    label.append(self.copy_token_id)
                    seen.append(src_id)
                label.append(self.shift_token_id)
                seen.append(self.shift_token_id)
                edu_idx += 1
            elif isinstance(action, Reduce):
                if self._words_mode:
                    seq = self.word_label_to_ids.get((action.nuc, action.rel))
                    if seq is None:
                        raise ValueError(f"linearize: Reduce {action!r} missing from the word-label vocab.")
                    label.extend(seq)
                    seen.extend(seq)
                else:
                    token_str = action.to_token()
                    tok_id = self.reduce_token_to_id.get(token_str)
                    if tok_id is None:
                        raise ValueError(f"linearize: Reduce {action!r} not in the action vocabulary.")
                    label.append(tok_id)
                    seen.append(tok_id)
        return source_ids, seen, label

    # loss_terms is shared (identical across serializations): see Serialization base.

    # ---- decode ----

    @property
    def _word_labels(self) -> frozenset:
        """The label trie (label token-id tuples) for a words-mode decode state,
        empty in token mode (a REDUCE is a single masked action id)."""
        return frozenset(self.word_label_to_ids.values()) if self._words_mode else frozenset()

    def initial_state(self, source_ids):
        min_edu_len = max(1, int(self.config.min_edu_length))
        return ShiftReduceDecodeState(
            source_len=len(source_ids),
            min_edu_length=min_edu_len,
            word_label_ids=self._word_labels,
            blank_positions=self.blank_positions(source_ids),
        )

    def _set_to_mask(self, ids, vocab_size: int) -> torch.Tensor:
        """A set of scoring-vocab indices (head indices in token mode, full-vocab
        ids in words mode) as a bool mask the decode loops apply."""
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for i in ids:
            ii = int(i)
            if 0 <= ii < vocab_size:
                mask[ii] = True
        return mask

    def pred_mask(self, st, vocab_size: int) -> torch.Tensor:
        return self._set_to_mask(self.legal_ids(st), vocab_size)

    def gold_mask_source(self, gold_ranges, source_len: int):
        # SR gold forcing is a pure function of (state, EDU ends): force COPY inside
        # an EDU and SHIFT at its boundary. The forcing lives in the mask, so the
        # gold state is just the pred state (see gold_initial_state's base default).
        edu_ends = [e for _, e in gold_ranges]
        return lambda st, vocab_size: self._set_to_mask(
            self.gold_legal_ids(st, edu_ends, source_len), vocab_size
        )

    def _ids_for_kinds(self, kinds: set, st) -> set:
        """Map abstract action kinds ({'copy', 'shift', 'reduce', 'eos'}) into the
        scoring space: head indices in token mode, full-vocab ids in words mode
        (a REDUCE there is the label trie's first tokens)."""
        small = self.uses_small_head()
        ids: set = set()
        if "copy" in kinds:
            ids.add(self.copy_head_idx if small else self.copy_token_id)
        if "shift" in kinds:
            ids.add(self.shift_head_idx if small else self.shift_token_id)
        if "reduce" in kinds:
            ids.update(self.reduce_head_indices if small else {seq[0] for seq in st.word_label_ids})
        if "eos" in kinds:
            ids.add(self.eos_head_idx if small else self.stream_end_id)
        return ids

    def legal_ids(self, st) -> set:
        """Validity-constrained legal ids for pred-EDU decoding, in the scoring
        space. Mid-label (words mode only) the sole legal ids are the label trie's
        continuations; otherwise the state's four predicates decide."""
        if st.label_cursor:
            return st._label_conts()
        kinds = set()
        if st.copy_ok:
            kinds.add("copy")
        if st.shift_ok:
            kinds.add("shift")
        if st.reduce_ok:
            kinds.add("reduce")
        if st.eos_ok:
            kinds.add("eos")
        return self._ids_for_kinds(kinds, st)

    def gold_legal_ids(self, st, edu_ends: list[int], source_len: int) -> set:
        """Gold-EDU-forced legal ids: force COPY inside an EDU and SHIFT at its
        boundary, leaving REDUCE / EOS model-driven. Which EDU we're on is
        `len(st.pred_edu_ranges)` (step_shift appends one range per shift), so no
        separate cursor is threaded."""
        if st.label_cursor:
            return st._label_conts()
        edu_idx = len(st.pred_edu_ranges)
        more_edus = edu_idx < len(edu_ends)
        current_end = edu_ends[edu_idx] if more_edus else source_len
        kinds = set()
        if more_edus and st.edu_length > 0 and st.cursor < current_end:
            kinds.add("copy")
        elif more_edus and st.cursor >= current_end:
            kinds.add("shift")
        else:
            if st.stack_size >= 2:
                kinds.add("reduce")
            if more_edus:
                kinds.add("copy")
            elif st.stack_size == 1:
                kinds.add("eos")
        return self._ids_for_kinds(kinds, st)

    def decode_id(self, idx: int) -> int:
        """Map an argmax index in the scoring space to a full-vocab id."""
        return self.full_id_for_head_idx[idx] if self.uses_small_head() else idx

    def apply(self, st, full_id: int, source_ids: list[int]):
        """Advance the shift-reduce state on one emitted id. Returns
        (next_decoder_input_or_None, kind); next_input None means stop. In words
        mode a REDUCE spans several 'label' steps and completes on the final one."""
        end_id = self.stream_end_id
        if self._words_mode:
            kind = st.words_step_full(full_id, self.copy_token_id, self.shift_token_id, end_id)
            if kind == "copy":
                return source_ids[st.cursor - 1], kind
            if kind in ("shift", "label", "reduce"):
                return full_id, kind
            return None, kind
        if full_id == end_id:
            st.step_eos()
            return None, "eos"
        if full_id == self.copy_token_id:
            return (source_ids[st.cursor - 1], "copy") if st.step_copy() else (None, "copy_exhausted")
        if full_id == self.shift_token_id:
            st.step_shift()
            return full_id, "shift"
        if full_id in self.reduce_token_ids:
            st.step_reduce()
            return full_id, "reduce"
        st.done = True
        return None, "illegal"

    def candidate_ranges(self, st, finished: bool) -> list:
        """The predicted EDU source-position ranges. A finished decode has shifted
        every EDU (each shift committed its range), so no trailing partial; an
        unfinished one (length cap / dead beam) commits the in-progress EDU."""
        ranges = list(st.pred_edu_ranges)
        if not finished and st.cursor > st.edu_start:
            ranges.append((st.edu_start, st.cursor))
        return ranges

    def stash_meta(self, tree, ranges, source_ids) -> None:
        tree._pred_edu_source_ranges = ranges  # type: ignore[attr-defined]
        tree._source_ids = source_ids  # type: ignore[attr-defined]

    def build_tree(self, action_ids: list[int], source_ids: list[int]):
        strings: list[str] = []
        source_buffer: list[int] = []
        label_buf: list[int] = []  # in-progress multi-token label ids (words mode)
        cursor = 0
        end_id = self.stream_end_id
        tok = self.tokenizer

        def flush_source():
            if source_buffer:
                decoded = tok.decode(source_buffer, skip_special_tokens=False)
                strings.extend(decoded.split())
                source_buffer.clear()

        for t in action_ids:
            if t == end_id:
                flush_source()
                break
            if t == self.copy_token_id:
                if cursor < len(source_ids):
                    source_buffer.append(source_ids[cursor])
                    cursor += 1
            elif t == self.shift_token_id:
                flush_source()
                strings.append(Shift().to_token())
            elif self._words_mode and t in self.word_label_token_ids:
                label_buf.append(t)
                label = self.word_ids_to_label.get(tuple(label_buf))
                if label is not None:
                    flush_source()
                    strings.append(Reduce(nuc=label[0], rel=label[1]).to_token())
                    label_buf.clear()
            elif t in self.reduce_token_ids:
                flush_source()
                strings.append(tok.convert_ids_to_tokens(t))
        flush_source()

        # Decoding is always constrained, so the stream is well-formed by
        # construction (words-mode labels are prefix-free via <label_end>); any
        # parse failure here means the constraints and the reconstruction
        # disagree, so both converters raise through DecodeInvariantError below.
        try:
            actions = strings_to_actions(strings, self.reduce_token_map)
            return RstTree.from_shift_reduce(actions, relation_types=self.config.relation_types)
        except RecursionError:
            # A pathologically deep but entirely LEGAL sequence (an undertrained model
            # decoding a long doc as hundreds of single-token EDUs + a linear reduce
            # chain) recurses past CPython's frame limit in the tree builders. The mask
            # and the automaton agree here -- this is a bad parse, not a bug -- so
            # degrade to a single-EDU tree and let it score near zero.
            # See tests/test_sr_deep_tree_degrades.py.
            warn(f"Pathologically deep shift-reduce tree ({len(strings)} strings). Degrading to a single-EDU tree.")
            full_text = " ".join(s for s in strings if not (s == "<shift>" or s in self.reduce_token_map))
            return empty_tree(self.config.relation_types, text=full_text)
        except Exception as e:
            raise DecodeInvariantError(
                f"Could not build a tree from a mask-legal shift-reduce stream "
                f"({type(e).__name__}: {e}): the constraints and the reconstruction "
                f"disagree. strings={len(strings)}"
            ) from e
