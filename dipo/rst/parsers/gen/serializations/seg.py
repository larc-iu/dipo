"""Segmentation only: the shift-reduce stream without REDUCE actions. Every source token is
copied (`<copy>`) and each EDU is closed with `<shift>`, so decoding inherits the
shift-reduce serialization's guarantee that the EDUs tile the input exactly. The tree it
returns spans the predicted EDUs with placeholder structure (right-branching, the first
relation type), so only segmentation is meaningful; `maekawa` pairs it with a parser.
"""

from dipo.rst.data.tree import Reduce, RstTree, Shift, strings_to_actions
from dipo.rst.parsers.common.seqgen import ShiftReduceDecodeState, align_edus_to_tokens, reconstruct_text
from dipo.rst.parsers.gen.serializations.sr import SRSerialization


class SegSerialization(SRSerialization):
    def action_tokens(self) -> list[str]:
        self.reduce_token_map = {}
        return [self.COPY_TOKEN, Shift().to_token()]

    def linearize(self, tree):
        """tree -> (source_ids, seen_tokens, label_tokens): per EDU, a COPY per subword
        then a SHIFT (see `SRSerialization.linearize`)."""
        if self.copy_token_id is None:
            raise RuntimeError("linearize called before build_vocab (cfg.relation_types unset?)")
        source_ids, spans = align_edus_to_tokens(self.tokenizer, reconstruct_text(tree), tree.edus)
        seen: list[int] = []
        label: list[int] = []
        for s, e in spans:
            seen.extend(source_ids[s:e])
            label.extend([self.copy_token_id] * (e - s))
            seen.append(self.shift_token_id)
            label.append(self.shift_token_id)
        return source_ids, seen, label

    def initial_state(self, source_ids):
        return ShiftReduceDecodeState(
            source_len=len(source_ids), min_edu_length=max(1, int(self.config.min_edu_length)), segment_only=True
        )

    def build_tree(self, action_ids: list[int], source_ids: list[int]):
        strings: list[str] = []
        buffer: list[int] = []
        cursor = 0

        def flush():
            if buffer:
                strings.extend(self.tokenizer.decode(buffer, skip_special_tokens=False).split())
                buffer.clear()

        for t in action_ids:
            if t == self.stream_end_id:
                break
            if t == self.copy_token_id and cursor < len(source_ids):
                buffer.append(source_ids[cursor])
                cursor += 1
            elif t == self.shift_token_id:
                flush()
                strings.append(Shift().to_token())
        flush()
        if not strings or strings[-1] != Shift().to_token():
            strings.append(Shift().to_token())  # an unfinished decode commits its last EDU
        actions = strings_to_actions(strings, {})
        rel, kind = self.config.relation_types[0]
        reduce = Reduce(nuc="NN" if kind == "multinuc" else "NS", rel=rel)
        n_edus = sum(isinstance(a, Shift) for a in actions)
        return RstTree.from_shift_reduce(actions + [reduce] * (n_edus - 1), relation_types=self.config.relation_types)
