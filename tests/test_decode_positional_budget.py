"""Inference respects the backbone's positional limit (`gen/decode.py::_step_budget`).

Training's `pack_example` drops any example whose realized single stream exceeds
`min(per-side sum, max_position_embeddings)`, but inference used to check only the
source side: prefix + a full `max_output_length` decode could run past the model's
positional limit with silent RoPE extrapolation instead of a loud failure. The decode
loops now tighten their step budget by `Backbone.max_decode_steps(prefix_len)` and
raise `OverLengthError` when it binds. Pure CPU, fake backbone + stub automaton
(same shape as test_decode_invariants).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from iudex.rst.parsers.gen.decode import beam_decode, greedy_decode
from iudex.rst.parsers.gen.errors import OverLengthError

STEP, EOS = 0, 1
V = 2

_TREE = object()


class _StubState:
    """STEPs `steps_to_eos` times, then only EOS is legal."""

    def __init__(self, steps_to_eos: int):
        self.count = 0
        self.done = False
        self.steps_to_eos = steps_to_eos

    def clone(self) -> "_StubState":
        st = _StubState(self.steps_to_eos)
        st.count, st.done = self.count, self.done
        return st

    def legal(self) -> set[int]:
        return {STEP} if self.count < self.steps_to_eos else {EOS}


class _StubSer:
    def decode_id(self, idx: int) -> int:
        return idx

    def pred_mask(self, st, vocab_size: int):
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for i in st.legal():
            mask[i] = True
        return mask

    def apply(self, st, full_id: int, source_ids):
        if full_id == EOS:
            st.done = True
            return None, "eos"
        st.count += 1
        return STEP, "step"

    def candidate_ranges(self, st, finished: bool):
        return []

    def stash_meta(self, tree, ranges, source_ids) -> None:
        pass

    def build_tree(self, action_ids, source_ids):
        return _TREE


class _CappedBackbone:
    """Fake with a positional limit: `max_decode_steps` mirrors decoder_only's
    `max_positions - prefix_len` arithmetic."""

    def __init__(self, max_positions: int | None):
        self.max_positions = max_positions
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    def decode_prefix(self, source_ids):
        return list(source_ids)

    def max_decode_steps(self, prefix_len: int):
        if self.max_positions is None:
            return None
        return max(0, self.max_positions - prefix_len)

    def seed(self, prefix_ids, num_rows: int = 1):
        shape = (V,) if num_rows == 1 else (num_rows, V)
        return torch.zeros(shape), None

    def advance(self, cache, next_input):
        rows = 1 if isinstance(next_input, int) else len(next_input)
        shape = (V,) if rows == 1 else (rows, V)
        return torch.zeros(shape), None

    def inference_mode(self):
        return contextlib.nullcontext()

    def beam_reorder_needed(self, step, parents, k, cache) -> bool:
        return False

    def reorder_cache(self, cache, parent_tensor):
        return cache


SOURCE = [7, 8, 9]  # prefix_len == 3 under the fake's identity decode_prefix


def _parser(max_positions: int | None, max_output_length: int = 64):
    return SimpleNamespace(
        backbone=_CappedBackbone(max_positions),
        serialization=_StubSer(),
        config=SimpleNamespace(max_output_length=max_output_length, relation_types=[("elaboration", "rst")]),
        device=torch.device("cpu"),
    )


def _factory(steps_to_eos: int):
    return lambda: _StubState(steps_to_eos)


def test_greedy_raises_when_the_positional_limit_binds():
    # 10 positions - 3 prefix = 7 steps, but the trace needs 8 (7 STEPs + EOS).
    parser = _parser(max_positions=10)
    with pytest.raises(OverLengthError, match="positional limit"):
        greedy_decode(parser, SOURCE, _factory(steps_to_eos=7), parser.serialization.pred_mask)


def test_greedy_decodes_within_the_positional_limit():
    # The same trace fits once one more position is available.
    parser = _parser(max_positions=11)
    ser = parser.serialization
    assert greedy_decode(parser, SOURCE, _factory(steps_to_eos=7), ser.pred_mask) is _TREE


def test_greedy_raises_before_seeding_when_the_prefix_alone_hits_the_limit():
    parser = _parser(max_positions=3)  # budget = 3 - 3 = 0
    parser.backbone.seed = None  # a seed attempt would TypeError, not OverLengthError
    with pytest.raises(OverLengthError, match="positional limit"):
        greedy_decode(parser, SOURCE, _factory(steps_to_eos=1), parser.serialization.pred_mask)


def test_greedy_without_a_positional_limit_keeps_max_output_length():
    parser = _parser(max_positions=None, max_output_length=4)
    with pytest.raises(OverLengthError, match="max_output_length=4"):
        greedy_decode(parser, SOURCE, _factory(steps_to_eos=10), parser.serialization.pred_mask)


def test_beam_raises_when_the_positional_limit_binds():
    parser = _parser(max_positions=10)
    with pytest.raises(OverLengthError, match="positional limit"):
        beam_decode(parser, SOURCE, 2, _factory(steps_to_eos=7), parser.serialization.pred_mask)


def test_beam_decodes_within_the_positional_limit():
    parser = _parser(max_positions=11)
    ser = parser.serialization
    assert beam_decode(parser, SOURCE, 2, _factory(steps_to_eos=7), ser.pred_mask) is _TREE
