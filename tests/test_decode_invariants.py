"""Mask/automaton disagreement is a crash, not a fallback (`gen/decode.py`).

Decoding is always constrained, and `beam_topk_step` scores mask-illegal actions
at -inf, so a FINITE-scored action the state machine rejects means the mask and
the automaton disagree -- a bug with no correct recovery. Both loops used to
absorb it: beam tried to `warn` (a NameError, since `warn` was never imported)
and drop the beam, then returned a single-EDU `empty_tree` if every beam died;
greedy silently built a tree from the partial action sequence. Either way a
legality bug scored as if the model had produced that output.

Pure CPU: fake backbone + stub state machine drive the decode loops directly, so
the disagreement can be staged deliberately (a real serialization's mask and
automaton agree, which is the whole point).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from iudex.rst.parsers.gen.decode import beam_decode, greedy_decode
from iudex.rst.parsers.gen.errors import DecodeInvariantError

# Action vocabulary for the stub automaton.
STEP, EOS, TRAP = 0, 1, 2
V = 3

_TREE = object()  # build_tree's sentinel return; these tests never inspect it


class _StubState:
    """Counts STEPs, then EOSes. `TRAP` is never legal, so a correct loop never
    applies it; applying it returns "illegal"."""

    def __init__(self, max_steps: int = 3):
        self.count = 0
        self.done = False
        self.max_steps = max_steps

    def clone(self) -> _StubState:
        st = _StubState(self.max_steps)
        st.count, st.done = self.count, self.done
        return st

    def legal(self) -> set[int]:
        if self.count == 0:
            return {STEP}  # only 1 legal action => forces -inf topk backfill
        if self.count >= self.max_steps:
            return {EOS}
        return {STEP, EOS}


class _StubSer:
    """`legal_override` / `apply_override` stage a mask/automaton disagreement."""

    def __init__(self, *, legal_override=None, apply_override=None):
        self.legal_override = legal_override
        self.apply_override = apply_override
        self.applied: list[int] = []

    def decode_id(self, idx: int) -> int:
        return idx

    def pred_mask(self, st, vocab_size: int):
        ids = self.legal_override(st) if self.legal_override else st.legal()
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for i in ids:
            mask[i] = True
        return mask

    def apply(self, st, full_id: int, source_ids):
        self.applied.append(full_id)
        if self.apply_override is not None:
            return self.apply_override(st, full_id)
        if full_id == EOS:
            st.done = True
            return None, "eos"
        if full_id == STEP:
            st.count += 1
            return STEP, "step"
        st.done = True
        return None, "illegal"

    def candidate_ranges(self, st, finished: bool):
        return []

    def stash_meta(self, tree, ranges, source_ids) -> None:
        pass

    def build_tree(self, action_ids, source_ids):
        return _TREE


class _FakeBackbone:
    """Uniform logits: the mask alone decides what is selectable."""

    def __init__(self):
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    def decode_prefix(self, source_ids):
        return list(source_ids)

    def max_decode_steps(self, prefix_len: int):
        return None  # no positional limit; max_output_length is the budget

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


def _parser(ser, max_output_length: int = 32):
    return SimpleNamespace(
        backbone=_FakeBackbone(),
        serialization=ser,
        config=SimpleNamespace(max_output_length=max_output_length, relation_types=[("elaboration", "rst")]),
        device=torch.device("cpu"),
    )


SOURCE = [7, 8, 9]


def _factory(ser):
    return lambda: _StubState()


# --- greedy ---------------------------------------------------------------


def test_greedy_raises_when_mask_admits_an_action_the_automaton_rejects():
    # The mask offers TRAP; apply rejects it. Greedy used to skip recording the
    # action and build a tree from the prefix, hiding the bug.
    ser = _StubSer(legal_override=lambda st: {TRAP})
    with pytest.raises(DecodeInvariantError, match="illegal"):
        greedy_decode(_parser(ser), SOURCE, _factory(ser), ser.pred_mask)


def test_greedy_raises_on_copy_exhausted():
    ser = _StubSer(apply_override=lambda st, fid: (None, "copy_exhausted"))
    with pytest.raises(DecodeInvariantError, match="copy_exhausted"):
        greedy_decode(_parser(ser), SOURCE, _factory(ser), ser.pred_mask)


def test_greedy_raises_on_empty_legal_set():
    # argmax over an all--inf row would silently return index 0.
    ser = _StubSer(legal_override=lambda st: set())
    with pytest.raises(DecodeInvariantError, match="deadlock"):
        greedy_decode(_parser(ser), SOURCE, _factory(ser), ser.pred_mask)


def test_greedy_still_decodes_a_legal_trace():
    ser = _StubSer()
    assert greedy_decode(_parser(ser), SOURCE, _factory(ser), ser.pred_mask) is _TREE
    assert TRAP not in ser.applied


# --- beam -----------------------------------------------------------------


def test_beam_raises_when_mask_admits_an_action_the_automaton_rejects():
    ser = _StubSer(legal_override=lambda st: {TRAP})
    with pytest.raises(DecodeInvariantError, match="illegal"):
        beam_decode(_parser(ser), SOURCE, 3, _factory(ser), ser.pred_mask)


def test_beam_raises_on_empty_legal_set():
    ser = _StubSer(legal_override=lambda st: set())
    with pytest.raises(DecodeInvariantError, match="deadlock"):
        beam_decode(_parser(ser), SOURCE, 3, _factory(ser), ser.pred_mask)


def test_beam_never_applies_a_backfilled_action():
    """The regression this fix turns on: at step 0 exactly one action is legal,
    so topk backfills the other K-1 rows with -inf garbage (here TRAP, whose
    apply returns "illegal"). Those rows are not real hypotheses. The old loop
    applied them anyway and leaned on a later sweep to discard the wreckage;
    now they are skipped, so a normal backfill must decode cleanly."""
    ser = _StubSer()
    assert beam_decode(_parser(ser), SOURCE, 3, _factory(ser), ser.pred_mask) is _TREE
    assert TRAP not in ser.applied, "a -inf backfilled action was applied to a live state"


def test_beam_terminates_when_only_dead_rows_remain():
    """Dead rows are never advanced, so their states never reach `done`:
    liveness, not `done`, must terminate the loop. If this regresses, the decode
    burns the full max_output_length budget on every document instead of
    breaking once the last live beam finishes."""
    ser = _StubSer()
    parser = _parser(ser, max_output_length=64)
    assert beam_decode(parser, SOURCE, 3, _factory(ser), ser.pred_mask) is _TREE
    # A 3-STEP + EOS trace over 3 beams cannot legally apply 64 actions; a
    # non-terminating loop would.
    assert len(ser.applied) < 32, f"loop did not stop at the last live beam: {len(ser.applied)} applies"
