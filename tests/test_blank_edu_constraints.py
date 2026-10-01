"""gen decoding never ends an EDU on whitespace-only subwords (larc-iu/iudex#1).

A source subword that decodes to whitespace only (a bare space, a newline byte) can
be copied into an EDU, and an EDU holding nothing else surfaces as "". Both pred
decode states now refuse to end such an EDU. SR also refuses to shift once no text
remains, so trailing blanks join the final EDU. The sexp PDA budgets TEXT positions, so every leaf is guaranteed one and there is
no fallback.

The sexp PDA's "legal set is never empty" invariant was verified exhaustively when
it was written; it is re-checked here over every reachable state, under every
blank-position subset, so the new budget cannot strand a decode.
"""

from itertools import combinations

import pytest

from iudex.rst.parsers.common.seqgen import ShiftReduceDecodeState
from iudex.rst.parsers.common.sexp_constraints import SexpDecodingState
from iudex.rst.parsers.gen.serializations.base import Serialization

OPEN_ID, CLOSE_ID, EOS_ID, COPY_ID = 1, 2, 3, 4
LABEL_IDS = frozenset({100, 101, 102})


def test_blank_positions_finds_whitespace_only_subwords():
    transformers = pytest.importorskip("transformers")
    ser = Serialization(config=None)
    ser.tokenizer = transformers.AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-400m")
    ids = ser.tokenizer("First.\n\nSecond.", add_special_tokens=False)["input_ids"]
    toks = ser.tokenizer.convert_ids_to_tokens(ids)
    expected = {i for i, t in enumerate(toks) if not t.replace("Ċ", "").replace("Ġ", "")}
    assert expected and ser.blank_positions(ids) == frozenset(expected)
    # An all-blank source has nothing to anchor an EDU to: no constraint.
    nl = ser.tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    assert ser.blank_positions(nl) == frozenset()


# ---- SR ----


def test_sr_withholds_shift_from_a_whitespace_only_edu():
    st = ShiftReduceDecodeState(source_len=4, blank_positions=frozenset({1}))
    st.step_copy()  # pos 0: text
    assert st.shift_ok
    st.step_shift()
    st.step_copy()  # pos 1: blank
    assert st.copy_ok and not st.shift_ok
    st.step_copy()  # pos 2: text
    assert st.shift_ok


def test_sr_trailing_blanks_join_the_final_edu():
    st = ShiftReduceDecodeState(source_len=3, blank_positions=frozenset({1, 2}))
    st.step_copy()  # pos 0: the last text
    assert not st.shift_ok and st.copy_ok
    st.step_copy()
    assert not st.shift_ok
    st.step_copy()
    assert st.at_end and st.shift_ok
    st.step_shift()
    assert st.pred_edu_ranges == [(0, 3)]


def test_sr_clone_carries_the_new_fields():
    st = ShiftReduceDecodeState(source_len=3, blank_positions=frozenset({0}))
    st.step_copy()
    c = st.clone()
    assert c.blank_positions == st.blank_positions and c.edu_has_text is False and not c.shift_ok


def test_sr_without_blank_positions_is_unchanged():
    st = ShiftReduceDecodeState(source_len=2)
    st.step_copy()
    assert st.shift_ok


# ---- sexp ----


def _sexp(source_len, order, blank):
    return SexpDecodingState(
        source_len=source_len,
        traversal_order=order,
        use_copy=True,
        open_id=OPEN_ID,
        close_id=CLOSE_ID,
        eos_id=EOS_ID,
        label_ids=LABEL_IDS,
        copy_id=COPY_ID,
        blank_positions=frozenset(blank),
    )


def _walk(state):
    """Every reachable state (DFS over legal actions)."""
    stack = [state]
    while stack:
        st = stack.pop()
        yield st
        if st.is_terminal():
            continue
        for a in st.legal_actions():
            stack.append(st.step(a))


def test_sexp_withholds_close_from_a_whitespace_only_leaf():
    st = _sexp(3, "preorder", {0})
    for a in (OPEN_ID, 100, OPEN_ID, COPY_ID):  # root, label, leaf, blank copy
        st = st.step(a)
    assert CLOSE_ID not in st.legal_actions() and COPY_ID in st.legal_actions()
    st = st.step(COPY_ID)  # text
    assert CLOSE_ID in st.legal_actions()


@pytest.mark.parametrize("order", ["preorder", "postorder"])
@pytest.mark.parametrize("source_len", [2, 3, 4, 5])
def test_sexp_never_strands_and_never_closes_a_blank_leaf(order, source_len):
    # k < source_len: an all-blank source gets no blank_positions (see above).
    for k in range(source_len):
        for blank in combinations(range(source_len), k):
            terminals = 0
            for st in _walk(_sexp(source_len, order, blank)):
                if st.is_terminal():
                    terminals += 1
                    continue
                legal = st.legal_actions()
                assert legal, f"empty legal set: blank={blank} state={st}"
                top = st.stack[-1] if st.stack else None
                if top is not None and top.kind == "leaf" and not top.leaf_has_text:
                    assert CLOSE_ID not in legal, f"blank leaf may close: blank={blank} state={st}"
            assert terminals > 0
