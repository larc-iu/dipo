"""PDA soundness for label_style='words': the sexp constraint automaton with
multi-token, prefix-free relation-word labels (a trie at each label slot) plus
literal single-id { } brackets. Pure-PDA tests (synthetic ids, no model)."""
import pytest

from iudex.rst.parsers.common.sexp_constraints import GoldEduForcer, SexpDecodingState

OPEN, CLOSE, COPY, EOS = 1, 2, 3, 4
# Prefix-free trie sharing a first token (10) across a 2- and a 3-token label,
# to exercise mid-label branching: {10:11 | 10:14:15}.
NS_ELAB = (10, 11)
NN_JOINT = (12, 13)
NS_TOPIC = (10, 14, 15)
WORD_LABELS = frozenset({NS_ELAB, NN_JOINT, NS_TOPIC})


def make_state(source_len, order="postorder"):
    return SexpDecodingState(
        source_len=source_len,
        traversal_order=order,
        use_copy=True,
        open_id=OPEN,
        close_id=CLOSE,
        eos_id=EOS,
        label_ids=frozenset(),  # words mode: single-id label set is empty
        copy_id=COPY,
        min_edu_length=1,
        word_label_ids=WORD_LABELS,
    )


# Tree ::= ("leaf", n_subwords) | ("node", label_seq, left, right)
def emit_postorder(node):
    if node[0] == "leaf":
        return [OPEN] + [COPY] * node[1] + [CLOSE]
    _, label, left, right = node
    return [OPEN] + emit_postorder(left) + emit_postorder(right) + list(label) + [CLOSE]


def leaves_subword_total(node):
    if node[0] == "leaf":
        return node[1]
    return leaves_subword_total(node[2]) + leaves_subword_total(node[3])


def n_leaves(node):
    if node[0] == "leaf":
        return 1
    return n_leaves(node[2]) + n_leaves(node[3])


def drive(state, stream):
    """Step the PDA through `stream`, asserting soundness (every action legal)
    and no deadlock (legal set never empty pre-terminal) at each step."""
    for a in stream:
        legal = state.legal_actions()
        assert legal, "empty legal set (deadlock) before consuming a valid stream token"
        assert a in legal, f"token {a} not in legal set {sorted(legal)}"
        state = state.step(a)
    return state


TREES = [
    ("node", NS_ELAB, ("leaf", 2), ("leaf", 3)),
    ("node", NS_TOPIC, ("leaf", 1), ("leaf", 4)),  # 3-token label
    (
        "node",
        NN_JOINT,
        ("node", NS_ELAB, ("leaf", 2), ("leaf", 2)),
        ("leaf", 3),
    ),
    # deep left spine
    (
        "node",
        NS_TOPIC,
        ("node", NN_JOINT, ("node", NS_ELAB, ("leaf", 1), ("leaf", 1)), ("leaf", 2)),
        ("leaf", 1),
    ),
]


@pytest.mark.parametrize("tree", TREES)
def test_valid_words_stream_accepted_no_deadlock(tree):
    src = leaves_subword_total(tree)
    stream = emit_postorder(tree) + [EOS]
    state = drive(make_state(src), stream)
    assert state.is_terminal()


def test_trie_branching_at_label_slot():
    # Reach the postorder label slot of a 2-leaf tree: after both leaves close,
    # the legal set is the labels' FIRST tokens; then the trie narrows.
    tree = ("node", NS_TOPIC, ("leaf", 1), ("leaf", 1))
    stream = emit_postorder(tree)  # ends right before the label + close
    # Drive up to just before the label tokens (drop label + trailing close).
    prefix = stream[: -(len(NS_TOPIC) + 1)]
    state = drive(make_state(leaves_subword_total(tree)), prefix)
    assert state.legal_actions() == frozenset({10, 12})  # first tokens of the labels
    state = state.step(10)
    assert state.legal_actions() == frozenset({11, 14})  # 10:11 | 10:14:15
    state = state.step(14)
    assert state.legal_actions() == frozenset({15})  # only completion of 10:14:15
    state = state.step(15)
    # Label complete: now the frame can close.
    assert CLOSE in state.legal_actions()


def test_illegal_mid_label_token_rejected():
    tree = ("node", NS_ELAB, ("leaf", 1), ("leaf", 1))
    stream = emit_postorder(tree)
    prefix = stream[: -(len(NS_ELAB) + 1)]
    state = drive(make_state(leaves_subword_total(tree)), prefix)
    state = state.step(10)  # start NS_ELAB / NS_TOPIC
    with pytest.raises(ValueError):
        state.step(99)  # 99 is not a legal continuation of (10,)
    with pytest.raises(ValueError):
        state.step(CLOSE)  # cannot close mid-label


@pytest.mark.parametrize("tree", TREES)
def test_gold_forcer_drives_to_n_leaves(tree):
    # Gold ranges tile the source; the forcer must produce exactly n leaves with
    # complete multi-token labels, whatever the (deterministic) model picks.
    src = leaves_subword_total(tree)
    n = n_leaves(tree)
    # simple 1-subword-per-leaf-min tiling is not required; use equal-ish ranges
    # derived from the tree's own leaf sizes so ranges are real and monotonic.
    sizes = []

    def collect(node):
        if node[0] == "leaf":
            sizes.append(node[1])
        else:
            collect(node[2])
            collect(node[3])

    collect(tree)
    ranges, c = [], 0
    for s in sizes:
        ranges.append((c, c + s))
        c += s
    forcer = GoldEduForcer(n, ranges)
    state = make_state(src)
    steps = 0
    while not state.is_terminal():
        steps += 1
        assert steps < 10_000, "forcer failed to terminate (possible deadlock)"
        narrowed = forcer.narrowed_legal(state)
        if narrowed is None:
            a = min(state.legal_actions())
        else:
            a = min(narrowed)
        before = state
        state = state.step(a)
        forcer.observe(before, state, a)
    assert forcer.closed_leaves == n


def test_token_mode_still_single_id_labels():
    # word_label_ids empty -> label_next_ids() == label_ids, mid-label guard never
    # fires, behavior identical to the pre-words PDA.
    st = SexpDecodingState(
        source_len=2, traversal_order="postorder", use_copy=True,
        open_id=OPEN, close_id=CLOSE, eos_id=EOS,
        label_ids=frozenset({20, 21}), copy_id=COPY, min_edu_length=1,
    )
    assert st.label_next_ids() == frozenset({20, 21})
    assert st.label_cursor == ()
    # A single-id label stream still parses: ( ( c c ) ( c c ) 20 ) EOS
    stream = [OPEN, OPEN, COPY, COPY, CLOSE, OPEN, COPY, COPY, CLOSE, 20, CLOSE, EOS]
    state = drive(SexpDecodingState(
        source_len=4, traversal_order="postorder", use_copy=True,
        open_id=OPEN, close_id=CLOSE, eos_id=EOS,
        label_ids=frozenset({20, 21}), copy_id=COPY, min_edu_length=1,
    ), stream)
    assert state.is_terminal()
