"""Prompts, gold training examples and tree assembly for the Maekawa et al. (2024) parser.

The prompt formats reproduce those of Maekawa et al. (2024) exactly, down to the trailing
space after each final colon and the en dash in the top-down range, so that adapters
trained with their setup receive identical inputs. This is an independent
reimplementation; no code from their repository is included.

Spans are EDU ranges `(b, e)`, end exclusive, and a binary node is `(b, s, e)` with
`s` the first EDU of its right child (the convention of `RstTree.spans_with_ranges`
and `RstTree.from_parsing_actions`).
"""

import re
from collections import Counter
from dataclasses import dataclass

from dipo.rst.data.tree import RstTree

NUC_SURFACE = {"NS": "nucleus-satellite", "NN": "nucleus-nucleus", "SN": "satellite-nucleus"}
NUC_CODE = {v: k for k, v in NUC_SURFACE.items()}
# The order their nucleus prompt lists the labels in, and their fallback for an
# illegal answer (the most frequent label).
NUC_ANSWERS = ("nucleus-satellite", "nucleus-nucleus", "satellite-nucleus")
DEFAULT_NUC = "nucleus-satellite"
SPAN_ANSWERS = ("shift", "reduce")
DEFAULT_SPAN = "shift"
# Queue position in a bottom-up unified decision's key when the queue is empty (only
# reduce is legal, but the reduced node's labels are still decided).
QUEUE_EMPTY = -1


@dataclass(frozen=True)
class Decision:
    """One prompt and its answer (the gold one in training data, "" when parsing). `key`
    identifies the parser state: for a shift/reduce decision the stack-top spans and next
    EDU `(b2, e2, b1, e1, q)` (`q` is QUEUE_EMPTY when the queue is), for a label decision
    the node `(b, s, e)`, for a top-down split the span `(b, s, e)` in training data and
    `(b, -1, e)` when parsing, where the split is what is being decided."""

    task: str
    key: tuple[int, ...]
    prompt: str
    answer: str


def span_text(edus: list[str], b: int, e: int) -> str:
    return " ".join(edu.strip() for edu in edus[b:e])


def span_prompt(edus: list[str], stack2: tuple[int, int], stack1: tuple[int, int], queue1: int) -> str:
    return "\n".join(
        [
            f"Stack2: {span_text(edus, *stack2)}",
            f"Stack1: {span_text(edus, *stack1)}",
            f"Queue1: {edus[queue1].strip()}",
            "Action (shift or reduce): ",
        ]
    )


def nuc_prompt(edus: list[str], left: tuple[int, int], right: tuple[int, int]) -> str:
    return "\n".join(
        [
            f"Span1: {span_text(edus, *left)}",
            f"Span2: {span_text(edus, *right)}",
            "Nucleus label (nucleus-nucleus, nucleus-satellite, or satellite-nucleus): ",
        ]
    )


def rel_with_nuc_prompt(
    edus: list[str], left: tuple[int, int], right: tuple[int, int], nuc: str, relation_answers: list[str]
) -> str:
    return "\n".join(
        [
            f"Span1: {span_text(edus, *left)}",
            f"Span2: {span_text(edus, *right)}",
            f"Nucleus label: {nuc}",
            "Relation label ({}, or {}): ".format(", ".join(relation_answers[:-1]), relation_answers[-1]),
        ]
    )


def joint_prompt(edus: list[str], left: tuple[int, int], right: tuple[int, int], joint_answers: list[str]) -> str:
    """Nuclearity and relation as one decision (not in Maekawa et al.): their
    span lines, then the legal "<NS|SN|NN> <relation>" labels in the style of their
    label lines."""
    return "\n".join(
        [
            f"Span1: {span_text(edus, *left)}",
            f"Span2: {span_text(edus, *right)}",
            "Nucleus and relation label ({}, or {}): ".format(", ".join(joint_answers[:-1]), joint_answers[-1]),
        ]
    )


def top_down_prompt(edus: list[str], b: int, e: int) -> str:
    text = edus[b].strip()
    for i, edu in enumerate(edus[b + 1 : e]):
        text += f" [{i}] {edu.strip()}"
    return "\n".join([f"Input: {text}", f"Split point (0–{e - b - 2}): "])


def unified_bu_prompt(edus: list[str], stack2: tuple[int, int], stack1: tuple[int, int], queue1: int | None) -> str:
    """The span (shift/reduce) prompt's state lines, asking for the action and, for a
    reduce, the new node's labels (not in Maekawa et al.). The labels are not listed, to
    keep the prompt short. With an empty queue (`queue1` None) the Queue1 line is bare."""
    return "\n".join(
        [
            f"Stack2: {span_text(edus, *stack2)}",
            f"Stack1: {span_text(edus, *stack1)}",
            f"Queue1: {edus[queue1].strip()}" if queue1 is not None else "Queue1:",
            "Action (shift, or reduce with nuclearity and relation): ",
        ]
    )


def unified_td_prompt(edus: list[str], b: int, e: int) -> str:
    """Their top-down prompt (only the span being split), asking for the split point and
    the node's labels (not in Maekawa et al.), labels not listed."""
    return top_down_prompt(edus, b, e).rsplit("\n", 1)[0] + f"\nSplit point (0–{e - b - 2}), nuclearity and relation: "


def middle_split(b: int, e: int) -> int:
    """The split index (0-based, as in the top-down answers) of the most balanced split."""
    return (e - b) // 2 - 1


_SPLIT = re.compile(r"[0-9]+")


def resolve_unified_bu(text: str, can_shift: bool, labels: "Labels") -> tuple[str, list[str]]:
    """A generated bottom-up unified answer as a legal one, part by part: an illegal or
    missing action falls back to the span default (shift) when shifting is legal and to
    reduce when it is not, and a reduce's illegal or missing label to the most frequent
    joint label. Returns the answer and the parts that fell back ("action", "label")."""
    action, _, label = text.strip().partition(" ")
    label = label.strip()
    fallbacks = []
    if action not in SPAN_ANSWERS or (action == "shift" and not can_shift):
        fallbacks.append("action")
        action = DEFAULT_SPAN if can_shift else "reduce"
    if action == "shift":
        return "shift", fallbacks
    if label not in labels.joint_of:
        fallbacks.append("label")
        label = labels.joint_answers[0]
    return f"reduce {label}", fallbacks


def resolve_unified_td(text: str, b: int, e: int, labels: "Labels") -> tuple[str, list[str]]:
    """A generated top-down unified answer as a legal one, part by part: an illegal or
    missing split index falls back to the middle split, an illegal or missing label to
    the most frequent joint label. Returns the answer and the parts that fell back
    ("split", "label")."""
    split, _, label = text.strip().partition(" ")
    label = label.strip()
    fallbacks = []
    if _SPLIT.fullmatch(split) and int(split) < e - b - 1:
        k = int(split)
    else:
        fallbacks.append("split")
        k = middle_split(b, e)
    if label not in labels.joint_of:
        fallbacks.append("label")
        label = labels.joint_answers[0]
    return f"{k} {label}", fallbacks


def top_down_answers(b: int, e: int) -> list[str]:
    return [str(i) for i in range(e - b - 1)]


def gold_nodes(tree: RstTree) -> dict[tuple[int, int], tuple[int, str, str]]:
    """`(b, e) -> (s, nuclearity code, relation)` for every internal node of the binarized tree."""
    return {(left[0], right[1]): (right[0], nuc, rel) for (left, right), nuc, rel in tree.spans_with_ranges()}


def relation_frequencies(trees: list[RstTree]) -> Counter:
    return Counter(rel for t in trees for _, _, rel in t.spans_with_ranges())


def joint_label_frequencies(trees: list[RstTree]) -> Counter:
    """(nuclearity code, relation) pairs as they occur in `trees`."""
    return Counter((nuc, rel) for t in trees for _, nuc, rel in t.spans_with_ranges())


class Labels:
    """Maps the data's relation names to the answers the model reads and writes, and, for
    joint labelling, (nuclearity code, relation) pairs to answers like "NS elaboration"."""

    def __init__(
        self,
        relations: list[str],
        surface: dict[str, str] | None = None,
        default: str | None = None,
        joint: list[tuple[str, str]] | None = None,
    ):
        if not relations:
            raise ValueError("no relations to offer the model")
        surface = surface or {}
        self.relations = list(relations)
        self.answers = [surface.get(r, r) for r in self.relations]
        if len(set(self.answers)) != len(self.answers):
            raise ValueError(f"relation surfaces collide: {self.answers}")
        self.relation_of = dict(zip(self.answers, self.relations, strict=True))
        self.answer_of = dict(zip(self.relations, self.answers, strict=True))
        self.default_answer = self.answer_of[default] if default is not None else self.answers[0]
        # Joint labels, most frequent first; the first is the fallback for an illegal answer.
        self.joint = [(code, rel) for code, rel in joint or []]
        self.joint_answers = [f"{code} {self.answer_of[rel]}" for code, rel in self.joint]
        self.joint_of = dict(zip(self.joint_answers, self.joint, strict=True))

    def joint_answer(self, nuc_code: str, relation: str) -> str:
        """The joint answer for a gold pair; one unseen in training is never produced."""
        return f"{nuc_code} {self.answer(relation)}"

    def answer(self, relation: str) -> str:
        """The answer for `relation`. A relation the model is never offered (absent from the
        training data) keeps its name: as a gold answer it is simply never produced."""
        return self.answer_of.get(relation, relation)


def gold_decisions(tree: RstTree, labels: Labels, tasks: tuple[str, ...]) -> list[Decision]:
    """Every decision Maekawa et al.'s training data holds for one gold tree, restricted to
    `tasks`. Span (shift/reduce) decisions only exist where both actions are legal, and
    top-down splits only for spans of more than two EDUs, as in their training data. The
    unified decisions (not theirs) cover every internal node exactly once: "unified_bu"
    wherever a reduce is legal, "unified_td" for every node including 2-EDU spans."""
    edus = tree.edu_strings
    nodes = gold_nodes(tree)
    out: list[Decision] = []
    for b, s, e in _postorder(nodes, 0, len(edus)):
        nuc_code, rel = nodes[(b, e)][1], nodes[(b, e)][2]
        if "nuc" in tasks:
            out.append(Decision("nuc", (b, s, e), nuc_prompt(edus, (b, s), (s, e)), NUC_SURFACE[nuc_code]))
        if "rel_with_nuc" in tasks:
            prompt = rel_with_nuc_prompt(edus, (b, s), (s, e), NUC_SURFACE[nuc_code], labels.answers)
            out.append(Decision("rel_with_nuc", (b, s, e), prompt, labels.answer(rel)))
        if "joint" in tasks:
            # A pair absent from the joint labels (unseen in training) is still a decision:
            # scored on dev, it counts as one the model cannot get right.
            prompt = joint_prompt(edus, (b, s), (s, e), labels.joint_answers)
            out.append(Decision("joint", (b, s, e), prompt, labels.joint_answer(nuc_code, rel)))
        if "top_down" in tasks and e - b > 2:
            out.append(Decision("top_down", (b, s, e), top_down_prompt(edus, b, e), str(s - b - 1)))
        if "unified_td" in tasks:  # every node, 2-EDU spans too: the split is forced, the labels are not
            answer = f"{s - b - 1} {labels.joint_answer(nuc_code, rel)}"
            out.append(Decision("unified_td", (b, s, e), unified_td_prompt(edus, b, e), answer))
    if "span" in tasks:
        out.extend(_span_decisions(edus, nodes))
    if "unified_bu" in tasks:
        out.extend(_span_decisions(edus, nodes, labels))
    return out


def _postorder(nodes: dict, b: int, e: int):
    """Internal nodes `(b, s, e)`, children before parents, left before right."""
    if e - b < 2:
        return
    s = nodes[(b, e)][0]
    yield from _postorder(nodes, b, s)
    yield from _postorder(nodes, s, e)
    yield b, s, e


def _span_decisions(edus: list[str], nodes: dict, labels: Labels | None = None) -> list[Decision]:
    """The gold shift-reduce run's decisions: with `labels` None, span decisions where both
    actions are legal (theirs). With `labels`, unified decisions wherever a reduce is legal
    (stack of two or more): where shifting is legal too, and where the queue is empty and
    only the reduce's labels are open. Every reduce, so every internal node, gets exactly
    one; states where only shift is legal get none."""
    actions: list[tuple] = []

    def walk(b: int, e: int) -> None:
        if e - b == 1:
            actions.append(("shift", b))
            return
        s = nodes[(b, e)][0]
        walk(b, s)
        walk(s, e)
        actions.append(("reduce", b, e))

    walk(0, len(edus))
    stack: list[tuple[int, int]] = []
    queue = 0
    out = []
    for act in actions:
        if labels is None and len(stack) >= 2 and queue < len(edus):
            key = (*stack[-2], *stack[-1], queue)
            out.append(Decision("span", key, span_prompt(edus, stack[-2], stack[-1], queue), act[0]))
        elif labels is not None and len(stack) >= 2:
            q = queue if queue < len(edus) else None
            key = (*stack[-2], *stack[-1], QUEUE_EMPTY if q is None else q)
            answer = "shift"
            if act[0] == "reduce":
                _, nuc_code, rel = nodes[(act[1], act[2])]
                answer = f"reduce {labels.joint_answer(nuc_code, rel)}"
            out.append(Decision("unified_bu", key, unified_bu_prompt(edus, stack[-2], stack[-1], q), answer))
        if act[0] == "shift":
            stack.append((queue, queue + 1))
            queue += 1
        else:
            stack.pop()
            stack.pop()
            stack.append((act[1], act[2]))
    return out


def tree_from_nodes(nodes: list[tuple[int, int, int, str, str]], tree: RstTree, relation_types) -> RstTree:
    """Assemble `(b, s, e, nuclearity code, relation)` nodes over `tree`'s EDUs.
    `RstTree.from_parsing_actions` wants parents before children."""
    ordered = sorted(nodes, key=lambda n: -(n[2] - n[0]))
    actions = [(s, nuc, rel) for _, s, _, nuc, rel in ordered]
    return RstTree.from_parsing_actions(actions, tree.edus, relation_types=relation_types)
