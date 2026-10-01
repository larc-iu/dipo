"""String <-> RstTree codecs and prompt fragments for the ICL parser.

Everything here is pure text-level (no tokenizer, no torch): the frozen LM emits
a string, and these functions render the worked examples and invert the model's
output back into an `RstTree`. Two serializations (`sr`, `sexp`), each supporting
two shapes:

- `render_e2e` / `parse_e2e`: raw text -> full tree (the model also segments).
- `render_parse` / `parse_over_edus`: a GIVEN numbered EDU list -> structure only.

Plus a segmentation codec (verbatim copy with `||` boundaries) used by the
two-call pipeline's first stage.

The `sr` serialization matches gen's `label_style='words'` form exactly (so the
paper describes ONE scheme for both the fine-tuned and the in-context parser):
content plus `<shift>`, and each merge written as a nuclearity code, the relation
spelled in words, and a `<label_end>` terminator (e.g. `NS elaboration additional
<label_end>`). gen realizes content via a `<copy>` sentinel over an added token,
ICL emits the words directly, but the linearized string is identical. `sexp` (the
unconstrained frontier arm only) uses the `(NUC:relation ...)` / `(EDU text)` form
`RstTree.to_sexp(format="iudex")` produces.
"""

from __future__ import annotations

import re

from iudex.common.log import warn
from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.common.seqgen import relation_to_words

RelationTypes = list[tuple[str, str]]

# gen's words-mode markers. `<shift>` / `<label_end>` are literal strings the
# frozen model reads and writes (not tokenizer specials as in gen).
SHIFT = "<shift>"
LABEL_END = "<label_end>"
_NUCS = ("NS", "SN", "NN")


def word_label(nuc: str, rel: str) -> str:
    """One reduce label in gen's words-mode form: nuclearity code, the relation
    spelled in words (`relation_to_words`, shared with gen so the spelling can
    never drift), then the `<label_end>` terminator. The terminator keeps the
    label set prefix-free (`elaboration` is a prefix of `elaboration-additional`),
    so any label set is unambiguously decodable."""
    return f"{nuc} {relation_to_words(rel)} {LABEL_END}"


# ---------------------------------------------------------------------------
# Relation inventories (prompt fragments) and the reduce-token map (decoder)
# ---------------------------------------------------------------------------


def _words_to_rel(relation_types: RelationTypes) -> dict[str, str]:
    """Map a label's relation-words back to the canonical hyphenated relation
    (for decoding words-mode labels). `relation_to_words` is injective over a
    relation inventory, so this inverts it."""
    return {relation_to_words(rel): rel for rel, _ in relation_types}


def sr_relation_inventory(relation_types: RelationTypes) -> str:
    lines = []
    for rel, kind in relation_types:
        if kind == "multinuc":
            lines.append(f"- {word_label('NN', rel)}  (multinuclear, both children are nuclei of '{rel}')")
        else:
            lines.append(f"- {word_label('NS', rel)}  (left child nucleus, right satellite)")
            lines.append(f"- {word_label('SN', rel)}  (left child satellite, right nucleus)")
    return "\n".join(lines)


def sexp_relation_inventory(relation_types: RelationTypes) -> str:
    lines = []
    for rel, kind in relation_types:
        if kind == "multinuc":
            lines.append(f"- NN:{rel}  (multinuclear; both children are nuclei)")
        else:
            lines.append(f"- NS:{rel}  (left nucleus, right satellite)")
            lines.append(f"- SN:{rel}  (left satellite, right nucleus)")
    return "\n".join(lines)


def relation_set(relation_types: RelationTypes) -> set[str]:
    return {rel for rel, _ in relation_types}


# ---------------------------------------------------------------------------
# Prompt input formatting (shared across serializations)
# ---------------------------------------------------------------------------


def doc_text(edus: list[str]) -> str:
    """Raw document input = single-space-joined EDUs (the trained parsers' convention)."""
    return " ".join(edus)


def numbered_edus(edus: list[str]) -> str:
    """1-indexed EDU list. Only needed when the OUTPUT references EDU numbers,
    which is now just `sexp` without text echoing (its leaves are bare integers)."""
    return "\n".join(f"{i}. {e}" for i, e in enumerate(edus, 1))


def delimited_edus(edus: list[str]) -> str:
    """The GIVEN-EDU input for the parse stage: the same ` || ` form stage 1 emits.

    Preferred over `numbered_edus` wherever the output does not cite EDU numbers.
    The numbering was pure input-side overhead once shifts began re-emitting EDU
    text -- the action stream contains no indices at all -- and it cost ~1.1k
    tokens (26% of this input) on RST-DT's longest document, in digit tokens that
    carry no discourse information. Reusing stage 1's delimiter also removes a
    gratuitous reformat: the parse call now consumes exactly what the
    segmentation call produces."""
    return " || ".join(edus)


# ---------------------------------------------------------------------------
# Segmentation codec (two-call stage 1): verbatim copy with " || " boundaries
# ---------------------------------------------------------------------------

SEGMENTATION_TASK = """\
You are doing Rhetorical Structure Theory (RST) discourse segmentation. Split the
document into Elementary Discourse Units (EDUs, roughly clause-level spans).

Reproduce the input document VERBATIM, inserting " || " at every boundary between
adjacent EDUs. Copy every word exactly, in order. Do NOT paraphrase, drop, merge,
or reorder any text. The ONLY characters you add are the " || " delimiters. EDUs
are fine-grained: split at clause boundaries, including subordinate clauses,
relative clauses, and many non-finite (to-infinitive and -ing) clauses, not just
at sentence boundaries.

Output ONLY the delimited text, no commentary.
"""


def render_segmentation(edus: list[str]) -> str:
    return " || ".join(edus)


def parse_segmentation(response: str) -> list[str]:
    """`||`-delimited response -> EDU list. Strips light fences / lead-ins."""
    text = response.strip()
    for marker in ("SEGMENTS:", "OUTPUT:", "Output:", "```text", "```"):
        if text.startswith(marker):
            text = text[len(marker) :].lstrip()
    if text.endswith("```"):
        text = text[: -len("```")].rstrip()
    edus = [s.strip() for s in text.split("||")]
    edus = [e for e in edus if e]
    if not edus:
        raise ValueError("segmentation produced no EDUs")
    return edus


# ---------------------------------------------------------------------------
# Serialization strategies
# ---------------------------------------------------------------------------


class Serialization:
    """Interface: a serialization renders worked examples and inverts responses.
    `relation_types` is passed where the legal label set is needed."""

    name: str

    def e2e_task_description(self, relation_types: RelationTypes) -> str:
        raise NotImplementedError

    def render_e2e(self, tree: RstTree) -> str:
        raise NotImplementedError

    def parse_e2e(self, response: str, relation_types: RelationTypes) -> RstTree:
        raise NotImplementedError

    def parse_task_description(self, relation_types: RelationTypes, *, echo_edu_text: bool = False) -> str:
        raise NotImplementedError

    def parse_input(self, edus: list[str], *, echo_edu_text: bool = False) -> str:
        """Format the GIVEN EDUs for the parse call. Numbering is only warranted
        when the output cites EDU numbers, so each serialization decides."""
        return delimited_edus(edus)

    def render_parse(self, tree: RstTree, *, echo_edu_text: bool = False) -> str:
        raise NotImplementedError

    def parse_over_edus(
        self, response: str, edus: list[str], relation_types: RelationTypes, *, echo_edu_text: bool = False
    ) -> RstTree:
        raise NotImplementedError


_NUCLEARITY_BLURB = """\
Nuclearity patterns:
  - NS: left child is the nucleus, right child is the satellite
  - SN: left child is the satellite, right child is the nucleus
  - NN: multinuclear, both children are nuclei (symmetric relations)"""


class SrSerialization(Serialization):
    name = "sr"

    _LABEL_BLURB = (
        'A REDUCE combines the top two stack items into a parent, written as its '
        'nuclearity code (NS, SN, or NN), the relation spelled in words, then '
        '"<label_end>". For example "NS elaboration additional <label_end>".'
    )

    def e2e_task_description(self, relation_types: RelationTypes) -> str:
        return f"""\
You are doing Rhetorical Structure Theory (RST) discourse parsing: a document is a
binary tree over Elementary Discourse Units (EDUs, roughly clause-level spans).
{_NUCLEARITY_BLURB}

Serialize the parse as a SHIFT-REDUCE action sequence, reading left to right:
  - Emit the next EDU's whitespace-split words, then "<shift>", to push that EDU.
  - {self._LABEL_BLURB}

For N EDUs there are exactly N "<shift>" actions and N-1 reduces. The only legal
reduce labels are:

{sr_relation_inventory(relation_types)}

Output ONLY the action sequence, no commentary.
"""

    def render_e2e(self, tree: RstTree) -> str:
        parts: list[str] = []
        for a in tree.to_shift_reduce(include_text=True):
            if isinstance(a, Shift):
                parts += [a.edu_text, SHIFT]
            else:
                parts.append(word_label(a.nuc, a.rel))
        return " ".join(parts)

    def parse_e2e(self, response: str, relation_types: RelationTypes) -> RstTree:
        actions = _parse_sr_word_actions(_strip_fences(response).split(), relation_types, with_content=True)
        if not any(isinstance(a, Shift) for a in actions):
            raise ValueError("sr e2e: no <shift> actions in output")
        return RstTree.from_shift_reduce(actions, relation_types=relation_types)

    def parse_task_description(self, relation_types: RelationTypes, *, echo_edu_text: bool = False) -> str:
        shift_blurb = (
            'Emit the next EDU\'s words verbatim, then "<shift>", to push it (EDUs are\n'
            "    consumed in the listed order, so the text you emit is fully determined)."
            if echo_edu_text
            else '"<shift>" pushes the next EDU (EDUs are consumed in the listed order, so you do\n'
            "    NOT repeat their text)."
        )
        return f"""\
You are doing RST discourse parsing. You are GIVEN the document's EDUs in order,
separated by " || ", and must produce ONLY the tree structure over them.
{_NUCLEARITY_BLURB}

Emit a bottom-up SHIFT-REDUCE action sequence over the given EDUs, in order:
  - {shift_blurb}
  - {self._LABEL_BLURB}

For N EDUs there are exactly N "<shift>" actions and N-1 reduces. The only legal
reduce labels are:

{sr_relation_inventory(relation_types)}

Output ONLY the action sequence, no commentary.
"""

    def render_parse(self, tree: RstTree, *, echo_edu_text: bool = False) -> str:
        if echo_edu_text:
            # Identical to render_e2e: the parse stage stops being a content-free
            # variant, so one serialization describes both stages (and gen).
            return self.render_e2e(tree)
        parts = [SHIFT if isinstance(a, Shift) else word_label(a.nuc, a.rel) for a in tree.to_shift_reduce()]
        return " ".join(parts)

    def parse_over_edus(
        self, response: str, edus: list[str], relation_types: RelationTypes, *, echo_edu_text: bool = False
    ) -> RstTree:
        actions = _parse_sr_word_actions(
            _strip_fences(response).split(), relation_types, with_content=echo_edu_text
        )
        n_shift = sum(isinstance(a, Shift) for a in actions)
        if n_shift != len(edus):
            raise ValueError(f"sr parse: {n_shift} <shift> actions for {len(edus)} given EDUs")
        if echo_edu_text:
            # The echoed text is grammar-forced, so a mismatch means the response was
            # NOT actually constrained (or the EDU list drifted). Check rather than
            # assume: silently accepting misaligned text would score a tree built over
            # the wrong units. Whitespace is normalized because the echo is
            # whitespace-joined while the given EDU may carry its original spacing.
            echoed = [a.edu_text for a in actions if isinstance(a, Shift)]
            for i, (got, want) in enumerate(zip(echoed, edus, strict=True)):
                if got.split() != want.split():
                    raise ValueError(
                        f"sr parse: echoed text for EDU {i + 1} does not match the given EDU "
                        f"({got[:60]!r} vs {want[:60]!r}); the response was not grammar-constrained."
                    )
        return RstTree.from_shift_reduce(actions, edus=edus, relation_types=relation_types)


class SexpSerialization(Serialization):
    name = "sexp"

    def e2e_task_description(self, relation_types: RelationTypes) -> str:
        return f"""\
You are doing Rhetorical Structure Theory (RST) discourse parsing: a document is a
binary tree over Elementary Discourse Units (EDUs, roughly clause-level spans).
{_NUCLEARITY_BLURB}

Serialize the parse as one S-expression. Each internal node is
`(NUC:relation CHILD1 CHILD2)` with exactly two children. Each leaf is
`(EDU surface text)`. Literal parentheses inside EDU text are escaped `-LRB-`/`-RRB-`.
For N EDUs there are N `(EDU ...)` leaves and N-1 internal nodes. Legal head tags:

{sexp_relation_inventory(relation_types)}

Output ONLY the S-expression, no commentary.
"""

    def render_e2e(self, tree: RstTree) -> str:
        return tree.to_sexp(format="iudex")

    def parse_e2e(self, response: str, relation_types: RelationTypes) -> RstTree:
        node = _read_sexp(_strip_fences(response))
        actions, edus = _sexp_to_actions_textleaf(node)
        if not actions:  # single-EDU degenerate tree (no internal node)
            return RstTree.from_shift_reduce([Shift()], edus=edus, relation_types=relation_types)
        _check_relations(actions, relation_set(relation_types))
        return RstTree.from_parsing_actions(actions, edus, relation_types=relation_types)

    def parse_task_description(self, relation_types: RelationTypes, *, echo_edu_text: bool = False) -> str:
        leaf_blurb = (
            "whose LEAVES are `(EDU surface text)`, repeating each given EDU verbatim in\nthe given order"
            if echo_edu_text
            else "whose LEAVES are the EDU numbers (bare\nintegers), in the given order"
        )
        leaf_count = (
            "For N EDUs there are exactly N `(EDU ...)` leaves left to right, and"
            if echo_edu_text
            else "For N EDUs the leaves are exactly 1..N left to right, and"
        )
        given = (
            'the document\'s EDUs in order, separated by " || "'
            if echo_edu_text
            else "a numbered list of EDUs"
        )
        return f"""\
You are doing RST discourse parsing. You are GIVEN {given} and must
produce ONLY the tree structure over them.
{_NUCLEARITY_BLURB}

Serialize the tree as one S-expression {leaf_blurb}. Each internal node is
`(NUC:relation CHILD1 CHILD2)` with exactly two children. {leaf_count}
there are N-1 internal nodes. Legal head tags:

{sexp_relation_inventory(relation_types)}

Output ONLY the S-expression, no commentary.
"""

    def parse_input(self, edus: list[str], *, echo_edu_text: bool = False) -> str:
        # Without echo this serialization's leaves ARE the EDU numbers, so the
        # input has to carry them; with echo the leaves are text and it need not.
        return delimited_edus(edus) if echo_edu_text else numbered_edus(edus)

    def render_parse(self, tree: RstTree, *, echo_edu_text: bool = False) -> str:
        if echo_edu_text:
            # Same convergence as `sr`: the parse form becomes the e2e form.
            return self.render_e2e(tree)
        # to_sexp(include_text=False) emits a bare `<edu>` per leaf in document
        # order (leaves are always left-to-right); number them 1..N.
        sexp = tree.to_sexp(traversal_order="preorder", include_text=False)
        parts = sexp.split("<edu>")
        out = parts[0]
        for i, seg in enumerate(parts[1:], 1):
            out += str(i) + seg
        return out

    def parse_over_edus(
        self, response: str, edus: list[str], relation_types: RelationTypes, *, echo_edu_text: bool = False
    ) -> RstTree:
        if len(edus) == 1 and not echo_edu_text:  # render_parse emits a bare "1"; no bracketed node
            return RstTree.from_shift_reduce([Shift()], edus=edus, relation_types=relation_types)
        node = _read_sexp(_strip_fences(response))
        if echo_edu_text:
            # Text leaves, so read them the way e2e does and then check the surface
            # against the given EDUs -- sexp is never grammar-constrained, so unlike
            # `sr` the leaves really can drift and must be validated, not trusted.
            actions, got_edus = _sexp_to_actions_textleaf(node)
            if len(got_edus) != len(edus):
                raise ValueError(f"sexp parse: {len(got_edus)} EDU leaves for {len(edus)} given EDUs")
            for i, (got, want) in enumerate(zip(got_edus, edus, strict=True)):
                if got.split() != want.split():
                    raise ValueError(
                        f"sexp parse: leaf {i + 1} does not match the given EDU "
                        f"({got[:60]!r} vs {want[:60]!r})."
                    )
            if not actions:  # single-EDU degenerate tree (no internal node)
                return RstTree.from_shift_reduce([Shift()], edus=edus, relation_types=relation_types)
        else:
            actions = _sexp_to_actions_indexleaf(node, len(edus))
        _check_relations(actions, relation_set(relation_types))
        return RstTree.from_parsing_actions(actions, edus, relation_types=relation_types)


_SERIALIZATIONS: dict[str, Serialization] = {"sr": SrSerialization(), "sexp": SexpSerialization()}


def get_serialization(name: str) -> Serialization:
    if name not in _SERIALIZATIONS:
        raise ValueError(f"unknown serialization {name!r} (have {sorted(_SERIALIZATIONS)})")
    return _SERIALIZATIONS[name]


# ---------------------------------------------------------------------------
# Response cleanup + sexp reader (shared)
# ---------------------------------------------------------------------------


def _strip_fences(response: str) -> str:
    text = response.strip()
    for marker in ("OUTPUT:", "Output:", "```text", "```sexp", "```scheme", "```lisp", "```"):
        if text.startswith(marker):
            text = text[len(marker) :].lstrip()
    if text.endswith("```"):
        text = text[: -len("```")].rstrip()
    return text


def _match_word_label(tokens: list[str], i: int, words_to_rel: dict[str, str]):
    """If `tokens[i:]` begins a `<nuc> <relation words> <label_end>` label, return
    `(nuc, rel, end_exclusive)`, else None. The lookahead (a nuclearity code, then
    known relation words, then the terminator) is what disambiguates a real label
    from ordinary content that merely starts with `NS`/`SN`/`NN`."""
    if tokens[i] not in _NUCS:
        return None
    j = i + 1
    words: list[str] = []
    while j < len(tokens) and tokens[j] != LABEL_END:
        if tokens[j] == SHIFT:  # a shift before the terminator means this was not a label
            return None
        words.append(tokens[j])
        j += 1
    if j >= len(tokens):  # ran off the end with no terminator
        return None
    rel = words_to_rel.get(" ".join(words))
    return None if rel is None else (tokens[i], rel, j + 1)


def _parse_sr_word_actions(tokens: list[str], relation_types: RelationTypes, *, with_content: bool):
    """Decode a words-mode SR token stream into shift/reduce actions. Reduces are
    `<nuc> <relation words> <label_end>`. With `with_content`, tokens that are not
    part of an action accumulate as the next EDU's surface text (e2e); without it
    (parse-over-given-EDUs) stray tokens are dropped."""
    words_to_rel = _words_to_rel(relation_types)
    actions: list = []
    content: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == SHIFT:
            actions.append(Shift(edu_text=" ".join(content)) if with_content else Shift())
            content = []
            i += 1
            continue
        label = _match_word_label(tokens, i, words_to_rel)
        if label is not None:
            nuc, rel, i = label
            actions.append(Reduce(nuc=nuc, rel=rel))
            continue
        if with_content:
            content.append(tokens[i])
        i += 1
    return actions


def _tokenize_sexp(s: str) -> list[str]:
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c in "(){}":
            out.append(c)
            i += 1
        else:
            j = i
            while j < n and not s[j].isspace() and s[j] not in "(){}":
                j += 1
            out.append(s[i:j])
            i = j
    return out


def _read_sexp(text: str):
    """Trim to the first balanced top-level expression, tokenize, parse to a
    nested tuple. Accepts `(` or `{` brackets (words-mode uses braces)."""
    first = next((i for i, c in enumerate(text) if c in "({"), -1)
    if first == -1:
        raise ValueError("sexp: no opening bracket in output")
    text = text[first:]
    depth, end = 0, None
    for i, c in enumerate(text):
        if c in "({":
            depth += 1
        elif c in ")}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("sexp: unbalanced brackets in output")
    tokens = _tokenize_sexp(text[:end])
    node, pos = _parse_sexp_node(tokens, 0)
    return node


_OPEN, _CLOSE = "({", ")}"


def _parse_sexp_node(tokens: list[str], pos: int):
    """Returns (node, new_pos). Node is ("edu", text) | ("index", int) |
    ("node", nuc, rel, left, right). A `(EDU ...)` head means a text leaf; a bare
    integer token means an index leaf; anything with a `:` is an internal node."""
    if pos >= len(tokens) or tokens[pos] not in _OPEN:
        # Bare integer = index leaf (parse-over-edus format).
        if pos < len(tokens) and tokens[pos].isdigit():
            return ("index", int(tokens[pos])), pos + 1
        raise ValueError(f"sexp: expected '(' or index at token {pos}, got {tokens[pos : pos + 1]!r}")
    pos += 1
    if pos >= len(tokens):
        raise ValueError("sexp: unexpected end after '('")
    head = tokens[pos]
    pos += 1
    if head == "EDU":
        text_tokens: list[str] = []
        while pos < len(tokens) and tokens[pos] not in _CLOSE:
            text_tokens.append(tokens[pos])
            pos += 1
        if pos >= len(tokens):
            raise ValueError("sexp: unclosed EDU leaf")
        pos += 1  # consume close
        text = " ".join(text_tokens).replace("-LRB-", "(").replace("-RRB-", ")")
        return ("edu", text), pos
    if ":" not in head:
        raise ValueError(f"sexp: internal node head missing ':': {head!r}")
    nuc, rel = head.split(":", 1)
    if nuc not in ("NS", "SN", "NN"):
        raise ValueError(f"sexp: unknown nuclearity {nuc!r}")
    left, pos = _parse_sexp_node(tokens, pos)
    right, pos = _parse_sexp_node(tokens, pos)
    if pos >= len(tokens) or tokens[pos] not in _CLOSE:
        raise ValueError(f"sexp: expected ')' to close internal node at {pos}")
    return ("node", nuc, rel, left, right), pos + 1


def _sexp_to_actions(node, leaf_text):
    """Post-order flatten to (parsing_actions, edu_strings). `leaf_text(leaf_node,
    running_index) -> str` yields each leaf's surface form; the running index is
    the leaf's document position (0-based)."""
    edus: list[str] = []

    def visit(n):
        if n[0] in ("edu", "index"):
            idx = len(edus)
            edus.append(leaf_text(n, idx))
            return [idx], []
        _, nuc, rel, left, right = n
        l_idx, l_acts = visit(left)
        r_idx, r_acts = visit(right)
        return (l_idx + r_idx, [(r_idx[0], nuc, rel)] + l_acts + r_acts)

    _, actions = visit(node)
    return actions, edus


def _sexp_to_actions_textleaf(node):
    return _sexp_to_actions(node, lambda n, i: n[1])


def _sexp_to_actions_indexleaf(node, n_edus: int):
    """Index-leaf variant: validate the leaves are exactly 1..N in document
    order, then return the parsing actions. Surface forms come from the caller's
    given EDUs positionally, so only the ordering is checked here."""
    seen: list[int] = []

    def leaf_text(n, i):
        if n[0] != "index":
            raise ValueError("sexp parse: expected an integer leaf, got EDU text")
        seen.append(n[1])
        return str(n[1])

    actions, _ = _sexp_to_actions(node, leaf_text)
    if seen != list(range(1, n_edus + 1)):
        raise ValueError(f"sexp parse: leaves must be 1..{n_edus} in order, got {seen}")
    return actions


def _check_relations(actions, rel_set: set[str]) -> None:
    for _, _, rel in actions:
        if rel not in rel_set:
            raise ValueError(f"sexp: unknown relation {rel!r}")


# ---------------------------------------------------------------------------
# Grammar-constrained decoding (the icl-c arm)
#
# GBNF (llama.cpp's grammar dialect). Both two-call stages are grammar-shaped:
# segmentation is a regular language (verbatim copy with optional breaks), and
# the parse-stage SR sequence is a compact non-left-recursive CFG. A grammar
# turns the model's free generation into a guaranteed-valid output, which is the
# whole point of the constrained arm.
# ---------------------------------------------------------------------------


def _gbnf_str(s: str) -> str:
    """Escape a Python string as a GBNF double-quoted literal."""
    out = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{out}"'


def segmentation_grammar(text: str) -> str:
    """GBNF forcing a VERBATIM copy of `text` (whitespace-tokenized) with a break
    choice (` ` vs ` || `) at each token boundary. The model cannot drop, add, or
    reorder words, so segmentation collapses to break placement (the ~solved part
    per the pilot). One-token documents produce a single literal, no gaps."""
    tokens = text.split()
    if not tokens:
        raise ValueError("segmentation_grammar: empty document")
    pieces = [_gbnf_str(tokens[0])]
    for t in tokens[1:]:
        pieces += ["gap", _gbnf_str(t)]
    return "root ::= " + " ".join(pieces) + '\ngap ::= " " | " || "'


def prefixed_root_grammar(gbnf: str, prefix: str) -> str:
    """Re-root `gbnf` behind a literal `prefix`, i.e. `root ::= "<prefix>" <old root>`.

    This exists for lazily-triggered grammars (the thinking icl-c arm). llama.cpp
    feeds the trigger text itself INTO the grammar when the trigger fires
    (`llama_grammar_accept_token` on the trigger token, and the pattern path
    replays the matched substring), so a grammar triggered on `</think>` must
    accept `</think>` as its first piece or the stack empties immediately.

    Both builders here emit `root` on the first line and never reference it on a
    right-hand side, which is what makes the rename safe; that is asserted rather
    than assumed."""
    lines = gbnf.split("\n")
    if not lines or not lines[0].startswith("root ::= "):
        raise ValueError("prefixed_root_grammar: expected the first line to define `root`")
    for line in lines[1:]:
        rhs = line.split("::=", 1)[1] if "::=" in line else line
        # Only a rule REFERENCE blocks re-rooting. Quoted literals must be stripped
        # first, because with echo_edu_text the right-hand sides carry arbitrary
        # document text -- wsj_1387 contains the EDU "since it took root as cheap
        # entertainment ...", which otherwise reads as a reference and rejects a
        # perfectly re-rootable grammar.
        rhs_no_literals = re.sub(r'"(?:[^"\\]|\\.)*"', "", rhs)
        if re.search(r"\broot\b", rhs_no_literals):
            raise ValueError("prefixed_root_grammar: `root` is referenced on a right-hand side; cannot re-root")
    body = lines[0][len("root ::= ") :]
    return "\n".join([f"root ::= {_gbnf_str(prefix)} answer-root", f"answer-root ::= {body}", *lines[1:]])


def _reduce_rule(relation_types: RelationTypes) -> str:
    # Each reduce is a full words-mode label. Trailing space so concatenated
    # actions stay whitespace-separated.
    labels = []
    for rel, kind in relation_types:
        for nuc in (("NN",) if kind == "multinuc" else ("NS", "SN")):
            labels.append(word_label(nuc, rel) + " ")
    return "reduce ::= " + " | ".join(_gbnf_str(lbl) for lbl in sorted(labels))


def sr_parse_grammar(
    n_edus: int, relation_types: RelationTypes, *, max_exact: int = 512, edus: list[str] | None = None
) -> str:
    """GBNF for a valid parse-stage SR sequence over `n_edus` EDUs.

    For `n_edus <= max_exact` the grammar is indexed by the automaton state
    `(shifts_done s, stack_depth d)` with right-linear rules
    `r_s_d ::= shift r_{s+1}_{d+1} | reduce r_s_{d-1}` (and the empty string only
    at the accepting state `(n_edus, 1)`). This forces EXACTLY n_edus shifts and
    n_edus-1 reduces in valid stack order (a guaranteed, correctly-sized tree).
    Crucially it is UNAMBIGUOUS: the emitted prefix uniquely fixes `(s, d)`, so
    only one grammar stack is ever live. The natural counting CFG
    `s_k ::= s_i s_{k-i} reduce` is ambiguous in the split point, which makes
    llama.cpp track a combinatorial set of stacks and grinds to a halt.

    Above the cap the O(n^2) rule count is unwieldy, so it falls back to a compact
    any-length valid-tree grammar (`tree ::= shift rest; rest ::= "" | tree reduce
    rest`): every accepted string is still a structurally valid tree, but the leaf
    count leans on the prompt. All formulations are non-left-recursive (GBNF).

    `max_exact` must clear the PREDICTED EDU count, not the gold one. Segmentation
    over-segments -- wsj_1146 (304 gold EDUs, the largest in RST-DT) came back with
    352 -- so a cap set at the largest gold document silently drops the exact-count
    guarantee on exactly the hardest inputs, which is the whole point of the arm.
    512 leaves ~68% headroom; measured on the 397B server, an n=512 grammar is
    5.9 MB and still compiles and decodes in ~2s, so the old 320 was over-cautious."""
    if n_edus < 1:
        raise ValueError(f"sr_parse_grammar: n_edus must be >= 1 (got {n_edus})")
    if edus is not None and len(edus) != n_edus:
        raise ValueError(f"sr_parse_grammar: got {len(edus)} edus for n_edus={n_edus}")
    base = ['shift ::= "<shift> "', _reduce_rule(relation_types)]
    if edus is not None:
        # Echo mode: the shift at state (s, d) consumes EDU s+1, and the state
        # already determines s, so the EDU's words can be FORCED as a literal --
        # the model gets grounding in what it is attaching without gaining any
        # freedom to paraphrase or skip. One rule per EDU rather than inlining the
        # text at every state: state (s, d) exists for each depth d, so inlining
        # would repeat EDU s+1's text ~s times and double the grammar (measured
        # 4.10 MB vs 2.22 MB on wsj_1146, against 2.02 MB for bare shifts).
        base = [f"shift-{k + 1} ::= {_gbnf_str(e + ' ' + SHIFT + ' ')}" for k, e in enumerate(edus)]
        base.append(_reduce_rule(relation_types))
    if n_edus > max_exact:
        warn(
            f"sr_parse_grammar: {n_edus} EDUs exceeds max_exact={max_exact}, falling back to the "
            "any-length valid-tree grammar (a valid tree is still guaranteed, but not the exact EDU count)."
        )
        if edus is not None:
            # The any-length fallback has no state to identify the next EDU, so it
            # cannot force per-EDU text; fall back to bare shifts and say so.
            warn("sr_parse_grammar: echo_edu_text is not available above max_exact; using bare shifts.")
            base = ['shift ::= "<shift> "', _reduce_rule(relation_types)]
        return "\n".join(["root ::= tree", "tree ::= shift rest", 'rest ::= "" | tree reduce rest', *base])
    # State name s{shifts}d{depth}. GBNF rule names are alphanumeric+dash only
    # (no underscore), so the separator is the letter 'd'.
    def st(s: int, d: int) -> str:
        return f"s{s}d{d}"

    rules = [f"root ::= {st(0, 0)}"]
    for s in range(n_edus + 1):
        for d in range(0 if s == 0 else 1, s + 1):
            alts = []
            if s < n_edus:
                shift_ref = f"shift-{s + 1}" if edus is not None else "shift"
                alts.append(f"{shift_ref} {st(s + 1, d + 1)}")
            if d >= 2:
                alts.append(f"reduce {st(s, d - 1)}")
            if s == n_edus and d == 1:
                alts.append('""')
            rules.append(f"{st(s, d)} ::= " + " | ".join(alts))
    return "\n".join([*rules, *base])
