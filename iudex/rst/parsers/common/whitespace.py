"""Whitespace normalization for raw-text (`predict_from_text`) input.

Every training document is EDU text joined by a single space or nothing (see
`tokenize_document` / `reconstruct_text`); no corpus EDU contains a newline, tab or
whitespace run. Raw user text does, and newline subwords (e.g. ModernBERT's `Ċ`) are
then out of distribution: the segmenter can isolate a run of them as its own EDU,
which strips to "" (larc-iu/iudex#1). Collapsing whitespace before encoding puts raw
text back in the training distribution, and is a no-op on corpus text.
"""

import re

_WS = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    """Strip `text` and collapse each internal whitespace run to a single space."""
    return _WS.sub(" ", text).strip()


def normalize_whitespace(text: str) -> tuple[str, list[int]]:
    """`(collapse_whitespace(text), char_map)`, where `char_map[i]` is the index in
    `text` of normalized char `i` (a collapsed run maps to its first char). Lets a
    caller encode the normalized string but slice EDUs out of the original."""
    out: list[str] = []
    char_map: list[int] = []
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    for m in re.finditer(r"\s+|\S+", text[start:end]):
        if m.group().isspace():
            out.append(" ")
            char_map.append(start + m.start())
        else:
            out.append(m.group())
            char_map.extend(range(start + m.start(), start + m.end()))
    return "".join(out), char_map
