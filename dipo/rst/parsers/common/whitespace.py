"""Whitespace normalization for raw-text (`predict_from_text`) input.

Every training document is EDU text joined by a single space or nothing (see
`tokenize_document` / `reconstruct_text`); no corpus EDU contains a newline, tab or
whitespace run. Raw user text does, and newline subwords (e.g. ModernBERT's `Ċ`) are
then out of distribution: the segmenter can isolate a run of them as its own EDU,
which strips to "" (larc-iu/dipo#1). Collapsing whitespace before encoding puts raw
text back in the training distribution, and is a no-op on corpus text.
"""

import re

_WS = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    """Strip `text` and collapse each internal whitespace run to a single space."""
    return _WS.sub(" ", text).strip()


def _is_cjk(ch: str) -> bool:
    """Scripts written without spaces between words (see `encoding._is_cjk`)."""
    cp = ord(ch)
    return (
        0x3000 <= cp <= 0x303F
        or 0x3040 <= cp <= 0x30FF
        or 0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0xFF00 <= cp <= 0xFFEF
    )


def normalize_whitespace(text: str, drop_cjk_gaps: bool = False) -> tuple[str, list[int]]:
    """`(collapse_whitespace(text), char_map)`, where `char_map[i]` is the index in
    `text` of normalized char `i` (a collapsed run maps to its first char). Lets a
    caller encode the normalized string but slice EDUs out of the original.

    With `drop_cjk_gaps`, a whitespace run between two CJK characters is dropped
    entirely instead of kept as one space. Models trained on unspaced Chinese have
    never seen whitespace between its characters, and a paragraph break there would
    reach them as a stray word-start token. The dropped characters stay in the
    original `text` the caller slices from."""
    out: list[str] = []
    char_map: list[int] = []
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    body = text[start:end]
    for m in re.finditer(r"\s+|\S+", body):
        if m.group().isspace():
            before = body[m.start() - 1] if m.start() > 0 else ""
            after = body[m.end()] if m.end() < len(body) else ""
            if drop_cjk_gaps and before and after and _is_cjk(before) and _is_cjk(after):
                continue
            out.append(" ")
            char_map.append(start + m.start())
        else:
            out.append(m.group())
            char_map.extend(range(start + m.start(), start + m.end()))
    return "".join(out), char_map


_NOT_HEADING_END = re.compile(r"[.!?:;,\u2026\u3002\uff01\uff1f\uff1a\uff1b\uff0c\u3001\u061f\u060c\u061b)\]\u201d\u2019\"'\u00bb\u300b\u300d]$")


def line_start_breaks(text: str) -> list[int]:
    """Character indices in `text` where a new paragraph or a line after a heading starts.

    The parsers collapse every whitespace run to one space, so a paragraph break and a
    heading are invisible to them, and a heading with no closing punctuation gets fused
    with the sentence below it. Gold EDUs never span a paragraph, so these are safe
    places to force a boundary. Conservative on purpose, because a boundary forced in
    the middle of a sentence costs more than a missed one:

      - the first line after a blank line always starts a unit;
      - the line after a single newline starts one only if the line before is
        heading-like: at most 8 words (25 characters for unspaced CJK), no closing
        punctuation, no longer than half the document's longest line, and the new line
        begins with an uppercase letter, a digit, or a character of an uncased script.
        Hard-wrapped prose is left alone, because a wrapped line continues in lowercase.

    Returns the index of the first non-space character of each such line.
    """
    lines: list[tuple[int, str]] = []  # (index of first non-space char, stripped content); blank -> (-1, "")
    pos = 0
    for raw in text.split("\n"):
        stripped = raw.strip()
        lines.append((pos + len(raw) - len(raw.lstrip()), stripped) if stripped else (-1, ""))
        pos += len(raw) + 1
    longest = max((len(c) for _, c in lines), default=0)

    def heading_like(content: str) -> bool:
        cjk = sum(1 for ch in content if _is_cjk(ch)) > len(content) / 2
        too_long = len(content) > 25 if cjk else len(content.split()) > 8
        return not too_long and not _NOT_HEADING_END.search(content) and len(content) <= longest / 2

    def starts_a_sentence(content: str) -> bool:
        first = content[0]
        return first.isupper() or first.isdigit() or not first.islower()

    out: list[int] = []
    prev_content, blank_between = None, False
    for start, content in lines:
        if start < 0:
            blank_between = prev_content is not None
            continue
        if prev_content is not None and (blank_between or (heading_like(prev_content) and starts_a_sentence(content))):
            out.append(start)
        prev_content, blank_between = content, False
    return out
