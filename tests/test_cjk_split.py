"""Unspaced Chinese: EDU boundaries inside a subword (`split_cjk_tokens`).

Raw Chinese has no spaces, so with prefix-glued EDUs a boundary can fall inside a
SentencePiece token. On GCDT with XLM-R that happens at ~3% of boundaries, and a
one-character EDU absorbed into its neighbour's subword maps to an empty token
span, which `tokenize_document` rejects (12 of 50 documents). `split_cjk` re-emits
multi-character CJK subwords one character per token.
"""

import pytest
import torch

import re

from dipo.rst.parsers.common.encoding import encode_with_offsets, tokenize_document
from dipo.rst.parsers.common.whitespace import normalize_whitespace
from dipo.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig
from dipo.rst.parsers.dmrst.modeling_dmrst import edus_from_breaks

TOKENIZER = "xlm-roberta-base"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(TOKENIZER)


def test_off_is_the_plain_tokenizer(tok):
    text = "人口老龄化进程在世界范围内急剧发展，"
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encode_with_offsets(tok, text)
    assert ids == enc["input_ids"]
    assert offsets == [tuple(o) for o in enc["offset_mapping"]]


def test_every_cjk_character_starts_a_token(tok):
    text = "人口老龄化进程在世界范围内急剧发展，但 Antonín Dvořák 仍然"
    ids, offsets = encode_with_offsets(tok, text, split_cjk=True)
    starts = {s for s, _ in offsets}
    assert all(i in starts for i, ch in enumerate(text) if "一" <= ch <= "鿿")
    # Latin runs keep their ordinary subwords.
    latin = text.index("Antonín"), text.index(" 仍然")
    plain = list(zip(*encode_with_offsets(tok, text)))
    split = list(zip(ids, offsets))
    in_latin = lambda pairs: [p for p in pairs if latin[0] <= p[1][0] < latin[1]]
    assert in_latin(split) == in_latin(plain)


def test_fullwidth_punctuation_is_split_after_normalization(tok):
    """XLM-R normalizes `：“` to the single piece `:“`; split it anyway."""
    ids, offsets = encode_with_offsets(tok, "他说：“好”", split_cjk=True)
    assert (2, 3) in offsets and (3, 4) in offsets


def test_one_character_edu_no_longer_maps_to_an_empty_span(tok):
    """`在这` is a single XLM-R subword, so a glued EDU `在` followed by `这...`
    has no token of its own unless CJK subwords are split."""
    edus = ["我们", "在", "这里工作"]
    prefixes = ["", "", ""]
    with pytest.raises(ValueError, match="empty token span"):
        tokenize_document(tok, edus, torch.device("cpu"), prefixes=prefixes)
    ids, spans = tokenize_document(tok, edus, torch.device("cpu"), prefixes=prefixes, split_cjk=True)
    assert len(spans) == 3 and all(e > s for s, e in spans)
    assert spans[-1][1] == len(ids)


def test_config_default_is_off():
    """Off by default, so every existing checkpoint tokenizes exactly as it was trained."""
    import dataclasses

    field = {f.name: f for f in dataclasses.fields(DMRSTConfig)}["split_cjk_tokens"]
    assert field.default is False


def test_whitespace_between_cjk_characters_is_dropped_only_on_request():
    text = "他说：“好”。\n\n研究表明，Hello world.\n\nNext 段 落"
    assert normalize_whitespace(text)[0] == "他说：“好”。 研究表明，Hello world. Next 段 落"
    normed, char_map = normalize_whitespace(text, drop_cjk_gaps=True)
    assert normed == "他说：“好”。研究表明，Hello world. Next 段落"
    # Every kept character still maps back to the same character of the original.
    assert all(text[char_map[i]] == ch or ch == " " for i, ch in enumerate(normed))


def test_edu_text_never_duplicates_a_character_the_lone_word_marker_overlaps(tok):
    """XLM-R gives a lone `▁` the offset of the character after it, so that character
    is covered by both the `▁` and the next token. A break between them used to copy
    it into two EDUs ("研" | "研究表明，")."""
    text = "他说：“好”。\n\n研究表明，老年人理解言语的能力明显下降。"
    src, char_map = normalize_whitespace(text)
    ids, offsets = encode_with_offsets(tok, src, split_cjk=True)
    marker = next(k for k, (s, e) in enumerate(offsets) if tok.convert_ids_to_tokens(ids[k]) == "\u2581" and k > 0)
    assert offsets[marker] == offsets[marker + 1], "the premise: the marker overlaps the next token"
    for breaks in ([marker, len(ids) - 1], [marker - 1, marker, len(ids) - 1], [len(ids) - 1]):
        mapping, edus = edus_from_breaks(text, char_map, offsets, breaks)
        assert all(e.strip() for e in edus), edus
        assert re.sub(r"\s+", "", "".join(edus)) == re.sub(r"\s+", "", text), edus
        assert len(mapping) == len(edus)
        assert mapping[0][0] == 0 and mapping[-1][1] == len(ids)
        assert all(a[1] == b[0] for a, b in zip(mapping, mapping[1:])), "token spans stay contiguous"
