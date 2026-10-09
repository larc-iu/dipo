"""Raw-text input with line breaks (larc-iu/dipo#1) and pre-curriculum checkpoints.

Newlines in `predict_from_text` input are out of distribution (every training doc is
single-space joined), and DMRST's segmenter could isolate a run of `Ċ` subwords as an
EDU that then stripped to "". Input is now whitespace-collapsed before encoding, EDUs
are still sliced from the caller's text, and a whitespace-only span can never
surface as an EDU.

Separately, every larc-iu Hub checkpoint predates 0c0345f and carries `max_epochs` /
`checkpoint_every`, which tonga rejected, so none of them loaded.
"""

import pytest

from dipo.rst.parsers.common.inference import migrate_checkpoint_config
from dipo.rst.parsers.common.whitespace import collapse_whitespace, line_start_breaks, normalize_whitespace
from dipo.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig
from dipo.rst.parsers.dmrst.modeling_dmrst import edus_from_breaks, with_forced_breaks

TOKENIZER = "jhu-clsp/ettin-encoder-400m"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(TOKENIZER)


def _offsets(tok, s):
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    return enc["offset_mapping"]


def test_normalize_whitespace_maps_back_to_the_original():
    text = "\n Title\n\nBody  text.\tMore\r\n"
    normed, char_map = normalize_whitespace(text)
    assert normed == "Title Body text. More"
    assert len(char_map) == len(normed)
    for i, ch in enumerate(normed):
        assert text[char_map[i]] == ch or (ch == " " and text[char_map[i]].isspace())


def test_normalize_whitespace_is_a_noop_on_corpus_shaped_text():
    """Corpus documents are single-space joined, so eval paths must be untouched."""
    text = "Hello , world . Second EDU here; و‌ها"
    normed, char_map = normalize_whitespace(text)
    assert normed == text
    assert char_map == list(range(len(text)))
    assert collapse_whitespace(text) == text


def test_whitespace_only_span_merges_left(tok):
    """Raw (unnormalized) encoding reproduces the issue's `ĊĊ`-only EDU."""
    text = "First paragraph ends.\n\nSecond one."
    offsets = _offsets(tok, text)
    toks = tok.convert_ids_to_tokens(tok(text, add_special_tokens=False)["input_ids"])
    nl = [i for i, t in enumerate(toks) if not t.replace("Ċ", "").replace("Ġ", "")]
    assert nl, f"expected whitespace-only subwords in {toks}"
    breaks = [nl[0] - 1, nl[-1], len(toks) - 1]  # isolate the newline run as an EDU
    mapping, texts = edus_from_breaks(text, list(range(len(text))), offsets, breaks)
    assert texts == ["First paragraph ends.", "Second one."]
    assert mapping[0][1] == mapping[1][0] and mapping[-1][1] == len(toks)


def test_leading_whitespace_only_span_merges_right(tok):
    text = "\n\nOnly text."
    offsets = _offsets(tok, text)
    toks = tok.convert_ids_to_tokens(tok(text, add_special_tokens=False)["input_ids"])
    first_word = next(i for i, t in enumerate(toks) if t.replace("Ċ", "").replace("Ġ", ""))
    breaks = [first_word - 1, len(toks) - 1]
    _, texts = edus_from_breaks(text, list(range(len(text))), offsets, breaks)
    assert texts == ["Only text."]


def test_normalized_edu_text_collapses_internal_line_breaks(tok):
    text = "TikTok Sale\n\n[News Briefing] China enacted\nnew rules.\n"
    normed, char_map = normalize_whitespace(text)
    offsets = _offsets(tok, normed)
    toks = tok.convert_ids_to_tokens(tok(normed, add_special_tokens=False)["input_ids"])
    assert not any(t.replace("Ġ", "") == "" or "Ċ" in t for t in toks)
    split = toks.index("ĠSale")
    _, texts = edus_from_breaks(text, char_map, offsets, [split, len(toks) - 1])
    assert texts == ["TikTok Sale", "[News Briefing] China enacted new rules."]


def _dmrst_dict(**over):
    d = dict(train_dir="<unused>", dev_dir="<unused>")
    d.update(over)
    return d


def test_pre_curriculum_checkpoint_config_loads():
    """The shape every larc-iu Hub checkpoint carries."""
    d = _dmrst_dict(max_epochs=100, checkpoint_every=None)
    before = dict(d)
    cfg = DMRSTConfig.from_dict(migrate_checkpoint_config(d))
    assert cfg.curriculum.epochs == 100
    assert d == before


def test_explicit_curriculum_wins_over_max_epochs():
    d = _dmrst_dict(max_epochs=100, curriculum={"type": "simple", "epochs": 7})
    assert DMRSTConfig.from_dict(migrate_checkpoint_config(d)).curriculum.epochs == 7


def test_current_checkpoint_config_is_untouched():
    d = _dmrst_dict(curriculum={"type": "simple", "epochs": 7}, validate_every=3)
    assert migrate_checkpoint_config(d) == d


def test_live_validate_every_survives_migration():
    """Removed by 0c0345f, since re-added as an epoch cadence: must not be dropped."""
    d = _dmrst_dict(max_epochs=100, checkpoint_every=None, validate_every=3)
    assert DMRSTConfig.from_dict(migrate_checkpoint_config(d)).validate_every == 3


def test_hand_written_configs_still_reject_retired_keys():
    """The migration is checkpoint-only; a stale training config should still fail."""
    with pytest.raises(Exception, match="max_epochs"):
        DMRSTConfig.from_dict(_dmrst_dict(max_epochs=100))


def test_generative_parser_rejects_empty_input_instead_of_returning_an_empty_edu():
    """Empty or whitespace-only text used to come back as a one-EDU tree whose EDU was ""."""
    from dipo.rst.parsers.gen.modeling_gen import GenParser

    parser = object.__new__(GenParser)  # the check runs before anything on the model is touched
    for text in ("", "   ", "\n\n \t\r\n"):
        with pytest.raises(ValueError, match="empty or contains only whitespace"):
            parser.predict_batch_from_texts([text])
    with pytest.raises(ValueError, match="empty or contains only whitespace"):
        parser.predict_batch_from_texts(["A real sentence.", ""])
    assert parser.predict_batch_from_texts([]) == []


def _units(text):
    return [text[i : i + 12] for i in line_start_breaks(text)]


def test_blank_lines_and_headings_start_units_but_hard_wrapped_prose_does_not():
    page = "Coffee\nCoffee is a brewed drink. It is popular.\nHistory\nThe earliest evidence is old."
    assert _units(page) == ["Coffee is a ", "The earliest"]
    wrapped = "It is a truth universally\r\nacknowledged, that a single man in\r\npossession of a good fortune.\r\n\r\nHowever little known."
    assert _units(wrapped) == ["However litt"]  # only the paragraph break, nothing inside the wrapped lines
    assert _units("A wrapped line that ends mid\nsentence and continues lowercase here.") == []
    assert _units("One long line with no newline at all.") == []
    assert _units("Title\n\nBody text starts here.\n") == ["Body text st"]
    assert _units("") == _units("\n\n  \n") == []


def test_cjk_headings_are_recognised_by_character_count():
    text = "唐朝\n唐朝是中国历史上的一個重要朝代，由唐高祖李淵所建立。\n歷史\n傳說在很久以前。"
    assert _units(text) == ["唐朝是中国历史上的一個重", "傳說在很久以前。"]
    assert _units("这是一行很长很长很长很长很长很长很长很长很长很长的文字\n另一行") == []  # long line before: not a heading


def test_forced_breaks_land_before_the_token_at_the_line_start():
    """A lone SentencePiece word marker owns the offset of the character after it, so the
    forced break has to go before the marker, and never inside a word."""
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
    from dipo.rst.parsers.common.encoding import encode_with_offsets

    text = "Terminologia\n1980ko hamarkada baino lehen, ez zegoen argi.\n"
    src, char_map = normalize_whitespace(text)
    ids, offsets = encode_with_offsets(tok, src)
    forced = line_start_breaks(text)
    assert forced == [text.index("1980ko")]
    breaks = with_forced_breaks(text, char_map, offsets, [len(ids) - 1], forced)
    mapping, edus = edus_from_breaks(text, char_map, offsets, breaks)
    assert edus[0] == "Terminologia" and edus[1].startswith("1980ko")
    assert len(edus) == 2
    # No forced positions: the breaks are unchanged.
    assert with_forced_breaks(text, char_map, offsets, [len(ids) - 1], []) == [len(ids) - 1]
