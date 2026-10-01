"""Raw-text input with line breaks (larc-iu/iudex#1) and pre-curriculum checkpoints.

Newlines in `predict_from_text` input are out of distribution (every training doc is
single-space joined), and DMRST's segmenter could isolate a run of `Ċ` subwords as an
EDU that then stripped to "". Input is now whitespace-collapsed before encoding, EDUs
are still sliced from the caller's text, and a whitespace-only span can never
surface as an EDU.

Separately, every larc-iu Hub checkpoint predates 0c0345f and carries `max_epochs` /
`checkpoint_every`, which tonga rejected, so none of them loaded.
"""

import pytest

from iudex.rst.parsers.common.inference import migrate_checkpoint_config
from iudex.rst.parsers.common.whitespace import collapse_whitespace, normalize_whitespace
from iudex.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig
from iudex.rst.parsers.dmrst.modeling_dmrst import edus_from_breaks

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
