"""The seq2seq encoder input budget accounts for each tokenizer's actual wrapper.

Construction probes the prefix/suffix specials `add_special_tokens=True` wraps
content with, and training checks the real wrapped length
(`pack_example`: `len(_encoder_input_ids(source_ids)) > max_input_length`). Inference
used to ignore both: it stripped a generic BOS/EOS/pad by token identity and checked
`len(ids) > cap - 1`, hardcoding "exactly one token gets re-added". Two consequences:

  * a tokenizer that wraps on BOTH sides passes that check with a document whose
    wrapped length exceeds the cap, so `decode_prefix` feeds an over-length encoder
    input -- the silent truncation 356bff2 exists to prevent, and a document training
    would have rejected outright;
  * identity-stripping eats a CONTENT token that happens to equal bos/eos/pad,
    shifting every COPY cursor against the training alignment. Not hypothetical:
    t5gemma-s-s-prefixlm sets bos_token_id=2 but prepends nothing.

Both live tokenizers wrap with exactly one token (T5 appends EOS, T5Gemma-2 prepends
BOS), for different reasons, which is why `cap - 1` was accidentally right and the
bug stayed latent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dipo.rst.parsers.gen.backbones.seq2seq import Seq2SeqBackbone, probe_enc_specials, strip_enc_wrapper
from dipo.rst.parsers.gen.errors import OverLengthError

BOS, EOS, PAD = 2, 1, 0

# (id, prefix, suffix): the layouts probe_enc_specials finds in the wild, plus a
# both-sides wrapper (the broken case) and a multi-token one.
LAYOUTS = [
    ("none", [], []),  # t5gemma-s-s-prefixlm
    ("suffix_eos", [], [EOS]),  # google-t5/t5-small
    ("prefix_bos", [BOS], []),  # google/t5gemma-2-1b-1b
    ("both", [BOS], [EOS]),  # the layout `cap - 1` gets wrong
    ("multi", [BOS, 7], [8, EOS]),
]
IDS = [layout[0] for layout in LAYOUTS]


class _FakeTokenizer:
    """Wraps content with a fixed prefix/suffix, like add_special_tokens=True."""

    def __init__(self, prefix: list[int], suffix: list[int], content: list[int] | None = None):
        self.prefix, self.suffix = prefix, suffix
        self.content = content if content is not None else [42]
        self.bos_token_id, self.eos_token_id, self.pad_token_id = BOS, EOS, PAD

    def __call__(self, text: str, add_special_tokens: bool = True, **kw):
        ids = list(self.content)
        if add_special_tokens:
            ids = [*self.prefix, *ids, *self.suffix]
        return {"input_ids": ids}


def _encoder_input_ids(content: list[int], prefix: list[int], suffix: list[int]) -> list[int]:
    """Mirror of Seq2SeqBackbone._encoder_input_ids."""
    return [*prefix, *content, *suffix]


@pytest.mark.parametrize("name,prefix,suffix", LAYOUTS, ids=IDS)
def test_probe_finds_the_wrapper(name, prefix, suffix):
    got = probe_enc_specials(_FakeTokenizer(prefix, suffix))
    assert got == (prefix, suffix)


@pytest.mark.parametrize("name,prefix,suffix", LAYOUTS, ids=IDS)
def test_strip_inverts_encoder_input_ids(name, prefix, suffix):
    content = [11, 12, 13]
    assert strip_enc_wrapper(_encoder_input_ids(content, prefix, suffix), prefix, suffix) == content


@pytest.mark.parametrize("name,prefix,suffix", LAYOUTS, ids=IDS)
def test_content_equal_to_a_special_survives(name, prefix, suffix):
    """Identity-stripping would eat these; position-stripping must not."""
    content = [BOS, 9, EOS, PAD]  # every token is also a special id
    assert strip_enc_wrapper(_encoder_input_ids(content, prefix, suffix), prefix, suffix) == content


def test_strip_raises_when_the_wrapper_is_absent():
    with pytest.raises(RuntimeError, match="prefix"):
        strip_enc_wrapper([11, 12], [BOS], [])
    with pytest.raises(RuntimeError, match="suffix"):
        strip_enc_wrapper([11, 12], [], [EOS])


# --- the budget itself ------------------------------------------------------


def _tokenize_source(prefix, suffix, content, cap):
    """Drive the REAL `Seq2SeqBackbone.tokenize_source` (and the real
    `_encoder_input_ids` it calls) against a duck-typed self, so no backbone or model
    has to load. Raises the real OverLengthError."""
    bb = SimpleNamespace(
        tokenizer=_FakeTokenizer(prefix, suffix, content),
        _enc_prefix_specials=list(prefix),
        _enc_suffix_specials=list(suffix),
        config=SimpleNamespace(max_input_length=cap),
    )
    bb._encoder_input_ids = lambda ids: Seq2SeqBackbone._encoder_input_ids(bb, ids)
    return Seq2SeqBackbone.tokenize_source(bb, "x")


def _tokenize_source_old(tok, cap):
    """The old check: strip generically, assume exactly one re-added token."""
    ids = list(tok("x")["input_ids"])
    while ids and ids[-1] == tok.pad_token_id:
        ids.pop()
    if ids and ids[-1] == tok.eos_token_id:
        ids.pop()
    if tok.bos_token_id is not None and ids and ids[0] == tok.bos_token_id:
        ids = ids[1:]
    if len(ids) > cap - 1:
        raise OverflowError(f"{len(ids)} > {cap - 1}")
    return ids


@pytest.mark.parametrize("name,prefix,suffix", LAYOUTS, ids=IDS)
def test_budget_boundary_is_the_wrapped_length(name, prefix, suffix):
    cap = 10
    n_specials = len(prefix) + len(suffix)
    fits = [50] * (cap - n_specials)  # wrapped length == cap exactly
    assert _tokenize_source(prefix, suffix, fits, cap) == fits

    over = fits + [50]  # one token too many
    with pytest.raises(OverLengthError):
        _tokenize_source(prefix, suffix, over, cap)


def test_old_check_admits_an_overflowing_document_when_wrapped_on_both_sides():
    """The bug: BOS+EOS, cap=10, 9 content tokens. The old check passes it (9 > 9 is
    false), then the encoder input is 1 + 9 + 1 = 11 > 10. Training rejects the same
    document, so the two paths disagree."""
    prefix, suffix, cap = [BOS], [EOS], 10
    content = [50] * 9
    tok = _FakeTokenizer(prefix, suffix, content)

    ids = _tokenize_source_old(tok, cap)  # does NOT raise
    assert len(_encoder_input_ids(ids, prefix, suffix)) == 11 > cap, "premise: this doc overflows"

    with pytest.raises(OverLengthError):
        _tokenize_source(prefix, suffix, content, cap)


def test_old_check_is_over_strict_when_nothing_is_wrapped():
    """The mirror-image error: with no specials the old check rejects a document that
    fits exactly (cap content tokens, wrapped length cap)."""
    prefix, suffix, cap = [], [], 10
    tok = _FakeTokenizer(prefix, suffix, [50] * cap)
    with pytest.raises(OverflowError):
        _tokenize_source_old(tok, cap)
    _tokenize_source(prefix, suffix, [50] * cap, cap)  # fits


@pytest.mark.parametrize(
    "model,prefix,suffix",
    [
        ("google-t5/t5-small", [], [1]),  # appends EOS
        ("google/t5gemma-2-1b-1b", [2], []),  # prepends BOS
    ],
)
def test_probe_against_the_real_tokenizers(model, prefix, suffix):
    """Pin what each live family actually wraps with, so a transformers upgrade that
    changes a wrapper surfaces here. Both wrap with exactly ONE token (on opposite
    sides), which is why the old `cap - 1` was accidentally correct for each."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(model)
    except Exception as e:  # noqa: BLE001 -- offline / gated
        pytest.skip(f"Could not load {model}: {e!r}")
    assert probe_enc_specials(tok) == (prefix, suffix)


@pytest.mark.parametrize("name,prefix,suffix", LAYOUTS, ids=IDS)
def test_training_and_inference_agree_at_the_boundary(name, prefix, suffix):
    """Whatever pack_example accepts, tokenize_source accepts, and vice versa."""
    cap = 10
    for n_content in range(1, cap + 3):
        content = [50] * n_content
        train_ok = len(_encoder_input_ids(content, prefix, suffix)) <= cap
        try:
            _tokenize_source(prefix, suffix, content, cap)
            infer_ok = True
        except OverLengthError:
            infer_ok = False
        assert train_ok == infer_ok, f"{name}: disagreement at {n_content} content tokens"
