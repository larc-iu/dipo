"""CPU-only tests for the encoder-backbone loading path in
`iudex.rst.parsers.common.encoding`.

Two contracts:

  (a) The BERT-style CLS/SEP striding path is UNCHANGED. `_window_sentinels`
      still selects [CLS] … [SEP], and `encode_tokens_strided` produces
      byte-for-byte the same output as the pre-change implementation (a copy of
      the old code lives in `_old_encode_tokens_strided_clssep` below as the
      regression oracle).

  (b) A CLS/SEP-less encoder (T5Gemma's encoder half + a Gemma SentencePiece
      tokenizer) loads via the new path: only the encoder stack is built (no
      decoder), BOS/EOS act as the per-window sentinels, the config exposes
      hidden_size, LoRA wrapping works, a couple of train steps run, and a fresh
      reconstruct + `load_state_dict(strict=True)` roundtrips.

Everything runs on CPU with tiny random-weight models. The real 2.6B T5Gemma is
never loaded here; a GPU smoke on `google/t5gemma-2b-2b-ul2` is still needed to
confirm the accessor/tokenizer against the actual released checkpoint.
"""

import tempfile
import types

import pytest
import torch

from iudex.rst.parsers.common import encoding
from iudex.rst.parsers.common.config import PeftConfig
from iudex.rst.parsers.common.encoding import (
    _resolve_base_dtype,
    _window_sentinels,
    encode_tokens_strided,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _StubTokenizer:
    """Minimal tokenizer exposing only what the striding/loader code reads."""

    def __init__(self, cls=None, sep=None, bos=None, eos=None, pad=0, model_max_length=32):
        self.cls_token_id = cls
        self.sep_token_id = sep
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.pad_token_id = pad
        self.model_max_length = model_max_length
        self.is_fast = True


class _EmbedEncoder(torch.nn.Module):
    """Deterministic stand-in encoder: last_hidden_state = embedding(input_ids).

    Deterministic per token id, so overlapping striding windows agree and the
    new vs. old comparison is meaningful. Exposes `.config.hidden_size`.
    """

    def __init__(self, vocab=64, hidden=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.config = types.SimpleNamespace(hidden_size=hidden, max_position_embeddings=512)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        return types.SimpleNamespace(last_hidden_state=self.embed(input_ids))


def _old_encode_tokens_strided_clssep(encoder, tokenizer, input_ids, max_length, stride):
    """Verbatim copy of the pre-change CLS/SEP striding code, kept as an oracle
    to prove the BERT-style path is unchanged."""
    max_content = max_length - 2
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    device = input_ids.device
    content_len = input_ids.shape[0]
    chunks, chunk_lens = [], []
    pos = 0
    while True:
        end = min(pos + max_content, content_len)
        chunk = torch.cat(
            [
                torch.tensor([cls_id], device=device),
                input_ids[pos:end],
                torch.tensor([sep_id], device=device),
            ]
        )
        chunks.append(chunk)
        chunk_lens.append(chunk.shape[0])
        if end >= content_len:
            break
        pos = end - stride
    max_chunk_len = max(chunk_lens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_ids = torch.full((len(chunks), max_chunk_len), pad_id, device=device, dtype=torch.long)
    batch_mask = torch.zeros(len(chunks), max_chunk_len, device=device, dtype=torch.long)
    for i, cids in enumerate(chunks):
        batch_ids[i, : cids.shape[0]] = cids
        batch_mask[i, : cids.shape[0]] = 1
    hidden = encoder(input_ids=batch_ids, attention_mask=batch_mask).last_hidden_state
    pieces = []
    for i, clen in enumerate(chunk_lens):
        emb = hidden[i, 1 : clen - 1]
        pieces.append(emb if i == 0 else emb[stride:])
    return torch.cat(pieces, dim=0)[:content_len]


# ---------------------------------------------------------------------------
# (a) CLS/SEP path is unchanged
# ---------------------------------------------------------------------------
def test_window_sentinels_selects_clssep_when_present():
    tok = _StubTokenizer(cls=101, sep=102, bos=1, eos=2)  # both pairs present
    assert _window_sentinels(tok) == ([101], [102])  # CLS/SEP wins


def test_strided_clssep_matches_old_implementation():
    torch.manual_seed(0)
    enc = _EmbedEncoder(vocab=64, hidden=8).eval()
    tok = _StubTokenizer(cls=50, sep=51, pad=0)
    # Several lengths (shorter/longer than the window) and strides.
    for content_len in (1, 5, 30, 71):
        for max_length, stride in ((16, 4), (10, 3), (32, 8)):
            ids = torch.randint(0, 48, (content_len,))
            with torch.no_grad():
                new = encode_tokens_strided(enc, tok, ids, max_length, stride)
                old = _old_encode_tokens_strided_clssep(enc, tok, ids, max_length, stride)
            assert new.shape == (content_len, 8), (content_len, max_length, stride)
            assert torch.equal(new, old), (content_len, max_length, stride)


# ---------------------------------------------------------------------------
# (b) CLS/SEP-less (BOS/EOS) path
# ---------------------------------------------------------------------------
def test_window_sentinels_falls_back_to_boseos():
    tok = _StubTokenizer(cls=None, sep=None, bos=2, eos=1)
    assert _window_sentinels(tok) == ([2], [1])


def test_window_sentinels_no_special_tokens():
    tok = _StubTokenizer(cls=None, sep=None, bos=None, eos=None)
    assert _window_sentinels(tok) == ([], [])


def test_strided_boseos_alignment_and_shape():
    """With BOS/EOS sentinels the output is still 1:1 with input positions, and
    a one-window doc simply wraps content in BOS … EOS then strips them."""
    torch.manual_seed(0)
    enc = _EmbedEncoder(vocab=64, hidden=8).eval()
    tok = _StubTokenizer(cls=None, sep=None, bos=2, eos=1, pad=0)
    ids = torch.randint(3, 48, (5,))
    with torch.no_grad():
        out = encode_tokens_strided(enc, tok, ids, max_length=16, stride=3)
        # Single window fits: emb == encoder([BOS, ids, EOS]) with sentinels stripped.
        ref = enc.embed(torch.cat([torch.tensor([2]), ids, torch.tensor([1])]))[1:-1]
    assert out.shape == (5, 8)
    assert torch.equal(out, ref)

    # Multi-window (forces striding) stays 1:1.
    long_ids = torch.randint(3, 48, (40,))
    with torch.no_grad():
        long_out = encode_tokens_strided(enc, tok, long_ids, max_length=10, stride=3)
    assert long_out.shape == (40, 8)


def test_gemma_tokenizer_selects_boseos_if_available():
    """Real Gemma SentencePiece tokenizer: no CLS/SEP, so BOS/EOS are picked.
    Skips cleanly when the tokenizer is not cached (offline CI)."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("google/t5gemma-2-1b-1b")
    except Exception:
        pytest.skip("Gemma tokenizer not cached")
    assert tok.cls_token_id is None and tok.sep_token_id is None
    prefix, suffix = _window_sentinels(tok)
    assert prefix == [tok.bos_token_id] and suffix == [tok.eos_token_id]


# ---------------------------------------------------------------------------
# base_dtype resolution
# ---------------------------------------------------------------------------
def test_resolve_base_dtype():
    assert _resolve_base_dtype(None) is torch.float32  # full-FT always fp32
    assert _resolve_base_dtype(PeftConfig()) is torch.float32  # LoRA default fp32
    assert _resolve_base_dtype(PeftConfig(base_dtype="bfloat16")) is torch.bfloat16


def test_peftconfig_rejects_bad_base_dtype():
    with pytest.raises(ValueError):
        PeftConfig(base_dtype="float16")


# ---------------------------------------------------------------------------
# T5Gemma encoder-half extraction + LoRA + strict roundtrip (tiny synthetic)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_t5gemma_dir():
    """A tiny random-weight T5Gemma encoder-decoder saved to disk. bos=2/eos=1/
    pad=0 fit the vocab so the stub Gemma tokenizer's sentinels are valid ids."""
    t5gemma = pytest.importorskip("transformers.models.t5gemma")
    from transformers import T5GemmaConfig, T5GemmaForConditionalGeneration

    mod = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    cfg = T5GemmaConfig(encoder=mod, decoder=mod)
    d = tempfile.mkdtemp()
    T5GemmaForConditionalGeneration(cfg).save_pretrained(d)
    return d


def _patch_tokenizer(monkeypatch):
    """Make the loader's AutoTokenizer return a Gemma-like stub (no CLS/SEP)."""
    stub = _StubTokenizer(cls=None, sep=None, bos=2, eos=1, pad=0, model_max_length=64)
    monkeypatch.setattr(encoding.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: stub))
    return stub


def test_load_encoder_half_no_decoder_and_config_exposed(tiny_t5gemma_dir, monkeypatch):
    _patch_tokenizer(monkeypatch)
    peft = PeftConfig(r=4, alpha=8, dropout=0.0, target_modules="all-linear")
    encoder, tok, max_length = encoding.load_encoder_and_tokenizer(tiny_t5gemma_dir, peft_config=peft)

    # Only the encoder half exists — the decoder was never allocated.
    assert not any("decoder" in n for n, _ in encoder.named_modules())
    # Top-level config exposes hidden_size for downstream (parser reads it).
    assert encoder.config.hidden_size == 16
    assert max_length == 64
    # LoRA adapters are present and trainable; base is frozen.
    assert any("lora" in k for k in encoder.state_dict())


def test_load_encoder_half_bf16_base(tiny_t5gemma_dir, monkeypatch):
    _patch_tokenizer(monkeypatch)
    peft = PeftConfig(r=4, alpha=8, dropout=0.0, target_modules="all-linear", base_dtype="bfloat16")
    encoder, _, _ = encoding.load_encoder_and_tokenizer(tiny_t5gemma_dir, peft_config=peft)
    # A frozen (non-LoRA) base weight is bf16; LoRA adapters stay fp32.
    base_w = encoder.base_model.model.encoder.layers[0].mlp.gate_proj.base_layer.weight
    assert base_w.dtype == torch.bfloat16
    lora_w = next(p for n, p in encoder.named_parameters() if "lora_A" in n)
    assert lora_w.dtype == torch.float32


def test_encoder_half_trains_and_roundtrips_strict(tiny_t5gemma_dir, monkeypatch):
    _patch_tokenizer(monkeypatch)
    peft = PeftConfig(r=4, alpha=8, dropout=0.0, target_modules="all-linear")

    encoder, tok, max_length = encoding.load_encoder_and_tokenizer(tiny_t5gemma_dir, peft_config=peft)
    encoder.train()

    # Per-token states come out (via the real BOS/EOS striding path) with the
    # right shape, and back-prop through the adapters works for a couple steps.
    opt = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=1e-3)
    ids = torch.randint(3, 48, (20,))
    for _ in range(2):
        states = encode_tokens_strided(encoder, tok, ids, max_length, stride=3)
        assert states.shape == (20, 16)
        loss = states.float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Fresh reconstruct + strict state-dict load (the checkpoint-load contract).
    fresh, _, _ = encoding.load_encoder_and_tokenizer(tiny_t5gemma_dir, peft_config=peft)
    result = fresh.load_state_dict(encoder.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
