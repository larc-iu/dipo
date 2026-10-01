"""Optional 4-bit QLoRA base loading for the gen decoder_only backbone.

The default (`load_in_4bit=False`) must leave the model-construction path
byte-for-byte identical to the pre-existing bf16/fp32 loading: no
`quantization_config` kwarg to `from_pretrained`, and the `.to(model_dtype)`
uniformity cast still runs. When the flag is on, a `BitsAndBytesConfig` is built
with the standard NF4 knobs, passed to `from_pretrained`, the quantized base is
NEVER `.to()`-cast, and `prepare_model_for_kbit_training` runs before the LoRA
adapters are applied.

Nothing here downloads a model or imports bitsandbytes: `from_pretrained` and the
peft entry points are monkeypatched to capture their kwargs.
"""

from types import SimpleNamespace

import pytest
import torch

from iudex.rst.parsers.common.config import PeftConfig


# --- config: field defaults, validation, BitsAndBytesConfig construction ------


def test_defaults_are_off_and_make_bnb_config_returns_none():
    cfg = PeftConfig()
    assert cfg.load_in_4bit is False
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_compute_dtype == "bfloat16"
    assert cfg.make_bnb_config() is None


def test_make_bnb_config_builds_the_standard_nf4_config():
    pytest.importorskip("transformers")
    cfg = PeftConfig(load_in_4bit=True)
    bnb = cfg.make_bnb_config()
    assert bnb is not None
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.bnb_4bit_use_double_quant is True
    assert bnb.bnb_4bit_compute_dtype == torch.bfloat16


def test_compute_dtype_knob_is_honored():
    pytest.importorskip("transformers")
    bnb = PeftConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16").make_bnb_config()
    assert bnb.bnb_4bit_compute_dtype == torch.float16


def test_invalid_quant_type_is_rejected():
    with pytest.raises(ValueError, match="bnb_4bit_quant_type"):
        PeftConfig(load_in_4bit=True, bnb_4bit_quant_type="int4")


def test_invalid_compute_dtype_is_rejected():
    with pytest.raises(ValueError, match="bnb_4bit_compute_dtype"):
        PeftConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bf16")


def test_bnb_knobs_are_inert_when_off():
    """A nonsense quant_type is ignored while load_in_4bit is False (the knob only
    binds on the on-path), so the default construction never validates them."""
    cfg = PeftConfig(bnb_4bit_quant_type="int4")  # not validated, load_in_4bit False
    assert cfg.make_bnb_config() is None


# --- backbone _init_model: from_pretrained kwargs + the .to() cast -------------


class _FakeModel:
    """Stands in for the HF model: `_init_model` only reads `.model` (tower probe)
    and calls `.to(dtype)` (the uniformity cast we must skip for 4-bit)."""

    def __init__(self):
        self.model = None  # no vision/audio towers to drop
        self.to_calls: list = []

    def to(self, dtype):
        self.to_calls.append(dtype)
        return self


def _run_decoder_init(monkeypatch, peft):
    """Drive DecoderOnlyBackbone._init_model in isolation (no tokenizer, no
    download), capturing the from_pretrained kwargs. Returns (captured, model)."""
    from iudex.rst.parsers.gen.backbones import decoder_only as mod

    captured: dict = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return _FakeModel()

    monkeypatch.setattr(mod, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=fake_from_pretrained))
    # The native-quant probe reads the checkpoint config; a plain (non-quantized)
    # checkpoint reports no quantization_config, so the off/full-FT/bnb paths are unchanged.
    monkeypatch.setattr(
        mod, "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(quantization_config=None)),
    )

    bb = mod.DecoderOnlyBackbone.__new__(mod.DecoderOnlyBackbone)
    bb.config = SimpleNamespace(model_name="fake/model", amp=True, peft=peft)
    bb._init_model()
    return captured, bb.model


def test_off_path_passes_no_quantization_config_and_casts(monkeypatch):
    """LoRA, flag off: identical to the pre-existing bf16 path -- no
    quantization_config kwarg, and the .to(bfloat16) uniformity cast runs."""
    captured, model = _run_decoder_init(monkeypatch, PeftConfig(load_in_4bit=False))
    assert "quantization_config" not in captured["kwargs"]
    assert captured["kwargs"] == {"dtype": torch.bfloat16}
    assert model.to_calls == [torch.bfloat16]


def test_full_ft_path_is_unchanged(monkeypatch):
    """peft=None (full fine-tuning): fp32, no quantization_config, cast runs."""
    captured, model = _run_decoder_init(monkeypatch, None)
    assert captured["kwargs"] == {"dtype": torch.float32}
    assert model.to_calls == [torch.float32]


def test_4bit_path_passes_bnb_config_and_skips_the_cast(monkeypatch):
    pytest.importorskip("transformers")
    from transformers import BitsAndBytesConfig

    captured, model = _run_decoder_init(monkeypatch, PeftConfig(load_in_4bit=True))
    kwargs = captured["kwargs"]
    assert kwargs["dtype"] == torch.bfloat16
    assert isinstance(kwargs["quantization_config"], BitsAndBytesConfig)
    assert kwargs["quantization_config"].load_in_4bit is True
    # the quantized base is never .to()-cast (would corrupt packed 4-bit weights)
    assert model.to_calls == []


# --- seq2seq refuses 4-bit -----------------------------------------------------


def test_seq2seq_4bit_raises_not_implemented(monkeypatch):
    from iudex.rst.parsers.gen.backbones import seq2seq as mod

    bb = mod.Seq2SeqBackbone.__new__(mod.Seq2SeqBackbone)
    bb.config = SimpleNamespace(model_name="fake/model", amp=True, peft=PeftConfig(load_in_4bit=True))
    with pytest.raises(NotImplementedError, match="load_in_4bit"):
        bb._init_model()


# --- install_peft runs prepare_model_for_kbit_training only when 4-bit ---------


def _run_install_peft(monkeypatch, peft):
    pytest.importorskip("peft")
    import peft as peft_mod

    from iudex.rst.parsers.gen.backbones import decoder_only as mod

    calls: dict = {"prepared": False}

    def fake_prepare(model, **kwargs):
        calls["prepared"] = True
        calls["prepare_kwargs"] = kwargs
        return model

    def fake_get_peft_model(model, lora_cfg):
        calls["lora_cfg"] = lora_cfg
        return ("wrapped", model)

    monkeypatch.setattr(peft_mod, "prepare_model_for_kbit_training", fake_prepare)
    monkeypatch.setattr(peft_mod, "get_peft_model", fake_get_peft_model)

    bb = mod.DecoderOnlyBackbone.__new__(mod.DecoderOnlyBackbone)
    bb.model = _FakeModel()
    bb.install_peft(peft)
    return calls


def test_install_peft_prepares_for_kbit_on_4bit(monkeypatch):
    calls = _run_install_peft(monkeypatch, PeftConfig(load_in_4bit=True))
    assert calls["prepared"] is True
    assert calls["prepare_kwargs"]["use_gradient_checkpointing"] is True
    assert calls["prepare_kwargs"]["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert "lora_cfg" in calls  # LoRA still applied afterwards


def test_install_peft_skips_kbit_prep_when_off(monkeypatch):
    calls = _run_install_peft(monkeypatch, PeftConfig(load_in_4bit=False))
    assert calls["prepared"] is False
    assert "lora_cfg" in calls  # plain LoRA path unchanged


def test_native_quant_path_loads_as_shipped(monkeypatch):
    """A natively-quantized checkpoint (e.g. gpt-oss mxfp4) loads with NO dtype= (which
    would trigger a dequant) and NO bnb quantization_config; it is our-side unquantized."""
    from iudex.rst.parsers.gen.backbones import decoder_only as mod

    captured: dict = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeModel()

    monkeypatch.setattr(mod, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=fake_from_pretrained))
    monkeypatch.setattr(
        mod, "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(quantization_config={"quant_method": "mxfp4"})),
    )
    bb = mod.DecoderOnlyBackbone.__new__(mod.DecoderOnlyBackbone)
    bb.config = SimpleNamespace(model_name="fake/mxfp4", amp=True, peft=PeftConfig(load_in_4bit=False))
    bb._init_model()
    assert "dtype" not in captured["kwargs"], "native-quant load must not force a dtype (would dequantize)"
    assert "quantization_config" not in captured["kwargs"], "native-quant is not our bnb path"
