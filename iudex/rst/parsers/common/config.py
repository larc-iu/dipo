"""Shared config-parsing helper and the shared PEFT/LoRA sub-configs."""

from dataclasses import dataclass, field
from typing import TypeVar

from tonga import FromParams

T = TypeVar("T", bound=FromParams)


def parse_config_dict(cls: type[T], d: dict) -> T:
    """Instantiate `cls` from a plain dict (e.g. `tonga.Params.as_dict()` output)."""
    return cls.from_params(d)


# The only accepted values of the two inert fields (see PeftConfig).
_INERT_MODULES_TO_SAVE = ["embed_tokens"]


@dataclass
class PeftConfig(FromParams):
    """LoRA fine-tuning hyperparameters, shared by every parser (`peft: null`
    means full fine-tuning).

    `r`, `alpha`, `dropout`, `target_modules`, `bias`, `dora` are the core LoRA
    knobs, read by every parser.

    `modules_to_save` / `train_only_new_embedding_rows` are INERT: no parser reads
    them, and no `LoraConfig` is built from them. The four hand-written generative
    parsers did (`modules_to_save=['embed_tokens']` plus a retie, so the full lm_head
    could learn to emit source subwords under `use_copy=false`); `gen` replaced that
    with the new-row shadow, which trains the new rows WITHOUT duplicating the
    vocab x hidden matrix, and rejects `use_copy=false` under LoRA outright.

    They stay declared, rather than being deleted, for two reasons: every archived
    run's `config.json` carries them and tonga rejects unknown keys, so removing them
    would make those configs unloadable (e.g. for `iudex gen eval`); and `peft` is
    hashed as a nested blob (HASH_EXCLUDE only filters top-level keys), so removing
    them would re-id every LoRA run. Any value but the default is rejected, because
    silently ignoring a knob a user set is worse than either.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    dora: bool = False
    modules_to_save: list[str] = field(default_factory=lambda: list(_INERT_MODULES_TO_SAVE))
    train_only_new_embedding_rows: bool = True
    # dtype of the FROZEN base encoder weights, read only by the discriminative
    # encoder parsers (`load_encoder_and_tokenizer`). Full fine-tuning always
    # uses fp32 (this field is ignored when `peft` is null). Under LoRA the base
    # is frozen, so bf16 halves its footprint with no optimizer-stability risk
    # (a 2.6B fp32 base is ~10GB); the low-rank adapters and the downstream
    # parser layers stay fp32. "float16" is rejected downstream (NaNs under
    # AdamW, same reason the base loader forces fp32 for fp16 checkpoints).
    base_dtype: str = "float32"

    # Activation checkpointing on the frozen base encoder (read only by the
    # discriminative encoder parsers, `load_encoder_and_tokenizer`). Recomputes
    # each transformer block's activations during backward instead of storing
    # them, trading ~25-35% train-step compute for a large activation-memory cut.
    # NUMERICALLY IDENTICAL (no effect on results) — a memory knob only. Needed to
    # fit the fp32 10.7B XLM-R-XXL base + per-document sliding-window activations
    # in 96GB. Default OFF leaves every existing run byte-for-byte unchanged.
    gradient_checkpointing: bool = False

    # --- 4-bit QLoRA (optional; default OFF keeps every existing run unchanged) ---
    # When True the backbone loads the frozen base in bitsandbytes NF4 and the LoRA
    # adapters train on top (QLoRA). Lets a 70B-class decoder-only model fit on a
    # single 96GB GPU. Only wired for backbone="decoder_only" (seq2seq raises). The
    # three knobs below are the standard NF4 defaults and are inert when this is
    # False. bitsandbytes must be installed on the training machine.
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    _BNB_QUANT_TYPES = ("nf4", "fp4")
    _BNB_COMPUTE_DTYPES = ("bfloat16", "float16", "float32")

    def __post_init__(self):
        if self.r < 1:
            raise ValueError(f"PeftConfig.r must be >= 1 (got {self.r})")
        if self.load_in_4bit:
            if self.bnb_4bit_quant_type not in self._BNB_QUANT_TYPES:
                raise ValueError(
                    f"PeftConfig.bnb_4bit_quant_type must be one of {self._BNB_QUANT_TYPES} "
                    f"(got {self.bnb_4bit_quant_type!r})"
                )
            if self.bnb_4bit_compute_dtype not in self._BNB_COMPUTE_DTYPES:
                raise ValueError(
                    f"PeftConfig.bnb_4bit_compute_dtype must be one of {self._BNB_COMPUTE_DTYPES} "
                    f"(got {self.bnb_4bit_compute_dtype!r})"
                )
        if self.modules_to_save != _INERT_MODULES_TO_SAVE:
            raise ValueError(
                f"PeftConfig.modules_to_save is inert and must stay {_INERT_MODULES_TO_SAVE} "
                f"(got {self.modules_to_save}). No parser reads it: newly-added token rows train via "
                f"gen's new-row shadow, and putting the embedding in LoRA's modules_to_save would "
                f"duplicate the whole vocab x hidden matrix to train a handful of rows. To train more "
                f"of the model, widen peft.target_modules or use full fine-tuning (peft: null)."
            )
        if self.train_only_new_embedding_rows is not True:
            raise ValueError(
                f"PeftConfig.train_only_new_embedding_rows is inert and must stay True "
                f"(got {self.train_only_new_embedding_rows}). Under LoRA the pretrained embedding rows "
                f"are frozen and only the new rows train (gen's new-row shadow); setting this False "
                f"would not change that. Use full fine-tuning (peft: null) to train the base rows."
            )
        if self.base_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                f"PeftConfig.base_dtype must be 'float32' or 'bfloat16' (got {self.base_dtype!r})"
            )

    def make_bnb_config(self):
        """The `transformers.BitsAndBytesConfig` for 4-bit QLoRA, or None when
        `load_in_4bit` is False (the default, so no backbone imports bitsandbytes
        unless this arm is explicitly turned on). `bitsandbytes` itself is not
        imported here -- `BitsAndBytesConfig` is a plain transformers dataclass; the
        actual bnb import happens inside `from_pretrained` when the model is
        quantized, which requires bitsandbytes + a CUDA GPU on the training host."""
        if not self.load_in_4bit:
            return None
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, self.bnb_4bit_compute_dtype),
        )

