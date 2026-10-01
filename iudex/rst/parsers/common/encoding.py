"""Shared token-encoding utilities for RST parsers which rely on BERT-like encoders."""

import copy
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from iudex.rst.parsers.common.detokenization import Detokenizer

# Tokenizers report `model_max_length = int(1e30)` when they have no advertised
# limit (e.g. SpanBERT). Treat anything above this sentinel as "unspecified".
_TOKENIZER_MAX_LEN_SENTINEL = 1_000_000

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}

# Encoder-decoder architectures whose bidirectional encoder half we support as a
# drop-in backbone. Maps HF `config.model_type` -> the encoder-only AutoClass
# name (imported lazily from `transformers`). The encoder is a Gemma-2-class
# bidirectional stack; loading only this class never allocates the decoder.
_ENCODER_ONLY_CLASS = {
    "t5gemma": "T5GemmaEncoderModel",
}


def _ensure_default_rope() -> None:
    """Restore the `"default"` entry in transformers' rotary-embedding registry.

    transformers 5.9 dropped `ROPE_INIT_FUNCTIONS["default"]` (and the
    `_compute_default_rope_parameters` helper behind it), leaving only the scaled
    variants. Checkpoints whose vendored modeling code predates that removal --
    EuroBERT is the one we use -- read the older flat `config.rope_scaling`,
    find nothing, fall back to `rope_type = "default"`, and die on a `KeyError`
    before a single weight is loaded.

    The replacement is the canonical unscaled RoPE, `1 / base^(2i/dim)`; it is
    verified bit-identical to transformers' own `"linear"` initialiser at
    `factor=1.0`, which is the same function by construction. Theta moved into
    the nested `rope_parameters` dict in 5.9, so read there first and fall back
    to the flat attribute for older configs.

    Registered with `setdefault`, so a future transformers that reinstates
    `"default"` keeps its own implementation and this becomes inert.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute(config: Any, device: Any = None, seq_len: int | None = None, **_: Any):
        params = getattr(config, "rope_parameters", None) or {}
        base = params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        exponent = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
        return 1.0 / (base**exponent), 1.0

    ROPE_INIT_FUNCTIONS.setdefault("default", _compute)


def load_encoder_and_tokenizer(model_name: str, peft_config: Any | None = None) -> tuple[torch.nn.Module, Any, int]:
    """Load a BERT-style HF encoder + tokenizer. Returns (encoder, tokenizer, max_length).

    Handles two encoder families:

      - Encoder-only checkpoints (BERT/RoBERTa/XLM-R/ModernBERT/Ettin): loaded
        via `AutoModel`. These have CLS/SEP, wrapped per striding window.
      - The encoder half of a supported encoder-decoder (T5Gemma): only the
        encoder stack is instantiated (the decoder is never allocated). Its
        SentencePiece tokenizer has no CLS/SEP, so BOS/EOS act as the per-window
        sentinels instead (see `_window_sentinels` / `encode_tokens_strided`).

    Base dtype: fp32 by default. transformers>=5 honors the checkpoint dtype, and
    fp16 checkpoints (e.g. SpanBERT) NaN immediately under AdamW, so the base is
    forced to fp32. Under LoRA the frozen base may instead load in bf16 (set
    `peft_config.base_dtype = "bfloat16"`), halving a large base's footprint; the
    downstream parser casts encoder outputs back to fp32 before its own layers.
    Full fine-tuning always stays fp32.

    When `peft_config` is non-null the encoder is wrapped in a LoRA `PeftModel`
    (base weights frozen, low-rank adapters trainable). The wrapper forwards
    attribute access (`.config`, `.embeddings`, `.encoder.layer`, `.forward`)
    and its `state_dict` keeps the full base weights plus adapters, so callers
    need no other changes. Because loads reconstruct the model via `Parser(cfg)`
    then `load_state_dict(strict=True)`, `peft_config` MUST live on the parser
    config so the identical wrapping is rebuilt at load time. `peft_config` is
    duck-typed: any object with `r`, `alpha`, `dropout`, `target_modules`,
    `bias`, `dora`, and optionally `base_dtype` (default "float32").
    """
    dtype = _resolve_base_dtype(peft_config)
    encoder = _load_base_encoder(model_name, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Validate that some consistent per-window sentinel scheme exists (CLS/SEP,
    # BOS/EOS, or none). `_window_sentinels` never raises for a released
    # tokenizer; this call just surfaces the selected scheme early.
    _window_sentinels(tokenizer)

    max_length = tokenizer.model_max_length
    if max_length > _TOKENIZER_MAX_LEN_SENTINEL:
        max_length = encoder.config.max_position_embeddings

    if peft_config is not None:
        encoder = _wrap_lora(encoder, peft_config)
        if getattr(peft_config, "gradient_checkpointing", False):
            _enable_gradient_checkpointing(encoder)
    return encoder, tokenizer, max_length


def _enable_gradient_checkpointing(encoder: torch.nn.Module) -> None:
    """Turn on activation checkpointing for a LoRA-wrapped base encoder.

    `use_reentrant=False` is required: the base is frozen, so with the reentrant
    autograd path the checkpointed segment would see no grad-requiring input and
    silently drop the LoRA-adapter gradients. `enable_input_require_grads` makes
    the (frozen) input-embedding output require grad, the pattern PEFT prescribes
    for training a frozen base under checkpointing. Numerically identical to no
    checkpointing; this only changes when activations are (re)materialized.
    """
    if not hasattr(encoder, "gradient_checkpointing_enable"):
        raise ValueError(
            f"{type(encoder).__name__} does not support gradient checkpointing "
            "(no gradient_checkpointing_enable); unset peft.gradient_checkpointing."
        )
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(encoder, "enable_input_require_grads"):
        encoder.enable_input_require_grads()


def _resolve_base_dtype(peft_config: Any | None) -> torch.dtype:
    """Base-encoder dtype: fp32 unless a LoRA config opts the frozen base into
    bf16. Full fine-tuning (`peft_config is None`) is always fp32."""
    if peft_config is None:
        return torch.float32
    name = getattr(peft_config, "base_dtype", "float32")
    if name not in _DTYPES:
        raise ValueError(f"peft base_dtype must be one of {sorted(_DTYPES)} (got {name!r})")
    return _DTYPES[name]


def _load_base_encoder(model_name: str, dtype: torch.dtype) -> torch.nn.Module:
    """Instantiate the base encoder. For encoder-decoder checkpoints, only the
    encoder half is built (the decoder is never allocated)."""
    _ensure_default_rope()
    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "is_encoder_decoder", False):
        return _load_encoder_half(model_name, config, dtype)
    # Encoder-only path. fp32 keeps the historical `AutoModel(...).float()`
    # behavior byte-for-byte; bf16 loads directly at the requested dtype.
    if dtype is torch.float32:
        return AutoModel.from_pretrained(model_name).float()
    return AutoModel.from_pretrained(model_name, dtype=dtype)


def _load_encoder_half(model_name: str, config: Any, dtype: torch.dtype) -> torch.nn.Module:
    """Load ONLY the encoder stack of a supported encoder-decoder checkpoint.

    Uses the model's dedicated encoder-only AutoClass (e.g. `T5GemmaEncoderModel`)
    with `is_encoder_decoder` flipped off in a copied config, so the decoder
    module is never constructed and its checkpoint weights are simply skipped.
    """
    model_type = getattr(config, "model_type", None)
    class_name = _ENCODER_ONLY_CLASS.get(model_type)
    if class_name is None:
        raise ValueError(
            f"{model_name!r} is an encoder-decoder ({model_type!r}) with no supported "
            f"encoder-only loader. Supported: {sorted(_ENCODER_ONLY_CLASS)}."
        )
    import transformers

    encoder_cls = getattr(transformers, class_name)
    enc_config = copy.deepcopy(config)
    enc_config.is_encoder_decoder = False
    if dtype is torch.float32:
        encoder = encoder_cls.from_pretrained(model_name, config=enc_config).float()
    else:
        encoder = encoder_cls.from_pretrained(model_name, config=enc_config, dtype=dtype)
    _expose_encoder_config(encoder)
    return encoder


def _expose_encoder_config(encoder: torch.nn.Module) -> None:
    """Lift the encoder sub-config's `hidden_size` / `max_position_embeddings`
    to the top-level config so downstream code (`encoder.config.hidden_size`,
    the max-length fallback) resolves them uniformly with encoder-only models.

    Encoder-decoder configs nest these under `config.encoder`; the model itself
    already reads that sub-config internally, so adding the top-level aliases is
    inert to the forward pass.
    """
    cfg = encoder.config
    sub = getattr(cfg, "encoder", None)
    if sub is None:
        return
    for attr in ("hidden_size", "max_position_embeddings"):
        if getattr(cfg, attr, None) is None and getattr(sub, attr, None) is not None:
            setattr(cfg, attr, getattr(sub, attr))


def _window_sentinels(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Per-window (prefix_ids, suffix_ids) sentinel token ids for striding.

    BERT-style encoders wrap each window in ``[CLS] … [SEP]``; that exact scheme
    is kept whenever both are present (Ettin/ModernBERT/XLM-R/BERT), so their
    behavior is unchanged. SentencePiece encoder stacks (Gemma/T5Gemma) have no
    CLS/SEP, so we fall back to ``BOS … EOS``, and finally to no sentinels if a
    tokenizer has neither pair. The same pair is applied when building each
    window and when slicing the sentinels back off, so boundary token indices
    stay consistent regardless of which scheme is selected.
    """
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls_id is not None and sep_id is not None:
        return [cls_id], [sep_id]
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    prefix = [bos_id] if bos_id is not None else []
    suffix = [eos_id] if eos_id is not None else []
    return prefix, suffix


def _wrap_lora(encoder: torch.nn.Module, peft_config: Any) -> torch.nn.Module:
    """Wrap `encoder` in a LoRA `PeftModel` (feature-extraction task)."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=peft_config.r,
        lora_alpha=peft_config.alpha,
        lora_dropout=peft_config.dropout,
        target_modules=peft_config.target_modules,
        bias=peft_config.bias,
        use_dora=peft_config.dora,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(encoder, lora_config)


def tokenize_edus(
    tokenizer: Any,
    edu_strings: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Tokenize a sequence of EDUs into a flat token-id tensor + per-EDU boundaries.

    Returns:
        input_ids: [num_tokens]
        boundaries: list of (start_token, end_token_exclusive) per EDU
    """
    all_ids: list[int] = []
    boundaries: list[tuple[int, int]] = []
    for edu_text in edu_strings:
        ids = tokenizer.encode(edu_text, add_special_tokens=False)
        start = len(all_ids)
        all_ids.extend(ids)
        boundaries.append((start, len(all_ids)))
    return torch.tensor(all_ids, dtype=torch.long, device=device), boundaries


def tokenize_document(
    tokenizer: Any,
    edu_strings: list[str],
    device: torch.device,
    detokenizer: "Detokenizer | None" = None,
    prefixes: "list[str | None] | None" = None,
    return_source: bool = False,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Tokenize EDUs as one continuous document, mapping gold EDU boundaries
    onto the continuous token offsets. Same return contract as `tokenize_edus`.

    Text reconstruction picks the most faithful source available:

      - `prefixes` given (detokenized corpora, e.g. data/gum_12.1.0_notok): EDU
        text is used verbatim and joined by its exact inter-EDU `prefix` string
        (`None` -> single space, "" -> glued). This reproduces the raw document
        byte-for-byte, so it supersedes `detokenizer` (which is ignored).
      - else `detokenizer` given: each word-tokenized EDU is detokenized to
        natural text and the EDUs are joined with single spaces.
      - else: EDUs are stripped and joined with single spaces.

    All three encode the whole string once. Unlike `tokenize_edus` (which
    encodes each EDU in isolation), this matters for joint segmenters: encoding
    an EDU in isolation strips the leading-space marker (e.g. RoBERTa/ModernBert
    `Ġ`) from its first subword, so every EDU-initial token looks word-initial.
    A segmenter trained that way learns "no leading-space marker = boundary", a
    cue absent from real continuous text, making it predict zero breaks at
    inference (where `predict_from_text` tokenizes the raw string continuously).
    Encoding continuously here keeps train and inference tokenization identical.
    SentencePiece encoders (e.g. XLM-R) mark word starts the same way in both
    modes, so they were unaffected, but this is correct for them too.

    Glued (`prefix=""`) boundaries have no joining space, so an EDU boundary can
    fall mid-token; the straddling token is then assigned to the later EDU. A
    boundary landing strictly inside a single token that also spans the previous
    boundary would empty out an EDU's token range, which we reject explicitly
    rather than letting it NaN downstream.

    Requires a fast tokenizer (offset mapping).
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("tokenize_document requires a fast tokenizer (offset mapping unavailable)")

    use_prefix = prefixes is not None and any(p is not None for p in prefixes)
    if use_prefix:
        edus = list(edu_strings)
        seps = ["" if i == 0 else (prefixes[i] if prefixes[i] is not None else " ") for i in range(len(edus))]
    else:
        edus = (
            [detokenizer.detokenize(e) for e in edu_strings]
            if detokenizer is not None
            else [e.strip() for e in edu_strings]
        )
        seps = ["" if i == 0 else " " for i in range(len(edus))]

    doc = "".join(seps[i] + edus[i] for i in range(len(edus)))
    enc = tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    # Exclusive char-end of each EDU within the reconstructed doc.
    char_ends: list[int] = []
    pos = 0
    for i, edu in enumerate(edus):
        pos += len(seps[i]) + len(edu)
        char_ends.append(pos)

    # A token belongs to EDU i iff its char-end falls within EDU i's char span.
    # A space-joined boundary always lands on a token break; a glued boundary
    # may not, in which case the straddling token (char-end past this EDU's end)
    # falls through to the next EDU.
    boundaries: list[tuple[int, int]] = []
    tok_idx, ntok = 0, len(offsets)
    for i, end_char in enumerate(char_ends):
        start = tok_idx
        while tok_idx < ntok and offsets[tok_idx][1] <= end_char:
            tok_idx += 1
        if tok_idx == start:
            raise ValueError(
                f"EDU {i} ({edus[i]!r}) maps to an empty token span: its boundary "
                f"falls inside a single token (a glued prefix with no token break)."
            )
        boundaries.append((start, tok_idx))
    # Defensive: a token overrunning the last EDU end (shouldn't happen) is
    # folded into the final EDU rather than dropped.
    if tok_idx < ntok and boundaries:
        s, _ = boundaries[-1]
        boundaries[-1] = (s, ntok)

    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device)
    if return_source:
        # `doc` + `offsets` let a caller recover a token span's text by SLICING
        # the source string rather than decoding ids back to text. Decoding is
        # lossy in ways that vary by tokenizer -- XLM-R's SentencePiece drops
        # Persian ZWNJ, so `بخش<ZWNJ>های` comes back as two words -- which made
        # predicted EDU text fail to reproduce the document it came from.
        return input_ids, boundaries, doc, offsets
    return input_ids, boundaries


def encode_tokens_strided(
    encoder: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_length: int,
    stride: int,
) -> torch.Tensor:
    """Encode a flat token sequence with overlapping sliding windows.

    Long documents exceed the LM's positional budget, so we tile with windows
    that overlap by `stride` tokens. Overlapped positions keep the embedding
    from the *earlier* window (more left context).

    Each window is wrapped in the tokenizer's sentinel pair (`[CLS] … [SEP]` for
    BERT-style encoders, `BOS … EOS` for CLS/SEP-less SentencePiece encoders like
    T5Gemma; see `_window_sentinels`). The prefix/suffix are prepended/appended
    at build time and sliced back off after encoding, so the choice of sentinels
    does not shift the returned 1:1 token alignment.

    Returns: [num_tokens, hidden_size]  (1:1 with input positions).
    """
    prefix_ids, suffix_ids = _window_sentinels(tokenizer)
    n_pre, n_suf = len(prefix_ids), len(suffix_ids)
    max_content = max_length - n_pre - n_suf  # leave room for the sentinels per chunk
    device = input_ids.device
    prefix = torch.tensor(prefix_ids, device=device, dtype=torch.long)
    suffix = torch.tensor(suffix_ids, device=device, dtype=torch.long)

    content_len = input_ids.shape[0]
    chunks, chunk_lens = [], []
    pos = 0
    while True:
        end = min(pos + max_content, content_len)
        chunk = torch.cat([prefix, input_ids[pos:end], suffix])
        chunks.append(chunk)
        chunk_lens.append(chunk.shape[0])
        if end >= content_len:
            break
        pos = end - stride  # next window starts `stride` tokens before this one ended

    max_chunk_len = max(chunk_lens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_ids = torch.full((len(chunks), max_chunk_len), pad_id, device=device, dtype=torch.long)
    batch_mask = torch.zeros(len(chunks), max_chunk_len, device=device, dtype=torch.long)
    for i, cids in enumerate(chunks):
        batch_ids[i, : cids.shape[0]] = cids
        batch_mask[i, : cids.shape[0]] = 1

    hidden = encoder(input_ids=batch_ids, attention_mask=batch_mask).last_hidden_state
    # hidden: [num_chunks, max_chunk_len, hidden_size]

    # Strip the sentinels. For chunks i > 0, also drop the first `stride` tokens
    # (which are duplicates of the previous chunk's tail).
    pieces = []
    for i, clen in enumerate(chunk_lens):
        emb = hidden[i, n_pre : clen - n_suf]
        pieces.append(emb if i == 0 else emb[stride:])
    return torch.cat(pieces, dim=0)[:content_len]
