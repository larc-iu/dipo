"""Causal decoder-only backbone: source subwords sit in front of the linearized
tree under one causal mask, separated by a learned `<|start_of_actions|>` marker.
"""

import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dipo.common.log import warn
from dipo.rst.parsers.common.seqgen import warm_init_head
from dipo.rst.parsers.gen.backbones.base import Backbone
from dipo.rst.parsers.gen.errors import OverLengthError

logger = logging.getLogger(__name__)


class DecoderOnlyBackbone(Backbone):
    SEP_TOKEN = "<|start_of_actions|>"

    def __init__(self, config):
        super().__init__(config)
        self.sep_token_id: int | None = None
        self._init_tokenizer()
        self._init_model()

    def _init_tokenizer(self) -> None:
        cfg = self.config
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        if not getattr(tok, "is_fast", False):
            raise RuntimeError(
                f"{cfg.model_name} loaded a slow tokenizer. align_edus_to_tokens needs "
                f"return_offsets_mapping, which only fast (Rust) tokenizers provide."
            )
        if not isinstance(tok.eos_token_id, int):
            raise RuntimeError(
                f"{cfg.model_name} has eos_token_id={tok.eos_token_id!r}. A single integer "
                f"EOS is required (stream terminator or opener, pad fallback)."
            )
        if tok.bos_token_id is None:
            logger.info("Tokenizer has no BOS (Qwen-style). EOS opens the stream instead, consistently at train and predict.")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        self.stream_end_id = int(tok.eos_token_id)
        if cfg.chat_template is not None:
            raise NotImplementedError("gen: chat_template lands in a later phase")

    def _init_model(self) -> None:
        cfg = self.config
        # bf16 master weights are fine under LoRA (adapters train on top); full-FT
        # AdamW on bf16 degenerates, so full-FT loads fp32 (autocast stays bf16).
        model_dtype = torch.bfloat16 if (cfg.amp and cfg.peft is not None) else torch.float32
        # 4-bit QLoRA (opt-in via peft.load_in_4bit): the frozen base loads in bnb NF4
        # and the LoRA adapters train on top. None when off, so the call below is the
        # exact pre-existing from_pretrained (no quantization_config kwarg).
        bnb_config = cfg.peft.make_bnb_config() if cfg.peft is not None else None
        # A checkpoint that ships its own quantization (e.g. gpt-oss's mxfp4 experts) is
        # loaded exactly as released: no dtype= (which would trigger a dequant-to-bf16 that
        # blows past a single GPU) and no .to() below (packed weights corrupt when cast).
        # Detected from the checkpoint config, independent of our own opt-in bnb 4-bit.
        native_quant = getattr(AutoConfig.from_pretrained(cfg.model_name), "quantization_config", None) is not None
        if native_quant and bnb_config is None:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        else:
            quant_kwargs = {"quantization_config": bnb_config} if bnb_config is not None else {}
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=model_dtype, **quant_kwargs)
        # Multimodal releases (Gemma 4) ship vision/audio towers this text-only
        # parser never feeds; drop them so the state_dict stays self-consistent.
        inner = getattr(model, "model", None)
        for attr in ("vision_tower", "audio_tower", "multi_modal_projector"):
            if inner is not None and getattr(inner, attr, None) is not None:
                delattr(inner, attr)
                setattr(inner, attr, None)
                logger.info(f"Dropped {attr} (text-only parser, tower never receives input).")
        # from_pretrained(dtype=...) can PARTIALLY cast a composite model (Gemma 4 nests a
        # text_config): a requested dtype that differs from the checkpoint's stored dtype
        # may leave some params at the stored dtype. Autocast hides the mix at train time
        # but fp32 eval crashes where a bf16 activation meets an fp32 weight. Force uniform
        # (.to leaves int buffers alone; no-op when already uniform, as under bf16 LoRA).
        # A bnb-quantized base is NEVER .to()-cast: the 4-bit params are packed uint8 and
        # casting them corrupts the weights, so skip the uniformity pass in that case. Same
        # for a natively-quantized checkpoint (mxfp4-packed experts).
        self.model = model if (bnb_config is not None or native_quant) else model.to(model_dtype)

    def stream_special_tokens(self) -> list[str]:
        return [self.SEP_TOKEN]

    PEFT_TASK_TYPE = "CAUSAL_LM"

    def _after_add_tokens(self) -> None:
        self.sep_token_id = int(self.tokenizer.convert_tokens_to_ids(self.SEP_TOKEN))

    def install_head(self, head_spec: dict) -> None:
        """Replace the output projection (unembedding) with a small fresh Linear over
        the serialization's head vocab. Done AFTER PEFT wrap so any LoRA adapter on it
        is discarded (this head is fully trainable). Rows warm-inited from the OLD
        projection rows (the unembedding basis, tied or not).

        The projection is located via get/set_output_embeddings rather than a hardcoded
        `lm_head` attribute, so backbones whose vocab projection is not named `lm_head`
        also work -- e.g. ModernBERT-decoder (Ettin), where `lm_head` is a
        dense+act+norm prediction-head transform and the vocab projection is a separate
        `decoder` Linear. The model's own forward applies the full head, so replacing
        the output-embedding module makes `out.logits` come out at head_vocab_size for
        every backbone. On the standard decoders (Gemma/Qwen/Llama) get_output_embeddings
        returns `lm_head`, so this is behavior-preserving there."""
        base = self.underlying_model()
        old = base.get_output_embeddings()
        if old is None:
            raise RuntimeError(
                f"Don't know how to replace the output projection on {type(base).__name__}: "
                f"get_output_embeddings() returned None."
            )
        weight = getattr(old, "weight", None)
        if weight is None and hasattr(old, "base_layer"):
            weight = old.base_layer.weight
        if weight is None:
            raise RuntimeError(
                f"Output projection on {type(base).__name__} has no `.weight` and no `.base_layer.weight`."
            )
        hidden = weight.shape[1]
        head_vocab_size = head_spec["head_vocab_size"]
        # Warm-init source: the old output projection (unembedding basis) or the input
        # embedding, per the serialization (both are [vocab, hidden]).
        source = head_spec.get("warm_init_source", "lm_head")
        warm_src = base.get_input_embeddings().weight if source == "input_embedding" else weight
        new = nn.Linear(hidden, head_vocab_size, bias=False).to(dtype=weight.dtype, device=weight.device)
        warm_init_head(new, warm_src, head_spec["full_id_for_head_idx"])
        base.set_output_embeddings(new)
        logger.info(f"Replaced output projection with fresh Linear(hidden={hidden}, head_vocab_size={head_vocab_size}).")

    # underlying_model / enable_grad_checkpointing / set_grad_checkpointing /
    # cache_configs / inference_mode are shared on the Backbone base.

    # ---- packaging + forward ----

    def _build_prefix_ids(self, source_ids: list[int]) -> list[int]:
        """The causal prefix the output stream is generated after: BOS, the
        verbatim source run (COPY feeds these back), then the SEP marker. Source
        ids must appear as one contiguous run so the COPY cursor and tree
        reconstruction can index them."""
        tok = self.tokenizer
        bos_id = int(tok.bos_token_id) if tok.bos_token_id is not None else int(tok.eos_token_id)
        return [bos_id, *source_ids, self.sep_token_id]

    def _max_positions(self) -> int | None:
        """The model's absolute positional limit, or None if the config does not
        expose one."""
        model_cfg = self.model.config
        get_text = getattr(model_cfg, "get_text_config", None)
        if callable(get_text):
            model_cfg = get_text() or model_cfg
        mp = getattr(model_cfg, "max_position_embeddings", None)
        return mp if isinstance(mp, int) and mp > 0 else None

    def pack_example(self, source_ids, seen_tokens, label_tokens, *, scores_separator: bool = False):
        """Assemble a single causal stream from a linearized target. Appends the
        stream terminator to both sides, prepends the masked [bos, source, sep]
        prefix, and drops the example (returns None) if any side or the combined
        stream overflows. `scores_separator` scores the SEP position (sexp) instead
        of masking it (SR)."""
        cfg = self.config
        if len(source_ids) > cfg.max_input_length:
            warn(
                f"Source side overflowed: {len(source_ids)} > max_input_length={cfg.max_input_length}. "
                f"Tree cannot be encoded (training raises on this)."
            )
            return None
        seen = [*seen_tokens, self.stream_end_id]
        label = [*label_tokens, self.stream_end_id]
        if len(label) > cfg.max_output_length:
            warn(
                f"Target side overflowed: {len(label)} > max_output_length={cfg.max_output_length}. "
                f"Tree cannot be encoded (training raises on this)."
            )
            return None

        prefix_ids = self._build_prefix_ids(source_ids)  # [bos, source, sep]
        input_ids = [*prefix_ids, *seen]
        if scores_separator:
            # Score the SEP position (last prefix slot); mask [bos, source] only.
            labels = [-100] * (len(prefix_ids) - 1) + [self.sep_token_id] + label
        else:
            labels = [-100] * len(prefix_ids) + label
        assert len(input_ids) == len(labels), (len(input_ids), len(labels))

        # The realized single stream is prefix + actions, so per-side caps don't
        # bound it. Drop trees whose combined length overflows the model's
        # positional limit (if cheaply known) or the summed per-side budget.
        sum_cap = cfg.max_input_length + cfg.max_output_length + (len(prefix_ids) - len(source_ids))
        max_positions = self._max_positions()
        combined_cap = min(sum_cap, max_positions) if max_positions is not None else sum_cap
        if len(input_ids) > combined_cap:
            warn(
                f"Combined stream overflowed: {len(input_ids)} > combined cap {combined_cap}. "
                f"Tree cannot be encoded (training raises on this)."
            )
            return None
        return input_ids, labels

    def collate(self, examples, pad_id):
        # examples: list of (input_ids, labels) single-stream tuples.
        max_len = max(len(ids) for ids, _ in examples)
        B = len(examples)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        for i, (ids, lab) in enumerate(examples):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def forward_logits(self, batch):
        """Causal-LM training pass. Returns (shifted_logits, shifted_labels): the causal
        shift so prediction at position i targets token i+1.

        For a single-example batch (gen trains one document per step) the scored labels are
        a contiguous suffix (the whole source prefix is -100), so `logits_to_keep` runs the
        lm_head only from the first scored position onward. In words mode at 27-31B this
        roughly halves the 262k-vocab logit + gradient memory (the top OOM), and it is a
        no-op numerically (the skipped positions were all -100)."""
        labels = batch["labels"]
        keep = self._suffix_logits_to_keep(labels)
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            use_cache=False,
            **({"logits_to_keep": keep} if keep is not None else {}),
        )
        logits = out.logits  # [B, (keep or L), head_vocab_size]
        if keep is None:
            return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        # logits cover the last `keep` positions; the first of them is the predecessor of
        # the first scored label, so logits[:-1] predict labels[first_scored:].
        first_scored = labels.shape[1] - keep + 1
        return logits[..., :-1, :].contiguous(), labels[..., first_scored:].contiguous()

    @staticmethod
    def _suffix_logits_to_keep(labels):
        """`logits_to_keep` = number of positions from the first scored label's predecessor
        to the sequence end, or None to keep all logits. Only fires for a single example
        whose scored labels form a contiguous suffix ending at the last position (the gen
        batch=1 case); anything else (batched, or a gap in the scored region) returns None
        so the caller falls back to the full-logits path."""
        if labels.shape[0] != 1:
            return None
        scored = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if scored.numel() == 0:
            return None
        first, last = int(scored[0].item()), int(scored[-1].item())
        if first == 0 or last != labels.shape[1] - 1 or scored.numel() != last - first + 1:
            return None
        return labels.shape[1] - first + 1

    # ---- decode I/O ----

    def tokenize_source(self, text: str) -> list[int]:
        """Decode-time source tokenization: the subword stream the COPY positions
        reference. No specials. Raises OverLengthError if the doc doesn't fit;
        a truncated source silently corrupts the metric (see errors.py).
        (Distinct from the training path's EDU-aligned, non-truncating tokenize.)"""
        cap = self.config.max_input_length
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > cap:
            raise OverLengthError(
                f"Source overflowed at inference: {len(ids)} > max_input_length={cap} subwords. "
                f"Bump max_input_length to fit the longest document, then re-run."
            )
        return ids

    def decode_prefix(self, source_ids: list[int]) -> list[int]:
        return self._build_prefix_ids(source_ids)

    def max_decode_steps(self, prefix_len: int) -> int | None:
        """The single causal stream spends `prefix_len` positions before the first
        generated token, so generation may only run to the model's positional limit.
        The inference mirror of pack_example's combined-stream cap."""
        max_positions = self._max_positions()
        if max_positions is None:
            return None
        return max(0, max_positions - prefix_len)

    def _fresh_cache(self):
        """Window-aware KV cache for the prefix forward, or None to let the model
        build its default. Caps sliding-window layers at their window (Gemma-4 OOM
        fix); returns None for models that build their own correct cache."""
        cfg = self.model.config
        get_text = getattr(cfg, "get_text_config", None)
        tc = get_text() if callable(get_text) else cfg
        layer_types = getattr(tc, "layer_types", None) or []
        if "sliding_attention" in layer_types and "linear_attention" not in layer_types:
            from transformers import DynamicCache

            cache = DynamicCache(config=tc)
            if not getattr(cache, "layers", None):
                # transformers 5.9.0 bug: shared-KV slice drops every layer at
                # num_kv_shared_layers==0. Build per-type layers ourselves.
                from transformers.cache_utils import Cache, DynamicLayer
                from transformers.cache_utils import LAYER_TYPE_CACHE_MAPPING as layer_map

                layers = [layer_map.get(t, DynamicLayer)(tc) for t in layer_types]
                cache = DynamicCache.__new__(DynamicCache)
                Cache.__init__(cache, layers=layers)
            return cache
        return None

    def seed(self, prefix_ids, num_rows: int = 1):
        """Prefix forward. Returns (logits[-1], past_key_values). num_rows > 1
        replicates the prefix across beams (beam support lands with the beam core)."""
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prefix_ids] * num_rows, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self._fresh_cache(),
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,  # only the last position seeds generation; the KV cache still covers the full prefix
        )
        logits = out.logits[:, -1, :] if num_rows > 1 else out.logits[0, -1, :]
        return logits, out.past_key_values

    def advance(self, cache, next_input):
        """One decode step. `next_input` an int (greedy) -> logits [head_V]; a
        sequence of K ids (beam) -> logits [K, head_V]."""
        device = next(self.model.parameters()).device
        if isinstance(next_input, int):
            step_input = torch.tensor([[next_input]], dtype=torch.long, device=device)
            out = self.model(input_ids=step_input, past_key_values=cache, use_cache=True, return_dict=True)
            return out.logits[0, -1, :], out.past_key_values
        step_input = torch.tensor(list(next_input), dtype=torch.long, device=device).unsqueeze(1)
        out = self.model(input_ids=step_input, past_key_values=cache, use_cache=True, return_dict=True)
        return out.logits[:, -1, :], out.past_key_values

    # ---- batched decode I/O (B different documents) ----

    def seed_rows(self, prefix_ids_rows):
        """Prefill B different documents in one forward, LEFT-padded.

        Left padding (HF's own convention for batched generation) keeps every row's
        real tokens flush against the right edge, so the seed logits are `[:, -1, :]`
        for every row and each generated token lands at the same cache index across
        rows. `position_ids` come from the mask's cumsum rather than a bare arange,
        so a row's real tokens occupy positions 0..L_i-1 exactly as they do when that
        document decodes alone -- padding shifts nothing. The pad columns are
        attention-masked out, so no real query ever attends to them.
        """
        device = self.device
        pad_id = int(self.tokenizer.pad_token_id)
        B, L = len(prefix_ids_rows), max(len(p) for p in prefix_ids_rows)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        for i, ids in enumerate(prefix_ids_rows):
            input_ids[i, L - len(ids) :] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, L - len(ids) :] = 1
        # Real tokens -> 0..L_i-1; the (masked-out) left pads collapse onto 0.
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self._fresh_cache(),
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,  # only the last position seeds generation; the KV cache still covers the full prefix
        )
        # next_pos: the position id the row's FIRST generated token takes (= its real
        # prefix length), tracked per row because the rows' prefixes differ in length.
        cache = {
            "past": out.past_key_values,
            "attention_mask": attention_mask,
            "next_pos": attention_mask.sum(-1),
        }
        return out.logits[:, -1, :], cache

    def advance_rows(self, cache, next_inputs):
        """One decode step for every row of a `seed_rows` batch. Extends the
        attention mask by one attended column (the token just emitted) and advances
        each row's own position counter."""
        device = self.device
        step_input = torch.tensor(list(next_inputs), dtype=torch.long, device=device).unsqueeze(1)
        attention_mask = torch.cat(
            [cache["attention_mask"], torch.ones((step_input.shape[0], 1), dtype=torch.long, device=device)], dim=1
        )
        out = self.model(
            input_ids=step_input,
            attention_mask=attention_mask,
            position_ids=cache["next_pos"].unsqueeze(1),
            past_key_values=cache["past"],
            use_cache=True,
            return_dict=True,
        )
        cache = {
            "past": out.past_key_values,
            "attention_mask": attention_mask,
            "next_pos": cache["next_pos"] + 1,
        }
        return out.logits[:, -1, :], cache

    def beam_reorder_needed(self, step, parents, k, cache):
        from dipo.rst.parsers.common.seqgen import beam_reorder_needed

        # The cache IS the past_key_values for a single causal stream.
        return beam_reorder_needed(step, parents, k, cache)

    def reorder_cache(self, cache, parent_tensor):
        from dipo.rst.parsers.common.seqgen import reorder_past_key_values

        return reorder_past_key_values(cache, parent_tensor, self.underlying_model())
