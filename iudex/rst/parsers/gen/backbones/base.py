"""Backbone strategy interface for the unified generative parser `gen`.

A backbone owns the pretrained model + tokenizer and everything that changes with
the model family: encoder-decoder `seq2seq` (two streams + cross-attention) vs
causal `decoder_only` (one source+output stream). It is an `nn.Module` and owns
`self.model`, so `GenParser`'s state_dict nests the weights under `backbone.model.*`.

The two axes never reference each other: a backbone knows nothing about SR vs
sexp. Everything serialization-specific (action vocabulary, scoring-head layout,
decode state machine, reconstruction) lives in the `Serialization` strategy. The
backbone consumes only a `head_spec` handshake from it (what small head to install,
if any) and receives already-linearized tokens to pack into a training example.

Subclasses set `self.model`, `self.tokenizer`, `self.stream_end_id`, and
`self._original_vocab_size` during construction and implement the packaging +
decode-I/O methods below; the grad-checkpointing / cache / PEFT-walk / new-row
infrastructure is shared here.
"""

import contextlib

import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = None
        # The token that terminates the output stream (raw EOS, or a chat
        # template's turn-end). The last slot of a small head in token mode.
        self.stream_end_id: int | None = None
        # Pre-add vocab boundary, set by add_tokens_and_resize; new rows train from here.
        self._original_vocab_size: int | None = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    # ---- construction ----

    def stream_special_tokens(self) -> list[str]:
        """Special tokens this backbone needs beyond the serialization's action
        vocab (e.g. the single-stream source/output separator). Added together
        with `Serialization.action_tokens()` in one resize."""
        raise NotImplementedError

    def add_tokens_and_resize(self, new_tokens: list[str]) -> None:
        """Add the given special tokens to `self.tokenizer`, resize the model
        embedding, then let the subclass resolve its post-add ids
        (`_after_add_tokens`). New ids must land at/after the pre-add vocab
        boundary so the new-row training scheme reaches them."""
        tok = self.tokenizer
        self._original_vocab_size = len(tok)
        existing = set(tok.get_vocab().keys())
        to_add = [t for t in new_tokens if t not in existing]
        if to_add:
            tok.add_special_tokens({"additional_special_tokens": to_add})
            # mean_resizing=False: mean+covariance init builds fp32 copies of the
            # whole embedding (multi-GB RAM spike at 27-31B vocabs). Plain normal
            # init is equivalent here (only new rows train; head warm-inited apart).
            self.model.resize_token_embeddings(len(tok), mean_resizing=False)
            bad = [
                (t, int(tok.convert_tokens_to_ids(t)))
                for t in to_add
                if int(tok.convert_tokens_to_ids(t)) < self._original_vocab_size
            ]
            if bad:
                raise RuntimeError(
                    f"New tokens got ids below the pre-add vocab boundary "
                    f"{self._original_vocab_size}: {bad}. The new-row training would freeze them."
                )
        self._after_add_tokens()

    def _after_add_tokens(self) -> None:
        """Resolve the per-family ids that depend on the post-add vocab (the
        single-stream separator, the decoder-start id)."""
        raise NotImplementedError

    # LoRA task type for install_peft, set by the subclass ("CAUSAL_LM" | "SEQ_2_SEQ_LM").
    PEFT_TASK_TYPE: str

    def install_peft(self, peft_cfg) -> None:
        """Wrap `self.model` in LoRA adapters. The input embedding is NOT in
        `modules_to_save` (that would duplicate the vocab x hidden matrix to train
        ~100 new rows); the new rows train via the new-row shadow (see
        GenParser.configure_new_row_training).

        Under 4-bit QLoRA (`peft_cfg.load_in_4bit`) the frozen bnb-quantized base is
        first run through `prepare_model_for_kbit_training` (upcasts layernorms to
        fp32, enables gradient checkpointing with use_reentrant=False, and hooks the
        input embedding so gradients flow through the frozen base to the adapters).
        This branch is skipped entirely when 4-bit is off, so the standard path is
        byte-for-byte unchanged."""
        from peft import LoraConfig, TaskType, get_peft_model

        if getattr(peft_cfg, "load_in_4bit", False):
            from peft import prepare_model_for_kbit_training

            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        lora_cfg = LoraConfig(
            task_type=getattr(TaskType, self.PEFT_TASK_TYPE),
            r=peft_cfg.r,
            lora_alpha=peft_cfg.alpha,
            lora_dropout=peft_cfg.dropout,
            target_modules=peft_cfg.target_modules,
            bias=peft_cfg.bias,
            use_dora=peft_cfg.dora,
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def install_head(self, head_spec: dict) -> None:
        """Replace the model's lm_head with a small fresh head over the
        serialization's action vocab (token mode). `head_spec` carries
        `head_vocab_size` and `full_id_for_head_idx` (the warm-init source rows).
        No-op backbones-side for full-head (words / use_copy=False) serializations,
        which never call this."""
        raise NotImplementedError

    def underlying_model(self):
        """Walk PEFT wrappers to the model that owns the embeddings + lm_head. Gates
        on PEFT module-origin (not attribute presence): HF's `base_model` shortcut
        also returns the inner transformer, so an attribute-only walk would descend
        past the LM head on a no-PEFT setup."""
        m = self.model
        if type(m).__module__.startswith("peft"):
            m = m.base_model
            if hasattr(m, "model") and not isinstance(m, nn.ModuleList):
                m = m.model
        return m

    def enable_grad_checkpointing(self) -> None:
        self.set_grad_checkpointing(True)
        for c in self.cache_configs():
            c.use_cache = False

    def set_grad_checkpointing(self, enabled: bool) -> None:
        method = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        fn = getattr(self.model, method, None)
        if callable(fn):
            if enabled:
                fn(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                fn()
            return
        for mod in self.model.modules():
            if hasattr(mod, "gradient_checkpointing"):
                mod.gradient_checkpointing = enabled

    def cache_configs(self) -> list:
        """Every config object whose `use_cache` gates KV caching (the model's, its
        PEFT base's, and any nested text config), deduped by identity."""
        cfgs: list = []
        for holder in (self.model, getattr(self.model, "base_model", None)):
            c = getattr(holder, "config", None) if holder is not None else None
            if c is None or any(c is seen for seen in cfgs):
                continue
            cfgs.append(c)
            get_text = getattr(c, "get_text_config", None)
            tc = get_text() if callable(get_text) else None
            if tc is not None and not any(tc is seen for seen in cfgs):
                cfgs.append(tc)
        return cfgs

    def _decode_autocast(self):
        """bf16 autocast for decode, or a null context.

        Only the training step autocasts, so a run whose weights are fp32 also
        DECODES in fp32 -- roughly 2x slower than bf16 on a bandwidth-bound decode.
        `_init_model` loads bf16 only under `amp and peft is not None`, so the fp32
        case is full fine-tuning (bf16 master weights degenerate under AdamW), where
        `amp` still means "compute in bf16". Autocast there and decode matches the
        precision the run trained at.

        Deliberately keyed off the live parameter dtype, not `cfg.peft`: a model that
        is ALREADY bf16 (the LoRA path) must not be wrapped, since autocast would
        re-cast ops that currently run in the checkpoint's own dtype and could perturb
        a decode whose numbers are already published.
        """
        if not self.config.amp:
            return contextlib.nullcontext()
        device = self.device
        if device.type not in ("cuda", "cpu"):
            return contextlib.nullcontext()
        if next(self.model.parameters()).dtype != torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    @contextlib.contextmanager
    def inference_mode(self):
        """Force use_cache on every cache-gating config + gradient checkpointing off
        during decode (some submodules consult the config flag, not just the forward
        kwarg), and run the decode under bf16 autocast when the weights are fp32 (see
        `_decode_autocast`). Restored on exit."""
        gc_was_on = self.config.gradient_checkpointing
        prev = [(c, getattr(c, "use_cache", None)) for c in self.cache_configs()]
        if gc_was_on:
            self.set_grad_checkpointing(False)
        for c, p in prev:
            if p is not None:
                c.use_cache = True
        try:
            with self._decode_autocast():
                yield
        finally:
            if gc_was_on:
                self.set_grad_checkpointing(True)
            for c, p in prev:
                if p is not None:
                    c.use_cache = p

    # ---- packaging + decode I/O ----

    def pack_example(self, source_ids, seen_tokens, label_tokens, *, scores_separator: bool = False):
        """Assemble a training example from a linearized target (`seen`/`label`
        token streams, terminator appended here): two aligned streams for seq2seq,
        one prefixed causal stream for decoder_only. `scores_separator` asks the
        backbone to score the transition out of the source prefix (an in-stream
        concept; seq2seq has no in-stream SEP and ignores it). Returns the
        backbone-shaped example (`(input_ids, labels)` tuple for decoder_only, a
        dict for seq2seq) or None if it overflows."""
        raise NotImplementedError

    def forward_logits(self, batch):
        """Run the model on a packed batch, returning `(shifted_logits,
        shifted_labels)` aligned for the serialization's loss (a causal shift for
        decoder_only; the decoder outputs for seq2seq)."""
        raise NotImplementedError

    def collate(self, examples, pad_id):
        """Pad a list of `pack_example` outputs into a batched tensor dict on CPU
        (the train loop moves it to device). The example shape is backbone-owned
        (a single (input_ids, labels) stream for decoder_only; an encoder+decoder
        dict for seq2seq), so batching lives here."""
        raise NotImplementedError

    # ---- new-row training: shadow-Parameter scheme ----

    def embedding_weight(self):
        """The input embedding weight (one tied storage for seq2seq encoder+decoder)."""
        return self.underlying_model().get_input_embeddings().weight

    def lm_head_weight(self):
        """The output-projection (unembedding) weight, `[vocab, hidden]`, across the
        lm_head layouts we support: `lm_head` as a Linear (decoder_only, T5/mT5) or a
        nested `lm_head.out_proj` (T5Gemma 2), each possibly PEFT-wrapped
        (`.base_layer.weight`). None if it can't be located."""
        base = self.underlying_model()
        head = getattr(base, "lm_head", None)
        if head is None:
            return None
        # T5Gemma 2 nests the projection under `.out_proj`; others expose it directly.
        out_proj = getattr(head, "out_proj", None)
        layer = out_proj if out_proj is not None else head
        w = getattr(layer, "weight", None)
        if w is None and hasattr(layer, "base_layer"):
            w = getattr(layer.base_layer, "weight", None)
        return w

    def lm_head_tied_to_embeddings(self) -> bool:
        """Whether the lm_head shares storage with the input embedding (tied). The
        full-lm_head modes (words / use_copy=False) depend on this under LoRA: a tied
        head lets the input-embedding new rows reach the output side (so the input
        shadow alone suffices), while an untied head needs its own output-row shadow."""
        w = self.lm_head_weight()
        emb = self.underlying_model().get_input_embeddings().weight
        return w is not None and w.data_ptr() == emb.data_ptr()

    def _shadow_target_weights(self):
        """Weight matrices whose new (id >= the pre-add boundary) rows this shadow install
        trains, in a stable order: the input embedding always, then the untied lm_head when
        `_shadow_includes_lm_head`. `install_new_row_shadow` and `_shadow_pairs` both build
        their lists from this, so ordering + membership stay in lockstep."""
        weights = [self.embedding_weight()]
        if getattr(self, "_shadow_includes_lm_head", False):
            head_w = self.lm_head_weight()
            if head_w is not None:
                weights.append(head_w)
        return weights

    def install_new_row_shadow(self, *, include_untied_lm_head: bool = False):
        """Train only the newly-added (id >= the pre-add boundary) rows through small
        shadow Parameters, keeping the full matrices they mirror OUT of the optimizer.

        Always shadows the input embedding. In full-head modes (words / use_copy=False)
        under LoRA with an UNTIED lm_head, `include_untied_lm_head` adds a SECOND shadow
        over the lm_head's new OUTPUT rows: LoRA freezes the base lm_head and, untied, the
        input-embedding shadow cannot reach the output side, so the new label/copy rows
        would score from their random init. A tied lm_head shares the input embedding's
        storage, so the input shadow already trains both sides (no second shadow, which
        would double-count).

        The alternative (full matrices in the optimizer, a hook zeroing pretrained-row
        grads) OOMs at 27-31B (HF Adafactor upcasts the param + its grad to fp32 inside
        step(), ~17GB of transients per matrix on top of the bf16 weights) and lets
        weight_decay drift the frozen rows. The shadows avoid both: the optimizer steps
        `[n_new, hidden]` per matrix, and the trainer calls `stage_new_embedding_row_grads`
        / `commit_new_embedding_rows` around each step. The forwards are never overridden,
        so backbone lookup behavior (Gemma sqrt(hidden) scaling) is preserved. Returns
        `(n_total, n_new)` or None.

        The shadows are deliberately UNregistered (not in state_dict): the rows they mirror
        ARE in state_dict and are identical after every commit, so the checkpoint schema is
        unchanged and old checkpoints strict-load. A load hook re-syncs them. `model.to()`
        skips them (optimizer_parameters re-homes them), so the trainer must clip over
        `GenParser.optimizer_parameters()`, not `model.parameters()`."""
        # The shadows live in a plain list, NOT as module attributes: assigning an
        # nn.Parameter attribute on an nn.Module registers it into the state_dict,
        # which is exactly what "unregistered" must avoid.
        self._new_row_shadow_holder: list = []
        self._shadow_includes_lm_head = False
        if self._original_vocab_size is None:
            return None
        emb = self.embedding_weight()
        n_old, n_total = self._original_vocab_size, emb.shape[0]
        if n_total <= n_old:
            return None
        if include_untied_lm_head:
            head_w = self.lm_head_weight()
            self._shadow_includes_lm_head = head_w is not None and head_w is not emb
        for weight in self._shadow_target_weights():
            weight.requires_grad_(True)  # backward must materialize the grad we slice from
            self._new_row_shadow_holder.append(nn.Parameter(weight.detach()[n_old:].clone()))
        self.register_load_state_dict_post_hook(lambda module, _incompat: module._sync_shadow_from_weight())
        return n_total, n_total - n_old

    def new_row_shadow_params(self):
        """The new-row shadow Parameters (input embedding, plus the untied lm_head when
        installed), for the trainer to place in a separate optimizer group at new_row_lr.
        Empty when no shadow is installed."""
        return list(getattr(self, "_new_row_shadow_holder", []))

    def _shadow_pairs(self):
        """Live `(weight, shadow)` pairs in install order (input embedding, then the untied
        lm_head). Empty when no shadow is installed."""
        holder = getattr(self, "_new_row_shadow_holder", [])
        if not holder:
            return []
        weights = self._shadow_target_weights()
        if len(weights) != len(holder):
            raise RuntimeError(
                f"Shadow target count drifted: {len(weights)} weights vs {len(holder)} shadows "
                f"(the lm_head weight moved after install?)."
            )
        return list(zip(weights, holder))

    def shadow_pairs(self):
        """Live `(weight, shadow)` pairs (input embedding, then the untied lm_head), or
        empty when no shadow is installed. Public for the checkpoint layer: a
        trainable-only checkpoint persists each SHADOW (the compact new rows) instead of
        its full matrix, which carries `requires_grad` only so backward materializes a
        grad to slice."""
        return self._shadow_pairs()

    def _sync_one(self, weight, shadow) -> None:
        with torch.no_grad():
            shadow.data = weight.detach()[self._original_vocab_size :].clone()
        shadow.grad = None

    def _sync_shadow_from_weight(self) -> None:
        for weight, shadow in self._shadow_pairs():
            self._sync_one(weight, shadow)

    def shadow_optimizer_params(self, trainable):
        """Drop the shadowed full matrices (input embedding, untied lm_head) from
        `trainable` (the requires_grad params) and add the re-homed shadows that train in
        their stead. No shadow -> `trainable` unchanged. The shadows are unregistered so
        `.to()` skips them; re-home each to its matrix here (call only before building the
        optimizer or per-step, never mid-optimizer)."""
        pairs = self._shadow_pairs()
        if not pairs:
            return trainable
        weights = [w for w, _ in pairs]
        params = [p for p in trainable if all(p is not w for w in weights)]
        for weight, shadow in pairs:
            if shadow.device != weight.device or shadow.dtype != weight.dtype:
                self._sync_one(weight, shadow)
            params.append(shadow)
        return params

    def stage_new_embedding_row_grads(self) -> None:
        """After the last backward of an accumulation window, BEFORE clipping: for each
        shadowed matrix move the new-row slice of its grad onto the shadow's grad and free
        the full grad (so clipping sees the slice and the optimizer's fp32 transient never
        forms)."""
        n_old = self._original_vocab_size
        for weight, shadow in self._shadow_pairs():
            if weight.grad is None:
                continue
            g = weight.grad[n_old:].detach()
            shadow.grad = g.clone().to(shadow.dtype) if shadow.grad is None else shadow.grad + g.to(shadow.dtype)
            weight.grad = None

    def commit_new_embedding_rows(self) -> None:
        """After optimizer.step(): write each stepped shadow back into its live matrix's
        new rows (the forwards always read the matrices)."""
        n_old = self._original_vocab_size
        with torch.no_grad():
            for weight, shadow in self._shadow_pairs():
                weight[n_old:].copy_(shadow)

    def tokenize_source(self, text: str) -> list[int]:
        """Decode-time source tokenization (the COPY stream). May truncate."""
        raise NotImplementedError

    def decode_prefix(self, source_ids: list[int]) -> list[int]:
        """The prefix the output stream is decoded after (causal) / the encoder
        input (seq2seq)."""
        raise NotImplementedError

    def max_decode_steps(self, prefix_len: int) -> int | None:
        """Positional-limit cap on generated steps after a `prefix_len`-token decode
        prefix, or None when the family imposes no cap beyond `max_output_length`.
        decoder_only overrides this: its single causal stream spends positions on the
        prefix, mirroring pack_example's combined-stream cap at training time, so
        without it inference could silently generate past max_position_embeddings
        (RoPE extrapolation, no error). seq2seq's decoder stream starts fresh, so the
        default (None) stands."""
        return None

    def seed(self, prefix_ids, num_rows: int = 1):
        """First decode step: return `(logits, cache)` after encoding the source /
        seeding the causal prefix, replicated across `num_rows` beams."""
        raise NotImplementedError

    def advance(self, cache, next_input):
        """One decode step: feed `next_input`, return `(logits, cache)`."""
        raise NotImplementedError

    # ---- batched decode I/O (B DIFFERENT documents, one forward per step) ----
    #
    # Distinct from seed/advance's `num_rows`, which replicates ONE document across
    # K beams: every row here carries its own document, so the rows have different
    # prefix lengths and the batch needs real padding + per-row positions. The
    # per-document decode is unchanged mathematically -- padding is attention-masked
    # out and each row's positions are set explicitly -- so a row's logits do not
    # depend on what else shares its batch (see `greedy_decode_batch`).

    def seed_rows(self, prefix_ids_rows: list[list[int]]):
        """Prefill B different documents in one forward. Returns `(logits [B, V],
        cache)`, where `logits[i]` is the next-token distribution after
        `prefix_ids_rows[i]` -- exactly what `seed(prefix_ids_rows[i])` returns."""
        raise NotImplementedError

    def advance_rows(self, cache, next_inputs: list[int]):
        """One decode step for every row of a `seed_rows` batch. `next_inputs[i]` is
        row i's token (retired rows pass a pad the caller ignores). Returns
        `(logits [B, V], cache)`."""
        raise NotImplementedError

    def beam_reorder_needed(self, step: int, parents: list[int], k: int, cache) -> bool:
        """Whether the beam step needs a KV-cache reorder by `parents`. Behind a
        backbone method because the loop's `cache` is opaque: for decoder_only it IS
        the past_key_values; for seq2seq it is compound (encoder outputs + decoder
        past), so each backbone extracts the right piece for the no-op check."""
        raise NotImplementedError

    def reorder_cache(self, cache, parent_tensor):
        """Reorder the decode cache along the beam dimension by `parent_tensor`."""
        raise NotImplementedError
