"""Unified generative RST parser `gen`.

`GenParser` composes a backbone strategy (`seq2seq` | `decoder_only`) and a
serialization strategy (`sr` | `sexp`), selected by config, into one parser that
subsumes the four hand-written generative parsers. The composition is functional:
the two axes never reference each other (a backbone knows nothing about SR vs
sexp, a serialization nothing about encoder-decoder vs causal). The generic decode
core (`decode.py`) parameterizes one greedy and one beam loop over both, plus a
mask source (validity for pred-EDU, gold forcing for gold-EDU).

The backbone is an `nn.Module` that owns `self.model`, so `GenParser`'s state_dict
nests the weights under `backbone.model.*`. The serialization owns its decode
buffers but no weights, and shares the backbone's tokenizer.
"""

import logging

import torch
import torch.nn as nn

from iudex.common.training import raise_on_unexpected_keys
from iudex.rst.parsers.common.seqgen import empty_tree, gold_edu_source_ranges, reconstruct_text
from iudex.rst.parsers.gen.backbones import BACKBONES
from iudex.rst.parsers.gen.configuration_gen import GenConfig
from iudex.rst.parsers.gen.decode import beam_decode, greedy_decode, greedy_decode_batch
from iudex.rst.parsers.gen.serializations import SERIALIZATIONS

logger = logging.getLogger(__name__)

# Namespace for the compact new-row slices a trainable-only checkpoint carries in
# place of the full matrices they belong to. Not a state_dict key: `load_trainable_state_dict`
# pops these before `load_state_dict` sees them.
NEW_ROW_KEY_PREFIX = "_new_rows."


class GenParser(nn.Module):
    def __init__(self, config: GenConfig, *, compile_encoder: bool = False):
        super().__init__()
        self.config = config
        # compile_encoder is accepted for parser-CLI uniformity; no effect here.
        del compile_encoder

        # The backbone owns the model + tokenizer; the serialization shares that
        # tokenizer and owns the head layout. Construction order mirrors the
        # hand-written parsers so the state_dict matches (modulo the backbone.
        # prefix): build -> add vocab + resize -> peft -> small head (token mode).
        self.backbone = BACKBONES[config.backbone](config)
        self.serialization = SERIALIZATIONS[config.serialization](config)
        self.serialization.tokenizer = self.backbone.tokenizer

        if config.relation_types is not None:
            new_tokens = self.backbone.stream_special_tokens() + self.serialization.action_tokens()
            self.backbone.add_tokens_and_resize(new_tokens)
            self.serialization.build_vocab(self.backbone.stream_end_id)

        if config.peft is not None:
            self.backbone.install_peft(config.peft)

        if config.relation_types is not None and self.serialization.uses_small_head():
            self.backbone.install_head(self.serialization.head_spec())

        if config.gradient_checkpointing:
            self.backbone.enable_grad_checkpointing()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def tokenizer(self):
        # The shared tokenizer lives on the backbone; expose it here so callers
        # (the shared evaluate_on_dev, the predict CLI) reach it uniformly.
        return self.backbone.tokenizer

    @property
    def segmenter(self):
        # Truthy so predict_cli._require_segmenter admits raw text: this parser
        # recovers segmentation from the model's own decoded output (no head).
        return self

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str,
        *,
        device=None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token=None,
        compile_encoder: bool = False,
    ) -> "GenParser":
        from iudex.rst.parsers.hfhub import load_parser_from_pretrained

        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return load_parser_from_pretrained(
            repo_or_path, parser_cls=cls, config_cls=GenConfig, device=dev,
            revision=revision, cache_dir=cache_dir, token=token, compile_encoder=compile_encoder,
        )

    def encode_target(self, tree):
        """Build the training example for a tree: the serialization linearizes it,
        the backbone packs it into the model's stream. Returns the backbone-shaped
        example (see `Backbone.pack_example`) or None when it overflows the
        configured budget."""
        source_ids, seen, label = self.serialization.linearize(tree)
        return self.backbone.pack_example(
            source_ids, seen, label, scores_separator=self.serialization.scores_separator
        )

    def forward(self, batch: dict) -> dict:
        """Training pass: the backbone runs the model and aligns logits/labels, the
        serialization computes the CE loss + its structural/copy split."""
        shifted_logits, shifted_labels = self.backbone.forward_logits(batch)
        return self.serialization.loss_terms(shifted_logits, shifted_labels)

    # ---- prediction ----

    @torch.no_grad()
    def predict_from_text(self, text: str, *, num_beams: int | None = None) -> "RstTree":
        return self.predict_batch_from_texts([text], num_beams=num_beams)[0]

    @torch.no_grad()
    def predict_batch_from_texts(self, texts: list[str], *, num_beams: int | None = None) -> list:
        if not texts:
            return []
        effective_beams = int(num_beams if num_beams is not None else self.config.num_beams)
        self.eval()
        if effective_beams > 1:
            # Beam already fills the batch dim with one document's K beams.
            return [self._predict_one_beam(t, effective_beams) for t in texts]
        if self._batches_decode(texts):
            ser = self.serialization
            docs = []
            for text in texts:
                source_ids = self.backbone.tokenize_source(text)
                docs.append((source_ids, self._pred_factory(source_ids), ser.pred_mask))
            return greedy_decode_batch(self, docs)
        return [self._predict_one_greedy(t) for t in texts]

    def decodes_in_batches(self, *, num_beams: int | None = None) -> bool:
        """Whether a `predict_batch` at this beam width decodes its argument in one
        shared forward (vs a document at a time). The shared eval keys length-bucketing
        + the batched gold-EDU pass off this. Greedy + `batched_decode` only; beam fills
        the batch dim with one document's K beams, so it never batches documents."""
        effective_beams = int(num_beams) if num_beams is not None else int(self.config.num_beams)
        return self.config.batched_decode and effective_beams <= 1

    def _batches_decode(self, items: list) -> bool:
        """Whether to decode `items` in one batched greedy pass. A single document is
        left on the per-document path: batching it would be the same arithmetic (one
        row, no padding) but the serial loop is what every published number ran, so
        there is no reason to move it."""
        return self.config.batched_decode and len(items) > 1

    def _pred_factory(self, source_ids):
        ser = self.serialization
        return lambda: ser.initial_state(source_ids)

    @torch.no_grad()
    def _predict_one_greedy(self, text: str):
        self.eval()
        source_ids = self.backbone.tokenize_source(text)
        ser = self.serialization
        factory = lambda: ser.initial_state(source_ids)  # noqa: E731
        return greedy_decode(self, source_ids, factory, ser.pred_mask)

    @torch.no_grad()
    def _predict_one_beam(self, text: str, num_beams: int):
        self.eval()
        source_ids = self.backbone.tokenize_source(text)
        ser = self.serialization
        factory = lambda: ser.initial_state(source_ids)  # noqa: E731
        return beam_decode(self, source_ids, num_beams, factory, ser.pred_mask)

    @torch.no_grad()
    def predict_with_gold_edus(self, tree, *, num_beams: int | None = None):
        """Decode with gold EDU boundaries forced at the copy/shift positions;
        structure stays model-driven. num_beams None/<=1 is the greedy forced
        decode."""
        K = int(num_beams) if num_beams is not None else 1
        if K <= 1:
            return self._predict_one_gold_edu(tree)
        return self._predict_one_gold_edu_beam(tree, K)

    @torch.no_grad()
    def predict_batch_with_gold_edus(self, trees: list, *, num_beams: int | None = None) -> list:
        """`predict_with_gold_edus` over several trees. The gold-EDU pass emits about
        as many tokens as the e2e pass (forcing changes the mask, not the length), so
        running it a document at a time doubles the cost of a `eval_gold_edu` eval;
        under `batched_decode` it shares the batched greedy core with e2e, differing
        only in which factory/mask_source each row carries."""
        if not trees:
            return []
        effective_beams = int(num_beams) if num_beams is not None else 1
        self.eval()
        if effective_beams > 1 or not self._batches_decode(trees):
            return [self.predict_with_gold_edus(t, num_beams=num_beams) for t in trees]
        ser = self.serialization
        docs = []
        empty_at: set[int] = set()
        for i, tree in enumerate(trees):
            source_ids, gold_ranges = self._gold_edu_setup(tree)
            if gold_ranges is None:
                # Nothing to force a decode against; `_predict_one_gold_edu` returns the
                # degenerate tree without touching the model. Keep it out of the batch
                # (an empty `source_ids` row has no prefix) and splice it back after.
                empty_at.add(i)
                continue
            docs.append((
                source_ids,
                self._gold_factory(source_ids, gold_ranges),
                ser.gold_mask_source(gold_ranges, len(source_ids)),
            ))
        decoded = iter(greedy_decode_batch(self, docs) if docs else ())
        return [empty_tree(self.config.relation_types) if i in empty_at else next(decoded) for i in range(len(trees))]

    def _gold_factory(self, source_ids, gold_ranges):
        ser = self.serialization
        return lambda: ser.gold_initial_state(source_ids, gold_ranges)

    def _gold_edu_setup(self, tree):
        """(source_ids, gold_ranges) for gold-forced decode, or (source_ids, None)
        when there is nothing to decode (caller returns empty_tree). gold_ranges are
        the clamped (start, end) source-token spans of the gold EDUs (SR reads the
        ends off them, sexp seeds its GoldEduForcer with them)."""
        text = reconstruct_text(tree)
        gold_ranges = gold_edu_source_ranges(self.backbone.tokenizer, tree)
        source_ids = self.backbone.tokenize_source(text)
        if not source_ids:
            return source_ids, None
        source_len = len(source_ids)
        clamped: list[tuple[int, int]] = []
        for s, e in gold_ranges:
            if s >= source_len:
                break
            clamped.append((s, min(e, source_len)))
        if not clamped:
            return source_ids, None
        return source_ids, clamped

    @torch.no_grad()
    def _predict_one_gold_edu(self, tree):
        self.eval()
        source_ids, gold_ranges = self._gold_edu_setup(tree)
        if gold_ranges is None:
            return empty_tree(self.config.relation_types)
        ser = self.serialization
        factory = lambda: ser.gold_initial_state(source_ids, gold_ranges)  # noqa: E731
        mask_source = ser.gold_mask_source(gold_ranges, len(source_ids))
        return greedy_decode(self, source_ids, factory, mask_source)

    @torch.no_grad()
    def _predict_one_gold_edu_beam(self, tree, num_beams: int):
        self.eval()
        source_ids, gold_ranges = self._gold_edu_setup(tree)
        if gold_ranges is None:
            return empty_tree(self.config.relation_types)
        ser = self.serialization
        factory = lambda: ser.gold_initial_state(source_ids, gold_ranges)  # noqa: E731
        mask_source = ser.gold_mask_source(gold_ranges, len(source_ids))
        return beam_decode(self, source_ids, num_beams, factory, mask_source)

    @torch.no_grad()
    def predict(self, tree, *, num_beams: int | None = None):
        return self.predict_from_text(reconstruct_text(tree), num_beams=num_beams)

    @torch.no_grad()
    def predict_batch(self, trees: list, *, num_beams: int | None = None) -> list:
        return self.predict_batch_from_texts([reconstruct_text(t) for t in trees], num_beams=num_beams)

    # ---- training setup ----

    def configure_new_row_training(self):
        """Install the new-row shadow(s) so only the newly-added token rows train (call
        once before building the optimizer). Returns `(n_total, n_new)` or None. The
        shadows are unregistered and their mirrored rows stay in the state_dict, so this
        never affects decode or the checkpoint schema.

        Always shadows the input embedding. In full-head modes (words / use_copy=False)
        with an UNTIED lm_head, the new <copy>/<shift>/<label_end>/label OUTPUT rows live in
        the full head; a second shadow trains them when they would otherwise be starved:
        under LoRA (the base head is frozen) or when a separate new_row_lr must reach them
        (the full head otherwise trains at the backbone lr). Shadowing the head freezes its
        pretrained rows, matching the already-frozen input side. A tied head needs no second
        shadow (the input shadow reaches the output side). (use_copy=False + peft is rejected
        in GenConfig.__post_init__, so full_head under peft means words mode.)"""
        full_head = not self.serialization.uses_small_head()
        untied = full_head and not self.backbone.lm_head_tied_to_embeddings()
        train_untied_head = untied and (self.config.peft is not None or self.config.new_row_lr is not None)
        if train_untied_head and self.backbone.lm_head_weight() is None:
            raise NotImplementedError(
                "label_style='words' under LoRA on an untied-lm_head backbone must train the new label/copy "
                "OUTPUT rows via a shadow, but this backbone's lm_head could not be located (expected `lm_head` "
                "or `lm_head.out_proj` as a Linear). Use full fine-tuning (peft=null)."
            )
        return self.backbone.install_new_row_shadow(include_untied_lm_head=train_untied_head)

    def optimizer_parameters(self):
        """Parameters the optimizer steps (and the trainer clips over): everything
        trainable EXCEPT the full input embedding, plus the new-row shadow that trains
        in its stead (see Backbone.install_new_row_shadow). Without a shadow this is
        just the trainable params."""
        return self.backbone.shadow_optimizer_params([p for p in self.parameters() if p.requires_grad])

    def new_row_params(self):
        """The new-row shadow params, for the trainer's separate new_row_lr optimizer
        group. Empty when no shadow is installed (small-head modes, or no new tokens)."""
        return self.backbone.new_row_shadow_params()

    def stage_new_embedding_row_grads(self):
        self.backbone.stage_new_embedding_row_grads()

    def commit_new_embedding_rows(self):
        self.backbone.commit_new_embedding_rows()

    # ---- trainable-only checkpoint state ----

    def _shadow_by_param_name(self) -> dict:
        """state_dict name -> its new-row shadow, for every shadowed matrix. Keyed by
        identity, since the name lives on GenParser and the shadow on the backbone."""
        pairs = self.backbone.shadow_pairs()
        if not pairs:
            return {}
        out = {}
        for name, p in self.named_parameters():
            for weight, shadow in pairs:
                if p is weight:
                    out[name] = shadow
        return out

    def trainable_state_dict(self) -> dict:
        """The trainable-only checkpoint state: the `requires_grad` params EXCEPT the
        shadowed full matrices, plus each shadow's compact new rows under
        `{NEW_ROW_KEY_PREFIX}<param name>`.

        The shadow scheme flags the whole embedding `requires_grad` only so backward
        materializes a grad to slice -- the optimizer trains the shadow, never the
        matrix (see `optimizer_parameters`). Selecting by `requires_grad` alone would
        therefore put a multi-GB embedding in a LoRA checkpoint (measured: 2.37GB of a
        2.83GB r=16 Qwen3.6-27B checkpoint) to carry a few new rows. The rows are the
        only trained part, so persist exactly those.
        """
        shadow_by_name = self._shadow_by_param_name()
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        state = {k: v for k, v in self.state_dict().items() if k in trainable and k not in shadow_by_name}
        for name, shadow in shadow_by_name.items():
            state[NEW_ROW_KEY_PREFIX + name] = shadow.detach().clone()
        return state

    def load_trainable_state_dict(self, state: dict) -> None:
        """Inverse of `trainable_state_dict`: load the ordinary params, then overlay the
        saved new rows onto the freshly-constructed matrices (`commit_new_embedding_rows`
        is the same write the training loop does after every step).

        Old checkpoints carry the full matrix and no `{NEW_ROW_KEY_PREFIX}` keys; they
        load unchanged (the overlay just doesn't fire).
        """
        new_rows = {k[len(NEW_ROW_KEY_PREFIX) :]: v for k, v in state.items() if k.startswith(NEW_ROW_KEY_PREFIX)}
        rest = {k: v for k, v in state.items() if not k.startswith(NEW_ROW_KEY_PREFIX)}
        raise_on_unexpected_keys(self.load_state_dict(rest, strict=False))
        if not new_rows:
            return
        shadow_by_name = self._shadow_by_param_name()
        if not shadow_by_name:
            # Standalone inference/eval constructs a fresh parser and loads it
            # immediately, unlike training which installs the shadows before resume.
            # Compact checkpoints must therefore make their own load prerequisite.
            self.configure_new_row_training()
            shadow_by_name = self._shadow_by_param_name()
        for name, rows in new_rows.items():
            shadow = shadow_by_name.get(name)
            if shadow is None:
                raise RuntimeError(
                    f"Checkpoint carries new embedding rows for {name!r}, but this model has no "
                    f"shadow installed for it (shadowed: {sorted(shadow_by_name)})."
                )
            if tuple(rows.shape) != tuple(shadow.shape):
                raise RuntimeError(
                    f"Checkpoint's new rows for {name!r} are {tuple(rows.shape)}, but this model's "
                    f"shadow is {tuple(shadow.shape)}. Vocab/architecture mismatch."
                )
            with torch.no_grad():
                shadow.data = rows.to(device=shadow.device, dtype=shadow.dtype).clone()
        self.backbone.commit_new_embedding_rows()
