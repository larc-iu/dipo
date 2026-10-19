"""EDU segmentation for the Maekawa parser, as one more LoRA adapter on its base model.

The segmenter is `gen`'s `seg` serialization (copy every source token, close each EDU with
a shift, so the EDUs always tile the text) on `gen`'s decoder-only backbone, built around
the parser's own model and tokenizer instead of loading a second copy. It adds gen's
special tokens to the shared vocabulary (their new embedding rows train with the
segmenter only), owns the "segment" adapter and a small three-way output head, and is
switched in for segmentation by `active()`: the adapter becomes the active one and the
head stands in for the LM head, then both are restored.
"""

import contextlib

import torch
import torch.nn as nn

from dipo.rst.data.tree import RstTree
from dipo.rst.parsers.common.seqgen import warm_init_head
from dipo.rst.parsers.gen.backbones.base import Backbone
from dipo.rst.parsers.gen.backbones.decoder_only import DecoderOnlyBackbone
from dipo.rst.parsers.gen.configuration_gen import GenConfig
from dipo.rst.parsers.gen.modeling_gen import GenParser
from dipo.rst.parsers.gen.serializations.seg import SegSerialization

SEGMENT_TASK = "segment"


class _SharedBackbone(DecoderOnlyBackbone):
    """gen's decoder-only backbone over a model and tokenizer that already exist."""

    def __init__(self, config: GenConfig, model, tokenizer):
        Backbone.__init__(self, config)
        self.sep_token_id = None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.stream_end_id = int(tokenizer.eos_token_id)
        self.model = model


class SharedSegmenter(GenParser):
    """A `gen` segmenter whose backbone is another parser's (PEFT-wrapped) causal LM."""

    def __init__(self, config: GenConfig, model, tokenizer):
        nn.Module.__init__(self)
        self.config = config
        self.backbone = _SharedBackbone(config, model, tokenizer)
        self.serialization = SegSerialization(config)
        self.serialization.tokenizer = tokenizer
        self.backbone.add_tokens_and_resize(self.backbone.stream_special_tokens() + self.serialization.action_tokens())
        self.serialization.build_vocab(self.backbone.stream_end_id)
        spec = self.serialization.head_spec()
        unembed = model.get_output_embeddings().weight
        self.serialization.to(unembed.device)  # its loss/decode lookup buffers start on CPU
        self.head = nn.Linear(unembed.shape[1], spec["head_vocab_size"], bias=False).to(unembed.device, unembed.dtype)
        warm_init_head(self.head, unembed, spec["full_id_for_head_idx"])

    def attach(self, peft_model) -> None:
        """Point the backbone at the PEFT wrapper once the parser has installed its adapters."""
        self.backbone.model = peft_model

    @contextlib.contextmanager
    def active(self):
        """Route the model through the segment adapter and the segmenter's head."""
        peft_model = self.backbone.model
        base = peft_model.get_base_model()
        lm_head = base.get_output_embeddings()
        previous = peft_model.active_adapter
        peft_model.set_adapter(SEGMENT_TASK)
        base.set_output_embeddings(self.head)
        try:
            yield
        finally:
            base.set_output_embeddings(lm_head)
            peft_model.set_adapter(previous)

    def segment(self, texts: list[str]) -> list[RstTree]:
        """Trees over each text's predicted EDUs (placeholder structure), carrying the
        source token ranges gen's evaluation keys segmentation on."""
        with self.active():
            return self.predict_batch_from_texts(texts)

    # ---- the segmenter's trained state: its LoRA adapter, head and new token rows ----

    def segment_state_dict(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        state = {
            f"lora.{k}": v for k, v in get_peft_model_state_dict(self.backbone.model, adapter_name=SEGMENT_TASK).items()
        }
        state.update({f"head.{k}": v for k, v in self.head.state_dict().items()})
        # The trained new token rows (kept in the matrix after every commit).
        state["new_rows"] = self.backbone.embedding_weight()[self.backbone._original_vocab_size :]
        return {k: v.detach().cpu().clone() for k, v in state.items()}

    def load_segment_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        from peft import set_peft_model_state_dict

        lora = {k[len("lora.") :]: v for k, v in state.items() if k.startswith("lora.")}
        result = set_peft_model_state_dict(self.backbone.model, lora, adapter_name=SEGMENT_TASK)
        unexpected = getattr(result, "unexpected_keys", [])
        if unexpected:
            raise RuntimeError(f"segment adapter: unexpected keys, e.g. {unexpected[:3]}")
        self.head.load_state_dict({k[len("head.") :]: v for k, v in state.items() if k.startswith("head.")})
        rows = state["new_rows"]
        weight = self.backbone.embedding_weight()
        n_old = self.backbone._original_vocab_size
        if tuple(rows.shape) != tuple(weight[n_old:].shape):
            raise RuntimeError(f"segmenter new rows are {tuple(rows.shape)}, the model's {tuple(weight[n_old:].shape)}")
        with torch.no_grad():
            weight[n_old:].copy_(rows.to(weight.device, weight.dtype))
        self.backbone._sync_shadow_from_weight()


def segmenter_config(parser_config, relation_types) -> GenConfig:
    """The gen config the segmenter runs under, from the Maekawa config."""
    seg = parser_config.segmentation
    return GenConfig(
        train_dir=parser_config.train_dir,
        dev_dir=parser_config.dev_dir,
        backbone="decoder_only",
        serialization="seg",
        relation_types=relation_types,
        model_name=parser_config.model_name,
        max_input_length=seg.max_input_tokens,
        max_output_length=2 * seg.max_input_tokens,
        min_edu_length=seg.min_edu_length,
        batched_decode=True,
        dev_batch_size=seg.inference_batch_size,
        amp=False,  # the shared model already runs in its own dtype
    )
