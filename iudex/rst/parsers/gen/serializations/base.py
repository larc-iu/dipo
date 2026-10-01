"""Serialization strategy interface for the unified generative parser `gen`.

A serialization owns everything that changes when the tree linearization is
swapped: `sr` (bottom-up shift-reduce action sequence) vs `sexp` (nested
s-expression). That is the action vocabulary and scoring-head layout, the
`tree -> tokens` linearization, the decode state machine + legal-action masking,
loss shaping, and tree reconstruction.

It is an `nn.Module` so its decode buffers (label lookups etc.) move with the
parser, but it owns NO model weights: the backbone owns the model. It reads the
shared tokenizer (assigned by `GenParser` after construction) and never touches
the model or the backbone. The only handshake to the backbone is `head_spec()`
(what small head to install, in token mode).

Subclasses populate their vocab/head state in `build_vocab` and implement the
behavior methods below (linearize / decode / reconstruct); `loss_terms` is shared
(identical across serializations) and lives here.
"""

import contextlib

import torch
import torch.nn as nn

from iudex.rst.parsers.common.seqgen import chunked_cross_entropy


class Serialization(nn.Module):
    # Whether the model scores the SEP position (the transition out of the source
    # prefix). Serialization-specific packing detail honored by the backbone.
    scores_separator: bool = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = None  # shared tokenizer, assigned by GenParser

    # ---- construction: vocab + scoring-head layout ----

    def action_tokens(self) -> list[str]:
        """Special tokens this serialization adds to the tokenizer, besides the
        backbone's stream separator."""
        raise NotImplementedError

    def uses_small_head(self) -> bool:
        """True when actions are scored over a small fresh lm_head the backbone
        installs (token mode); False keeps the full pretrained lm_head (words /
        use_copy=False)."""
        raise NotImplementedError

    def build_vocab(self, stream_end_id: int) -> None:
        """After the backbone has added `action_tokens()` + its separator and
        resized, read the assigned ids and build the head layout + decode buffers.
        `stream_end_id` is the backbone's stream terminator."""
        raise NotImplementedError

    def head_spec(self) -> dict:
        """Handshake to `Backbone.install_head` (token mode only): the small head
        size and the full-vocab ids to warm-init its rows from."""
        raise NotImplementedError

    def _register_head_buffers(self, structural_head_ids) -> None:
        """Token mode: register the label lookup (full-vocab id -> head index, -100
        elsewhere) and the structural head indices, both consumed by `loss_terms`.
        Requires `full_id_for_head_idx` / `head_idx_for_full_id` to be built."""
        max_full_id = max(self.full_id_for_head_idx) + 1
        lookup = torch.full((max_full_id,), -100, dtype=torch.long)
        for fid, hi in self.head_idx_for_full_id.items():
            lookup[fid] = hi
        self.register_buffer("_label_to_head_lookup", lookup, persistent=False)
        self.register_buffer(
            "_structural_token_ids_buf",
            torch.tensor(sorted(structural_head_ids), dtype=torch.long),
            persistent=False,
        )

    def _register_structural_full_ids(self, structural_full_ids) -> None:
        """Full-head modes (words / use_copy=False): register the full-vocab
        structural ids consumed by `loss_terms`."""
        self.register_buffer(
            "_structural_full_ids_buf",
            torch.tensor(sorted(structural_full_ids), dtype=torch.long),
            persistent=False,
        )

    # ---- behavior (later phases) ----

    def linearize(self, tree):
        """`tree -> (target_token_ids, aux)`. aux carries per-token side info the
        backbone's packing / the loss needs (copy positions, width weights)."""
        raise NotImplementedError

    def loss_terms(self, shifted_logits, shifted_labels) -> dict:
        """CE over the output positions with a structural-vs-copy split, shared by
        every serialization: token mode remaps full-vocab labels to head indices,
        words / use_copy=False score over the full lm_head. `action_loss_weight`
        optionally upweights the structural positions. Loss reduction is per
        document first, then across documents, so callers can apply a distinct
        document weight without losing its association with that document.
        Requires the subclass to have built `head_vocab_size`,
        `_label_to_head_lookup` (small head), and `_structural_token_ids_buf` /
        `_structural_full_ids_buf`."""
        batch_size = shifted_labels.shape[0]
        labels_flat = shifted_labels.reshape(batch_size, -1)
        if self.uses_small_head():
            max_id = self._label_to_head_lookup.size(0) - 1
            in_range = (labels_flat >= 0) & (labels_flat <= max_id)
            clamped = labels_flat.clamp(min=0, max=max_id)
            head_labels_flat = torch.where(
                in_range, self._label_to_head_lookup[clamped], torch.full_like(labels_flat, -100)
            )
            structural_buf = self._structural_token_ids_buf
        else:
            head_labels_flat = labels_flat
            structural_buf = self._structural_full_ids_buf

        logits_flat = shifted_logits.reshape(batch_size, -1, self.head_vocab_size)
        losses: list[torch.Tensor] = []
        base_losses: list[torch.Tensor] = []
        action_losses: list[tuple[torch.Tensor, int]] = []
        counts: list[tuple[int, int]] = []
        w = self.config.action_loss_weight
        classify_structural = structural_buf.numel() > 0

        for doc_logits, doc_labels in zip(logits_flat, head_labels_flat, strict=True):
            valid_mask = doc_labels != -100
            valid_idx = valid_mask.nonzero(as_tuple=True)[0]
            # Full-vocab CE over only the scored positions, computed in row-chunks
            # so the fp32 logits + log-softmax peak is [chunk, V], never
            # [n_valid, V]. Reduction happens independently per document.
            base_loss = chunked_cross_entropy(
                doc_logits, doc_labels, valid_idx, label_smoothing=self.config.label_smoothing
            )
            loss = base_loss
            n_total = int(valid_mask.sum().item())
            n_structural = 0
            if classify_structural:
                is_structural = torch.isin(doc_labels, structural_buf) & valid_mask
                n_structural = int(is_structural.sum().item())
                if n_structural > 0:
                    structural_idx = is_structural.nonzero(as_tuple=True)[0]
                    n_copy = n_total - n_structural
                    reweight = n_copy > 0 and w != 1.0
                    # Chunked like the base loss, so the fp32 peak is [chunk, V], never
                    # [n_structural, V] (multi-GB on a long words-mode document). At
                    # w == 1.0 the value is a metrics-only diagnostic, so skip the
                    # autograd graph entirely.
                    with contextlib.nullcontext() if reweight else torch.no_grad():
                        action_loss = chunked_cross_entropy(
                            doc_logits, doc_labels, structural_idx,
                            label_smoothing=self.config.label_smoothing,
                        )
                    action_losses.append((action_loss, n_structural))
                    if reweight:
                        alpha = (w - 1.0) * n_structural / n_total
                        loss = base_loss + alpha * action_loss
            base_losses.append(base_loss)
            losses.append(loss)
            counts.append((n_total, n_structural))

        loss_per_example = torch.stack(losses)
        metrics: dict = {
            "loss": loss_per_example.mean(),
            "loss_per_example": loss_per_example,
        }

        n_total = sum(n for n, _ in counts)
        n_structural = sum(n for _, n in counts)
        n_copy = n_total - n_structural
        if n_structural == 0 or n_copy == 0:
            return metrics

        with torch.no_grad():
            base_loss = sum(loss.detach() * n for loss, (n, _) in zip(base_losses, counts, strict=True)) / n_total
            action_loss = sum(loss.detach() * n for loss, n in action_losses) / n_structural
            copy_loss = (base_loss * n_total - action_loss * n_structural) / n_copy
        metrics["action_loss"] = action_loss
        metrics["copy_loss"] = copy_loss
        metrics["n_action_tokens"] = torch.tensor(n_structural, dtype=torch.long)
        return metrics

    def initial_state(self, source_ids):
        """Fresh pred-EDU decode state for a source token sequence (wraps the
        shared ShiftReduceDecodeState / SexpDecodingState). Mutable: the decode
        loops advance it in place via `apply` and copy it for beams via `clone`."""
        raise NotImplementedError

    def gold_initial_state(self, source_ids, gold_ranges):
        """Fresh gold-EDU forced decode state. `gold_ranges` is a list of
        `(start, end)` source-token spans; segmentation follows them, tree shape
        stays model-driven. Default: the pred state (SR carries the gold plan in
        the mask closure); sexp overrides to attach a GoldEduForcer to the state."""
        return self.initial_state(source_ids)

    def pred_mask(self, state, vocab_size: int):
        """Bool mask over the scoring vocab of the validity-legal ids at `state`
        (pred-EDU decoding). Both decode loops always constrain through this: a
        feasible tree is required to reconstruct, and unconstrained decoding is
        not a supported mode (see `decode.py`'s DecodeInvariantError)."""
        raise NotImplementedError

    def gold_mask_source(self, gold_ranges, source_len: int):
        """A `(state, vocab_size) -> BoolTensor` closure for gold-EDU forced
        decoding (segmentation from `gold_ranges`, structure model-driven)."""
        raise NotImplementedError

    def decode_id(self, idx: int) -> int:
        """Map an argmax index in the scoring space to a full-vocab id."""
        raise NotImplementedError

    def apply(self, state, full_id: int, source_ids):
        """Advance the state on one emitted id (mutates `state`). Returns
        `(next_input_or_None, kind)`; a None next_input stops decoding, and
        `kind == "illegal"` marks an off-grammar action the loop must not record."""
        raise NotImplementedError

    def candidate_ranges(self, state, finished: bool) -> list:
        """The predicted EDU source-position ranges for a decode that ended in
        `state` (tree meta for the dev eval). `finished` is whether it stopped on
        EOS (vs hitting the length cap / an unfinished beam), which governs
        whether a trailing partial EDU is committed."""
        raise NotImplementedError

    def stash_meta(self, tree, ranges, source_ids) -> None:
        """Attach the source-position meta the dev eval reads (`_pred_edu_source_ranges`,
        `_source_ids`) to a built tree, nulling the ranges when reconstruction fell
        back to a degenerate tree."""
        raise NotImplementedError

    def build_tree(self, action_ids, source_ids):
        raise NotImplementedError
