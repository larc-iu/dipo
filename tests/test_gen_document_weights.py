from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from dipo.rst.parsers.gen.serializations.base import Serialization
from dipo.rst.parsers.gen.train_gen import _weighted_document_loss


class _FullVocabSerialization(Serialization):
    def uses_small_head(self) -> bool:
        return False


def test_document_weights_apply_to_matching_document_loss():
    ser = _FullVocabSerialization(SimpleNamespace(label_smoothing=0.0, action_loss_weight=1.0))
    ser.head_vocab_size = 2
    ser.register_buffer("_structural_full_ids_buf", torch.empty(0, dtype=torch.long))

    logits = torch.tensor([[[0.0, 0.0]], [[4.0, 0.0]]])
    labels = torch.tensor([[0], [1]])
    out = ser.loss_terms(logits, labels)

    expected_per_doc = torch.stack(
        [F.cross_entropy(logits[0], labels[0]), F.cross_entropy(logits[1], labels[1])]
    )
    weights = torch.tensor([0.5, 1.5])  # mean 1: the old batch-scalar weighting was a no-op
    expected_weighted = (expected_per_doc * weights).mean()

    assert torch.allclose(out["loss_per_example"], expected_per_doc)
    assert torch.allclose(out["loss"], expected_per_doc.mean())
    assert torch.allclose(_weighted_document_loss(out["loss_per_example"], weights), expected_weighted)
    assert not torch.allclose(expected_weighted, out["loss"] * weights.mean())


def test_document_weight_shapes_must_match():
    with pytest.raises(ValueError, match="matching 1-D"):
        _weighted_document_loss(torch.ones(2), torch.ones(1))
