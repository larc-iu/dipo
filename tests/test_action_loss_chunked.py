"""The structural (action) CE is chunked and graph-free at w == 1.0 (`Serialization.loss_terms`).

The base CE was chunked (9a91222), but the structural-position CE still ran plain
`F.cross_entropy` over an `index_select`'d fp32 copy of the logits: [n_structural, V]
plus a softmax intermediate, with the autograd graph retained even at
`action_loss_weight == 1.0`, where the value is only used detached for metrics. On a
long words-mode document at a 262k vocab that is a multi-GB transient on top of the
chunked base loss. These tests pin the numerics against the plain reference in both
the reweighted and diagnostic cases, and pin that the diagnostic case runs its CE
under no_grad.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

import iudex.rst.parsers.gen.serializations.base as ser_base
from iudex.rst.parsers.gen.serializations.base import Serialization

V = 11
STRUCTURAL = [3, 5]
SMOOTHING = 0.1


class _FullHeadSer(Serialization):
    """Minimal full-head serialization: just the state loss_terms reads."""

    def __init__(self, action_loss_weight: float):
        super().__init__(SimpleNamespace(action_loss_weight=action_loss_weight, label_smoothing=SMOOTHING))
        self.head_vocab_size = V
        self.register_buffer(
            "_structural_full_ids_buf", torch.tensor(STRUCTURAL, dtype=torch.long), persistent=False
        )

    def uses_small_head(self) -> bool:
        return False


LABELS = torch.tensor([[-100, 3, 7, 5, -100, 2, 3, 8, 1]])  # 7 valid, 3 structural


def _logits():
    torch.manual_seed(0)
    return torch.randn(1, LABELS.shape[1], V, requires_grad=True)


def _reference(logits, w: float):
    """The pre-chunking computation, spelled out with plain F.cross_entropy."""
    doc_logits, doc_labels = logits[0], LABELS[0]
    valid = doc_labels != -100
    base = F.cross_entropy(doc_logits[valid].float(), doc_labels[valid], label_smoothing=SMOOTHING)
    structural = torch.isin(doc_labels, torch.tensor(STRUCTURAL)) & valid
    action = F.cross_entropy(doc_logits[structural].float(), doc_labels[structural], label_smoothing=SMOOTHING)
    n_total, n_structural = int(valid.sum()), int(structural.sum())
    loss = base if w == 1.0 else base + (w - 1.0) * n_structural / n_total * action
    return loss, action


def _grad_of(loss_fn):
    logits = _logits()
    loss_fn(logits).backward()
    return logits.grad.clone()


def test_reweighted_loss_matches_the_unchunked_reference():
    ser = _FullHeadSer(action_loss_weight=2.0)
    out = ser.loss_terms(_logits(), LABELS)
    ref_loss, ref_action = _reference(_logits(), 2.0)
    assert torch.allclose(out["loss"], ref_loss, atol=1e-6)
    assert torch.allclose(out["action_loss"], ref_action.detach(), atol=1e-6)
    got = _grad_of(lambda lg: ser.loss_terms(lg, LABELS)["loss"])
    want = _grad_of(lambda lg: _reference(lg, 2.0)[0])
    assert torch.allclose(got, want, atol=1e-6)


def test_w1_loss_and_grad_are_exactly_the_base_ce():
    ser = _FullHeadSer(action_loss_weight=1.0)
    out = ser.loss_terms(_logits(), LABELS)
    ref_loss, ref_action = _reference(_logits(), 1.0)
    assert torch.allclose(out["loss"], ref_loss, atol=1e-6)
    assert torch.allclose(out["action_loss"], ref_action.detach(), atol=1e-6)  # diagnostic still reported
    got = _grad_of(lambda lg: ser.loss_terms(lg, LABELS)["loss"])
    want = _grad_of(lambda lg: _reference(lg, 1.0)[0])
    assert torch.allclose(got, want, atol=1e-6)


def test_w1_action_ce_runs_under_no_grad(monkeypatch):
    """The memory half of the fix: at w == 1.0 the structural CE must build no
    autograd graph. Spy on the chunked-CE calls and check grad mode per call,
    telling base (7 rows) from action (3 rows) by the index count."""
    calls: list[tuple[int, bool]] = []
    real = ser_base.chunked_cross_entropy

    def spy(logits, labels, idx, **kw):
        calls.append((int(idx.numel()), torch.is_grad_enabled()))
        return real(logits, labels, idx, **kw)

    monkeypatch.setattr(ser_base, "chunked_cross_entropy", spy)

    _FullHeadSer(action_loss_weight=1.0).loss_terms(_logits(), LABELS)
    assert calls == [(7, True), (3, False)]

    calls.clear()
    _FullHeadSer(action_loss_weight=2.0).loss_terms(_logits(), LABELS)
    assert calls == [(7, True), (3, True)]
