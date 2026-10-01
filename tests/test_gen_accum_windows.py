"""Gradient accumulation normalizes by the documents actually accumulated.

`train_gen`'s loop used to divide every batch loss by `cfg.grad_accum`. The epoch's
last window is short whenever `len(loader) % grad_accum != 0` (the loop steps early
on the final batch), so its gradient came out scaled by `window/grad_accum` -- while
`spe = ceil(len(loader) / grad_accum)` still counted it as a full step and the LR
scheduler advanced accordingly. Dividing by a configured constant also mis-weights
uneven batches: averaging per-batch means gives a 2-document batch the same pull as
a 3-document one.

The loop now backwards the weighted SUM, counts documents, and hands the real
`_normalize_window_grads` the count at the step. These tests drive that function
directly with a tiny linear model on CPU -- no backbone, no data -- comparing
against a directly-averaged reference. The surrounding `(loss * n_docs).backward()`
idiom is mirrored here rather than invoked: it lives inside `train()`, which needs a
model, data dirs, and a curriculum to reach.
"""

from __future__ import annotations

import torch

from iudex.rst.parsers.gen.train_gen import _normalize_window_grads

# Fixed per-document "losses" as a function of the model: loss_i = (w . x_i).
# With a linear model the gradient of a mean of losses is the mean of gradients,
# so a reference computed in one shot is exact.
DOCS = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 1.0],
        [1.0, 3.0],
        [-1.0, 2.0],
        [0.5, 0.5],
        [3.0, -1.0],
        [-2.0, 1.5],
    ]
)
WEIGHTS = torch.tensor([0.5, 1.5, 1.0, 0.75, 1.25, 1.0, 2.0, 0.0])


def _model() -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.tensor([0.3, -0.7]))


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    """allclose with a float32-realistic atol: summing in a different order than the
    one-shot reference lands components that cancel to 0.0 at ~1e-8, which the
    default atol=1e-8 would flag."""
    return torch.allclose(a, b, atol=1e-6)


def _doc_losses(w: torch.nn.Parameter, idx: list[int]) -> torch.Tensor:
    """Per-document losses for `idx`, mirroring `out["loss_per_example"]`."""
    return (DOCS[idx] * w).sum(dim=1)


def _weighted_mean(w: torch.nn.Parameter, idx: list[int]) -> torch.Tensor:
    """What `_weighted_document_loss` returns for this batch: the weighted mean."""
    return (_doc_losses(w, idx) * WEIGHTS[idx]).mean()


def _reference_grad(idx: list[int]) -> torch.Tensor:
    """Gradient of the weighted mean over ALL of `idx` computed in one shot."""
    w = _model()
    (_doc_losses(w, idx) * WEIGHTS[idx]).mean().backward()
    return w.grad.clone()


def _accumulate(batches: list[list[int]], *, normalize: str, grad_accum: int) -> torch.Tensor:
    """Run one accumulation window. `normalize='docs'` is the fixed loop (backward
    the sum, divide the accumulated grad by the document count); `normalize='const'`
    is the old loop (divide each batch's mean by cfg.grad_accum)."""
    w = _model()
    window_docs = 0
    for idx in batches:
        loss = _weighted_mean(w, idx)
        if normalize == "docs":
            (loss * len(idx)).backward()
            window_docs += len(idx)
        else:
            (loss / grad_accum).backward()
    if normalize == "docs":
        _normalize_window_grads([w], window_docs)
    return w.grad.clone()


def test_short_final_window_is_normalized_by_its_own_documents():
    # 5 single-document batches with grad_accum=8: the loop steps early on the last
    # batch of the epoch, so this window holds 5 of a possible 8.
    batches = [[i] for i in range(5)]
    got = _accumulate(batches, normalize="docs", grad_accum=8)
    assert _close(got, _reference_grad([0, 1, 2, 3, 4]))


def test_short_final_window_under_scaled_by_dividing_by_grad_accum():
    """The bug this fixes: the old normalization yields exactly 5/8 of the mean."""
    batches = [[i] for i in range(5)]
    old = _accumulate(batches, normalize="const", grad_accum=8)
    reference = _reference_grad([0, 1, 2, 3, 4])
    assert not _close(old, reference)
    assert _close(old, reference * (5 / 8))


def test_uneven_batches_weight_every_document_equally():
    """Batches of 3, 3, 2 documents: the accumulated gradient must be the mean over
    all 8 documents. Averaging per-batch means (what dividing by a batch count does)
    over-weights the 2-document batch."""
    batches = [[0, 1, 2], [3, 4, 5], [6, 7]]
    got = _accumulate(batches, normalize="docs", grad_accum=4)
    assert _close(got, _reference_grad(list(range(8))))

    mean_of_batch_means = torch.stack([_reference_grad(b) for b in batches]).mean(dim=0)
    assert not _close(mean_of_batch_means, _reference_grad(list(range(8))))


def test_full_window_matches_the_old_normalization():
    """A window with exactly grad_accum single-document batches is the case the old
    code got right; the fix must not disturb it."""
    batches = [[i] for i in range(8)]
    got = _accumulate(batches, normalize="docs", grad_accum=8)
    old = _accumulate(batches, normalize="const", grad_accum=8)
    assert _close(got, _reference_grad(list(range(8))))
    assert _close(got, old)


def test_window_counter_resets_between_windows():
    """window_docs must reset at each step, or window 2 inherits window 1's count and
    is normalized by too large a divisor."""
    w = _model()
    grads = []
    window_docs = 0
    for window in ([[0], [1], [2]], [[3], [4], [5]]):
        for idx in window:
            (_weighted_mean(w, idx) * len(idx)).backward()
            window_docs += len(idx)
        _normalize_window_grads([w], window_docs)
        grads.append(w.grad.clone())
        w.grad = None  # optimizer.zero_grad()
        window_docs = 0
    assert _close(grads[0], _reference_grad([0, 1, 2]))
    assert _close(grads[1], _reference_grad([3, 4, 5]))
