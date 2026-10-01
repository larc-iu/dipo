"""Checkpoint hardening tests for iudex.common.training: atomic saves leave no
.tmp behind, try_resume tolerates a truncated last.pt (falls back to
best_model.pt or fresh, with a warning, never a crash), SIGTERM triggers the
soft-abort flag, and trainable-only checkpoints round-trip (frozen weights from
init, trained weights from the checkpoint, unexpected keys raise).

Tiny modules, CPU, no network.
"""

from __future__ import annotations

import os
import signal

import pytest
import torch
import torch.nn as nn

import iudex.common.training as training
from iudex.common.training import (
    install_abort_handler,
    load_model_state,
    save_checkpoint,
    try_resume,
)


class _Tiny(nn.Module):
    """Linear pair with one frozen submodule, standing in for LoRA-over-base."""

    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(4, 4)
        self.frozen = nn.Linear(4, 4)
        for p in self.frozen.parameters():
            p.requires_grad = False


def _save(path: str, model: nn.Module, *, trainable_only: bool = False, config_hash: str = "h") -> None:
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)
    save_checkpoint(
        path,
        model,
        opt,
        sched,
        trainable_only=trainable_only,
        config_hash=config_hash,
        global_step=1,
        epoch=1,
        best_val=0.5,
    )


def _capture_warns(monkeypatch) -> list[str]:
    msgs: list[str] = []
    monkeypatch.setattr(training, "warn", lambda m: msgs.append(m))
    return msgs


def test_atomic_save_leaves_no_tmp(tmp_path):
    path = str(tmp_path / "last.pt")
    _save(path, _Tiny())
    assert os.path.exists(path)
    assert os.path.exists(str(tmp_path / "last.json"))
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_try_resume_happy_path(tmp_path):
    path = str(tmp_path / "last.pt")
    _save(path, _Tiny(), config_hash="h")
    ckpt = try_resume(path, expected_hash="h")
    assert ckpt is not None
    assert ckpt["global_step"] == 1


@pytest.mark.parametrize("keep_fraction", [0.01, 0.5])
def test_try_resume_truncated_falls_back_fresh(tmp_path, monkeypatch, keep_fraction):
    """A last.pt truncated by a mid-write kill warns and starts fresh. Both
    truncation points matter: a near-empty file fails on the zip magic, while
    a half-written one has the magic but a torn central directory (torch
    raises OSError(EINVAL) there, not BadZipFile)."""
    msgs = _capture_warns(monkeypatch)
    path = str(tmp_path / "last.pt")
    _save(path, _Tiny())
    with open(path, "r+b") as f:
        f.truncate(max(1, int(os.path.getsize(path) * keep_fraction)))
    assert try_resume(path, expected_hash="h") is None
    assert any("last.pt" in m for m in msgs)
    assert any("fresh" in m.lower() for m in msgs)


def test_try_resume_garbage_falls_back_fresh(tmp_path, monkeypatch):
    msgs = _capture_warns(monkeypatch)
    path = str(tmp_path / "last.pt")
    with open(path, "wb") as f:
        f.write(b"not a checkpoint")
    assert try_resume(path, expected_hash="h") is None
    assert any("last.pt" in m for m in msgs)


def test_try_resume_corrupt_prefers_best_model(tmp_path, monkeypatch):
    """With a corrupt last.pt, a loadable sibling best_model.pt wins over fresh."""
    msgs = _capture_warns(monkeypatch)
    _save(str(tmp_path / "best_model.pt"), _Tiny(), config_hash="h")
    last = str(tmp_path / "last.pt")
    with open(last, "wb") as f:
        f.write(b"garbage")
    ckpt = try_resume(last, expected_hash="h")
    assert ckpt is not None
    assert ckpt["global_step"] == 1
    assert any("best_model.pt" in m for m in msgs)


def test_sigterm_sets_abort_flag():
    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    try:
        flag = install_abort_handler()
        assert flag.value is False
        os.kill(os.getpid(), signal.SIGTERM)
        assert flag.value is True
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def test_trainable_only_roundtrip(tmp_path):
    torch.manual_seed(0)
    src = _Tiny()
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)  # move away from any plausible fresh init
    path = str(tmp_path / "last.pt")
    _save(path, src, trainable_only=True)

    ckpt = torch.load(path, weights_only=False)
    assert ckpt["trainable_only"] is True
    assert set(ckpt["model_state_dict"]) == {"trainable.weight", "trainable.bias"}

    torch.manual_seed(1)
    fresh = _Tiny()
    frozen_init = {k: v.clone() for k, v in fresh.frozen.state_dict().items()}
    load_model_state(fresh, ckpt)
    assert torch.equal(fresh.trainable.weight, src.trainable.weight)
    assert torch.equal(fresh.trainable.bias, src.trainable.bias)
    for k, v in fresh.frozen.state_dict().items():
        assert torch.equal(v, frozen_init[k])  # from init, not the checkpoint
        assert not torch.equal(v, src.frozen.state_dict()[k])


def test_trainable_only_unexpected_key_raises():
    ckpt = {
        "trainable_only": True,
        "model_state_dict": {
            "trainable.weight": torch.zeros(4, 4),
            "bogus.weight": torch.zeros(2),
        },
    }
    with pytest.raises(RuntimeError, match="bogus.weight"):
        load_model_state(_Tiny(), ckpt)


def test_full_checkpoint_loads_strict(tmp_path):
    torch.manual_seed(0)
    src = _Tiny()
    path = str(tmp_path / "last.pt")
    _save(path, src)
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["trainable_only"] is False

    torch.manual_seed(1)
    fresh = _Tiny()
    load_model_state(fresh, ckpt)
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, src.state_dict()[k])

    # Strictness is unchanged for full checkpoints: a missing key still raises.
    del ckpt["model_state_dict"]["frozen.weight"]
    with pytest.raises(RuntimeError):
        load_model_state(_Tiny(), ckpt)
