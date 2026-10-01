"""The WSD (warmup / hold-peak / re-warmup / decay-to-floor) LR schedule that the
curriculum gen parsers use. The whole point is that the anneal lands inside the
final full-document phase, not on the unvalidated subtree phases: LR holds peak
across the subtree phases, dips and re-warms into the full-doc phase, then decays
to a floor over the rest of it.
"""

import torch

from iudex.common.training import make_wsd_scheduler


def _trace(warmup, hold_end, decay_start, decay_end, min_lr_frac=0.1):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = make_wsd_scheduler(opt, warmup, hold_end, decay_start, decay_end, min_lr_frac)
    lrs = []
    for _ in range(decay_end + 1):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return lrs


def test_four_regions_have_the_intended_shape():
    lrs = _trace(warmup=200, hold_end=1560, decay_start=1710, decay_end=3120, min_lr_frac=0.1)
    assert lrs[0] == 0.0                     # warmup starts at 0
    assert abs(lrs[100] - 0.5) < 1e-6        # halfway through initial warmup
    assert abs(lrs[200] - 1.0) < 1e-6        # peak reached
    assert lrs[800] == 1.0                   # held at peak through the subtree phases
    assert lrs[1559] == 1.0                  # still peak at the last hold step
    assert lrs[1560] == 0.0                  # dips at the full-doc boundary (re-warmup start)
    assert abs(lrs[1635] - 0.5) < 1e-6       # halfway through the re-warmup
    assert abs(lrs[1710] - 1.0) < 1e-6       # re-warmup back to peak
    # linear decay to the floor over the rest of the full-doc phase
    assert 0.5 < lrs[2415] < 0.6             # ~mid-decay
    assert abs(lrs[-1] - 0.1) < 1e-3         # ends at the floor, not zero


def test_hold_region_never_decays():
    lrs = _trace(warmup=200, hold_end=1560, decay_start=1710, decay_end=3120)
    assert all(lr == 1.0 for lr in lrs[200:1560])  # peak throughout the subtree phases


def test_floor_is_respected_and_nonzero():
    lrs = _trace(warmup=50, hold_end=400, decay_start=450, decay_end=1000, min_lr_frac=0.25)
    assert min(lrs[450:]) >= 0.25 - 1e-9           # never dips below the floor after decay
    assert abs(lrs[-1] - 0.25) < 1e-3


def test_single_phase_degenerates_to_warmup_then_decay():
    # SimpleCurriculum: hold_end == decay_start == warmup -> no hold, no re-warmup.
    lrs = _trace(warmup=100, hold_end=100, decay_start=100, decay_end=1000, min_lr_frac=0.1)
    assert lrs[0] == 0.0
    assert abs(lrs[100] - 1.0) < 1e-6              # peak right after warmup
    assert lrs[100] > lrs[500] > lrs[900]          # monotone decay after the peak
    assert abs(lrs[-1] - 0.1) < 1e-3


def test_selection_side_fields_are_hash_excluded():
    from iudex.common.training import DEFAULT_HASH_EXCLUDE

    assert "patience_window" in DEFAULT_HASH_EXCLUDE
