"""A run trained before a knob was retired must still load.

`use_validity_constraints` and `constrain_content` were removed from GenConfig once
their only supported value became the only behavior. But a finished run's frozen
config.json still carries them, and tonga rejects unknown keys -- so removing the
fields made every pre-retirement run unloadable, `iudex gen eval` (which re-evaluates a
run from its frozen config) included. Measured at the time: 26 of 26 archived gen runs
failed to parse.

They are dropped, loudly, when they match today's behavior. A value that today's code
cannot reproduce raises instead: loading it would produce a config that misdescribes
the run it came from.
"""

import pytest

from iudex.rst.parsers.gen.configuration_gen import GenConfig


def _cfg(**over) -> dict:
    d = dict(
        backbone="decoder_only",
        serialization="sr",
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name="<unused>",
        relation_types=[("elaboration", "rst")],
    )
    d.update(over)
    return d


@pytest.mark.parametrize("key", ["use_validity_constraints", "constrain_content"])
def test_retired_key_at_its_only_behavior_is_dropped(key):
    cfg = GenConfig.from_dict(_cfg(**{key: True}))
    assert not hasattr(cfg, key)
    assert cfg.backbone == "decoder_only"


@pytest.mark.parametrize("key", ["use_validity_constraints", "constrain_content"])
def test_retired_key_at_an_unreproducible_value_raises(key):
    """Silently dropping `use_validity_constraints: false` would claim a run decoded
    constrained when it did not."""
    with pytest.raises(ValueError, match="retired key"):
        GenConfig.from_dict(_cfg(**{key: False}))


def test_a_real_archived_shape_loads():
    """Both keys together, as gen-era config.json files actually carry them."""
    cfg = GenConfig.from_dict(_cfg(use_validity_constraints=True, constrain_content=True, num_beams=6))
    assert cfg.num_beams == 6


def test_from_dict_does_not_mutate_the_caller_dict():
    """parse_config_dict pops keys, so a caller reusing its dict (e.g. re-parsing a
    checkpoint's embedded config) silently got an emptied one."""
    d = _cfg(use_validity_constraints=True)
    before = dict(d)
    GenConfig.from_dict(d)
    assert d == before


def test_unknown_keys_are_still_rejected():
    """Only RETIRED keys are tolerated; a typo must still fail loudly."""
    with pytest.raises(Exception, match="Extra parameters|totally_bogus"):
        GenConfig.from_dict(_cfg(totally_bogus_key=1))
