"""Regression: an undertrained generative SR model can emit a *valid* action
sequence that builds a pathologically deep tree (e.g. a long doc decoded as
hundreds of single-token EDUs, then a linear reduce chain). `from_shift_reduce`
-> `binarize_tree` -> `compute_edu_yields` recurses once per node and blows
Python's recursion limit. Real trees are shallow (GUM maxes ~235 EDUs), so this
only fires on untrusted model output, where degrading to a single-EDU tree is
correct (mirrors the sexp parsers' `_tree_from_emitted`).

Caught live by the full-GUM small-model regime: seq2seq_sr crashed at the
epoch-1 dev eval before this guard existed.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("transformers")

from dipo.rst.parsers.gen.configuration_gen import GenConfig
from dipo.rst.parsers.gen.modeling_gen import GenParser

T5 = os.environ.get("DIPO_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")
CAUSAL = os.environ.get("DIPO_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")


def _build(backbone: str, model: str) -> GenParser:
    d = {
        "backbone": backbone,
        "serialization": "sr",
        "train_dir": "<unused>",
        "dev_dir": "<unused>",
        "model_name": model,
        "relation_types": [["elaboration", "rst"], ["joint", "multinuc"]],
        "amp": False,
        "max_input_length": 256,
        "max_output_length": 512,
        "num_beams": 1,
    }
    try:
        return GenParser(GenConfig.from_dict(d))
    except Exception as e:  # network / gated / arch mismatch on this host
        pytest.skip(f"Could not construct gen {backbone}/sr with {model}: {e!r}")


@pytest.mark.parametrize("backbone,model", [("seq2seq", T5), ("decoder_only", CAUSAL)])
def test_deep_action_sequence_degrades_not_crashes(backbone, model):
    parser = _build(backbone, model)
    ser = parser.serialization
    # ~1300 single-token EDUs then a linear reduce chain => ~1300-deep tree,
    # past CPython's default 1000-frame limit.
    src = parser.tokenizer("word " * 1300, add_special_tokens=False).input_ids[:1300]
    if len(src) < 1200:
        pytest.skip("tokenizer produced too few ids to force deep recursion")
    # gen's SR build_tree consumes source subwords via <copy> sentinels (source_ids
    # passed separately), so a single-token EDU is COPY + SHIFT per source position.
    action_ids: list[int] = []
    for _s in src:
        action_ids += [ser.copy_token_id, ser.shift_token_id]
    action_ids += [sorted(ser.reduce_token_ids)[0]] * (len(src) - 1)
    tree = ser.build_tree(action_ids, src)  # must not raise RecursionError
    assert len(tree.edus) >= 1
