"""Gold-EDU forcing contract for the unified parser's `sexp` serialization
(both backbones).

Sexp contract: force EXACTLY `n_edus_target` leaves via a `GoldEduForcer`
(budget-range planner) so even untrained backbones terminate cleanly. Inside a
leaf, force content / close at the gold end. Outside a leaf, force only what the
per-frame EDU-budget range entails; the leaf-vs-internal choice (the tree shape)
stays the model's wherever both fit, as do the label slots.

gen routes both sexp backbones through one gold-forced decode path
(`GenParser._predict_one_gold_edu` + `SexpSerialization.gold_*`), so the former
per-parser parity checks collapse to pinning that single path's contract.
"""

import os
from typing import List

import pytest

pytest.importorskip("transformers")

from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.gen.configuration_gen import GenConfig
from iudex.rst.parsers.gen.modeling_gen import GenParser

SMALL_SEQ2SEQ = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")
SMALL_CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")


def _toy_tree() -> RstTree:
    actions = [
        Shift(edu_text="Cats sleep."),
        Shift(edu_text="Dogs bark."),
        Reduce(nuc="NS", rel="elaboration"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst")])


def _gen(backbone: str) -> GenParser:
    model = SMALL_CAUSAL if backbone == "decoder_only" else SMALL_SEQ2SEQ
    d = dict(
        backbone=backbone,
        serialization="sexp",
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=model,
        relation_types=[("elaboration", "rst")],
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=256,
        min_edu_length=1,
        traversal_order="postorder",
        use_copy=True,
    )
    try:
        return GenParser(GenConfig.from_dict(d))
    except Exception as e:
        pytest.skip(f"Could not load {model}: {e!r}")


def test_gold_edu_docstring_mentions_contract():
    # Migrated from the two per-parser docstring checks: gen has one gold-forced
    # entry point, so the boundary/structure contract is documented there.
    doc = (GenParser.predict_with_gold_edus.__doc__ or "").lower()
    assert "boundaries" in doc or "gold" in doc
    assert "structure" in doc or "leaf" in doc


@pytest.mark.parametrize("backbone", ["seq2seq", "decoder_only"])
def test_gold_edu_runs_and_ranges_are_monotone(backbone):
    """Both sexp backbones run gold-EDU forced decode on a toy tree without
    exceptions, and any emitted ranges are monotone non-decreasing in start
    position (the shared contract; strict gold alignment requires a trained
    model)."""
    parser = _gen(backbone)
    tree = _toy_tree()
    pred = parser.predict_with_gold_edus(tree)
    pred_ranges: List[tuple] = getattr(pred, "_pred_edu_source_ranges", [])
    assert isinstance(pred_ranges, list)
    starts = [s for s, _ in pred_ranges]
    assert starts == sorted(starts), f"pred ranges not monotone: {pred_ranges}"


def test_sexp_gold_edu_strategy_uses_forcer_and_clamped_ranges():
    """The unified sexp gold path drives off the shared `GoldEduForcer` planner and
    clamped gold ranges (no per-parser ad hoc loop, no LABEL sentinel)."""
    import inspect

    from iudex.rst.parsers.gen.serializations.sexp import SexpSerialization

    gold_setup = inspect.getsource(SexpSerialization.gold_initial_state)
    assert "GoldEduForcer" in gold_setup, "sexp gold_initial_state doesn't use GoldEduForcer"
    assert '"LABEL"' not in inspect.getsource(SexpSerialization), "sexp still uses a LABEL sentinel"
    clamp_src = inspect.getsource(GenParser._gold_edu_setup)
    assert "clamped" in clamp_src, "gold-EDU setup lost the clamped-ranges drive"
