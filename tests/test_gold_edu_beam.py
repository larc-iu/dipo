"""Tests for the beam-search gold-EDU forced decode across all four (backbone,
serialization) combos of the unified generative parser `gen`.

The gold-EDU condition used to decode greedily while the e2e condition decoded
with beam-6, an inconsistent protocol across the two eval columns. gen now
reaches a beam-searched gold-forced decode via
`predict_with_gold_edus(tree, num_beams=K)`. These tests pin the two invariants
that make the beam path safe to swap in for the greedy one at final eval:

  * `num_beams=1` is byte-identical to the greedy forced decode (same tree,
    same stashed source ranges). log_softmax is monotonic and the legal set is
    identical, so beam-1 argmax == greedy argmax at every step.
  * `num_beams=K>1` still emits EXACTLY the gold EDU count (segmentation is
    forced, only tree shape / labels are model-chosen), so the gold-EDU
    Parseval keeps seeing aligned span counts.

Untrained tiny backbones are fine here: the forced mask / `GoldEduForcer`
guarantee termination and the gold segmentation regardless of weights. Real
beam quality is a cluster-eval concern, not a unit-test one.
"""

import os

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


def _multi_edu_tree() -> RstTree:
    """A 5-EDU document, enough gold boundaries and free (leaf-vs-internal /
    reduce-placement) decisions for beams to actually diverge and for the KV
    cache to need reordering across several steps."""
    actions = [
        Shift(edu_text="The market opened higher today."),
        Shift(edu_text="Investors were optimistic."),
        Reduce(nuc="NS", rel="elaboration"),
        Shift(edu_text="Tech stocks led the gains."),
        Reduce(nuc="NS", rel="elaboration"),
        Shift(edu_text="But bonds slipped."),
        Shift(edu_text="Yields rose sharply."),
        Reduce(nuc="NS", rel="elaboration"),
        Reduce(nuc="NN", rel="contrast"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst"), ("contrast", "rst")])


# ---------------------------------------------------------------------------
# Parser builders (one tiny backbone load per combo, cached module-wide). Keyed
# by the legacy parser names, now mapped to gen (backbone, serialization) combos.
# ---------------------------------------------------------------------------

_COMBOS = {
    "decoder_only_sr": ("decoder_only", "sr"),
    "seq2seq_sr": ("seq2seq", "sr"),
    "decoder_only_sexp": ("decoder_only", "sexp"),
    "seq2seq_sexp": ("seq2seq", "sexp"),
}


def _build(name):
    backbone, serialization = _COMBOS[name]
    model = SMALL_CAUSAL if backbone == "decoder_only" else SMALL_SEQ2SEQ
    d = dict(
        backbone=backbone,
        serialization=serialization,
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=model,
        relation_types=[("elaboration", "rst"), ("contrast", "rst")],
        gradient_checkpointing=False,
        amp=False,
        max_input_length=128,
        max_output_length=256,
        min_edu_length=1,
    )
    if serialization == "sexp":
        d.update(traversal_order="postorder", use_copy=True)
    try:
        return GenParser(GenConfig.from_dict(d))
    except Exception as e:
        pytest.skip(f"Could not load {model}: {e!r}")


_CACHE: dict = {}


def _get_parser(name):
    if name not in _CACHE:
        _CACHE[name] = _build(name)
    return _CACHE[name]


ALL_PARSERS = list(_COMBOS)
TREES = {"toy": _toy_tree, "multi": _multi_edu_tree}


@pytest.mark.parametrize("name", ALL_PARSERS)
@pytest.mark.parametrize("tree_key", list(TREES))
def test_beam1_equals_greedy(name, tree_key):
    """num_beams=1 reproduces the greedy forced decode exactly: same tree
    serialization and same stashed per-EDU source ranges."""
    parser = _get_parser(name)
    tree = TREES[tree_key]()
    greedy = parser.predict_with_gold_edus(tree)
    beam1 = parser.predict_with_gold_edus(tree, num_beams=1)
    assert beam1.to_rs4_string() == greedy.to_rs4_string(), f"{name}/{tree_key}: beam-1 != greedy tree"
    assert getattr(beam1, "_pred_edu_source_ranges", None) == getattr(greedy, "_pred_edu_source_ranges", None), (
        f"{name}/{tree_key}: beam-1 ranges != greedy ranges"
    )


@pytest.mark.parametrize("name", ALL_PARSERS)
@pytest.mark.parametrize("tree_key", list(TREES))
@pytest.mark.parametrize("K", [2, 4])
def test_beamK_emits_exact_gold_edu_count(name, tree_key, K):
    """Segmentation is forced, so the selected beam must emit exactly the gold
    EDU count (this is the precondition `_evaluate_gold_edu` needs to avoid
    skipping the tree for EDU-count drift)."""
    parser = _get_parser(name)
    tree = TREES[tree_key]()
    pred = parser.predict_with_gold_edus(tree, num_beams=K)
    assert len(pred.edus) == len(tree.edus), (
        f"{name}/{tree_key}/K={K}: {len(pred.edus)} pred EDUs != {len(tree.edus)} gold"
    )


@pytest.mark.parametrize("name", ALL_PARSERS)
def test_beamK_ranges_are_monotone(name):
    """Stashed pred source ranges stay monotone non-decreasing in start under
    beam decode (the shared gold-EDU contract)."""
    parser = _get_parser(name)
    tree = _multi_edu_tree()
    pred = parser.predict_with_gold_edus(tree, num_beams=4)
    ranges = getattr(pred, "_pred_edu_source_ranges", [])
    starts = [s for s, _ in ranges]
    assert starts == sorted(starts), f"{name}: pred ranges not monotone: {ranges}"


@pytest.mark.parametrize("name", ALL_PARSERS)
def test_evaluate_gold_edu_beam_wiring(name):
    """`_evaluate_gold_edu(..., num_beams=K)` threads the beam width through to
    `predict_with_gold_edus` and still returns the four finite gold_edu_* keys.
    Guards the shared-eval wiring that makes final eval beam-search both columns."""
    from iudex.rst.parsers.common.generative_eval import _evaluate_gold_edu

    parser = _get_parser(name)
    pairs = [("toy.rs4", _toy_tree()), ("multi.rs4", _multi_edu_tree())]
    metrics = _evaluate_gold_edu(parser, pairs, num_beams=4)
    expected = {"gold_edu_span_f1", "gold_edu_nuc_f1", "gold_edu_rel_f1", "gold_edu_full_f1"}
    assert expected.issubset(metrics.keys())
    for k in expected:
        v = metrics[k]
        assert v == v  # not NaN
        assert 0.0 <= v <= 1.0


def test_evaluate_on_dev_beam_gold_edu_matches_e2e_width():
    """End-to-end: evaluate_on_dev with eval_gold_edu=True and num_beams>1
    produces gold_edu_* keys (the gold path took the beam width, not greedy).
    One parser is enough to cover the shared orchestration."""
    from iudex.rst.parsers.common.generative_eval import evaluate_on_dev

    parser = _get_parser("seq2seq_sr")
    pairs = [("toy.rs4", _toy_tree())]
    metrics = evaluate_on_dev(parser, pairs, num_beams=3, eval_gold_edu=True)
    assert "gold_edu_full_f1" in metrics
    assert "e2e_full_f1" in metrics or any(k.startswith("e2e") for k in metrics)
