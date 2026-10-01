"""Correctness gate for batched greedy decode in the unified generative parser `gen`.

Batched greedy (`greedy_decode_batch`, opted into by `batched_decode`) decodes B
different documents in one shared forward per step instead of one batch-1 forward per
document. Greedy argmax per document is independent of what shares the batch, so the
batched path MUST return the byte-identical tree the per-document path returns --
otherwise a batched eval would silently mix two decode regimes across the paper's
table cells.

These tests assert EXACT equality (the action-id sequence, the reduce labels, AND the
stashed per-EDU source ranges), not "close enough", across:

  * both serializations (`sr`, `sexp`) on both backbones (`decoder_only`, `seq2seq`);
  * ragged batches -- documents of deliberately different lengths, so left/right
    padding + per-row position ids are genuinely exercised and a short doc shares a
    batch with a much longer one;
  * both conditions, e2e (`predict_batch`) and gold-EDU-forced
    (`predict_batch_with_gold_edus`), since the gold pass is the second decode the
    optimization also batches;
  * batch composition invariance -- a document decodes to the same tree whether it
    rides with short or long neighbors.

Tiny random-weights backbones on CPU: correctness of the batching is weight-agnostic
(argmax equality is a property of the padding/masking, not of what the model learned).
"""

import os

import pytest

pytest.importorskip("transformers")

import torch

from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.common.seqgen import reconstruct_text
from iudex.rst.parsers.gen.configuration_gen import GenConfig
from iudex.rst.parsers.gen.modeling_gen import GenParser

SMALL_CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
SMALL_SEQ2SEQ = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")

COMBOS = [("decoder_only", "sr"), ("seq2seq", "sr"), ("decoder_only", "sexp"), ("seq2seq", "sexp")]
RELATION_TYPES = [("elaboration", "rst"), ("joint", "multinuc"), ("contrast", "rst")]


def _gen(backbone: str, serialization: str, **ov) -> GenParser:
    model = SMALL_CAUSAL if backbone == "decoder_only" else SMALL_SEQ2SEQ
    d = dict(
        backbone=backbone, serialization=serialization, train_dir="x", dev_dir="x",
        model_name=model, relation_types=RELATION_TYPES,
        amp=False, max_input_length=512, max_output_length=512, batched_decode=True,
    )
    d.update(ov)
    try:
        parser = GenParser(GenConfig.from_dict(d))
    except Exception as e:  # network / gated weights / mode unsupported on this backbone
        pytest.skip(f"Could not build gen {backbone}/{serialization} on {model}: {e!r}")
    torch.manual_seed(0)  # deterministic tiny random head so argmax ties are stable across runs
    parser.eval()
    return parser


def _tree(edu_texts, reduces) -> RstTree:
    """Build an RstTree from EDU texts and a postorder list of (nuc, rel) reduces."""
    actions = [Shift(edu_text=t) for t in edu_texts]
    for nuc, rel in reduces:
        actions.append(Reduce(nuc=nuc, rel=rel))
    return RstTree.from_shift_reduce(actions, relation_types=RELATION_TYPES)


def _short_tree() -> RstTree:
    return _tree(
        ["Cats sleep all day.", "Dogs bark loudly."],
        [("NS", "elaboration")],
    )


def _medium_tree() -> RstTree:
    return _tree(
        ["The market opened higher.", "Investors were optimistic.", "Tech stocks led the gains."],
        [("NS", "elaboration"), ("NN", "joint")],
    )


def _long_tree() -> RstTree:
    """A deliberately longer document so a batch spans very different lengths (its
    prefix + output are several times the short doc's -- the padding stress case)."""
    edus = [
        "The committee convened early on Monday morning.",
        "Several members raised procedural objections.",
        "The chair overruled them after a brief recess.",
        "Debate then turned to the budget proposal.",
        "Costs had risen sharply over the prior quarter.",
        "But projected revenue remained essentially flat.",
        "A subcommittee was tasked with the reconciliation.",
        "It would report back before the next session.",
    ]
    reduces = [
        ("NS", "elaboration"),
        ("NN", "contrast"),
        ("NS", "elaboration"),
        ("NS", "elaboration"),
        ("NN", "joint"),
        ("NS", "elaboration"),
        ("NN", "joint"),
    ]
    return _tree(edus, reduces)


def _actions(tree: RstTree) -> list:
    return [(type(a).__name__, getattr(a, "nuc", None), getattr(a, "rel", None), getattr(a, "edu_text", None))
            for a in tree.to_shift_reduce(include_text=True)]


def _ranges(tree: RstTree):
    return getattr(tree, "_pred_edu_source_ranges", None)


def _assert_identical(a: RstTree, b: RstTree, ctx: str) -> None:
    assert _actions(a) == _actions(b), f"{ctx}: action sequence differs\n serial={_actions(a)}\n batched={_actions(b)}"
    assert _ranges(a) == _ranges(b), f"{ctx}: stashed EDU source ranges differ\n serial={_ranges(a)}\n batched={_ranges(b)}"


RAGGED = [_short_tree(), _long_tree(), _medium_tree(), _short_tree()]


@pytest.mark.parametrize("backbone,serialization", COMBOS)
def test_batched_greedy_matches_serial_e2e(backbone, serialization):
    """predict_batch (batched) == predict_from_text (per-document) for every doc in a
    ragged batch, exactly."""
    parser = _gen(backbone, serialization)
    texts = [reconstruct_text(t) for t in RAGGED]

    serial = [parser.predict_from_text(txt, num_beams=1) for txt in texts]
    batched = parser.predict_batch(RAGGED, num_beams=1)

    assert len(serial) == len(batched) == len(RAGGED)
    for k, (s, b) in enumerate(zip(serial, batched, strict=True)):
        _assert_identical(s, b, f"{backbone}/{serialization} e2e doc {k}")


@pytest.mark.parametrize("backbone,serialization", COMBOS)
def test_batched_greedy_matches_serial_gold_edu(backbone, serialization):
    """predict_batch_with_gold_edus (batched) == predict_with_gold_edus (per-document),
    exactly -- the forced-decode pass the optimization also batches."""
    parser = _gen(backbone, serialization)

    serial = [parser.predict_with_gold_edus(t, num_beams=1) for t in RAGGED]
    batched = parser.predict_batch_with_gold_edus(RAGGED, num_beams=1)

    assert len(serial) == len(batched) == len(RAGGED)
    for k, (s, b) in enumerate(zip(serial, batched, strict=True)):
        _assert_identical(s, b, f"{backbone}/{serialization} gold-EDU doc {k}")


@pytest.mark.parametrize("backbone,serialization", COMBOS)
def test_batch_composition_invariance(backbone, serialization):
    """A document decodes to the same tree regardless of its batch neighbors: the
    medium doc alone, batched with a long doc, and batched with a short doc all agree.
    This is the property that lets length-bucketing reorder documents freely."""
    parser = _gen(backbone, serialization)
    med = _medium_tree()

    solo = parser.predict_batch([med], num_beams=1)[0]
    with_long = parser.predict_batch([_long_tree(), med], num_beams=1)[1]
    with_short = parser.predict_batch([_short_tree(), med, _short_tree()], num_beams=1)[1]

    _assert_identical(solo, with_long, f"{backbone}/{serialization} composition (long neighbor)")
    _assert_identical(solo, with_short, f"{backbone}/{serialization} composition (short neighbors)")


@pytest.mark.parametrize("backbone,serialization", COMBOS)
def test_singleton_and_empty_batch(backbone, serialization):
    """Degenerate batch shapes: a one-document batch equals the solo decode, and an
    empty batch returns nothing (no forward)."""
    parser = _gen(backbone, serialization)
    assert parser.predict_batch([], num_beams=1) == []
    one = parser.predict_batch([_medium_tree()], num_beams=1)
    assert len(one) == 1
