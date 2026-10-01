"""`build_tree` distinguishes a malformed stream from a merely bad one.

MALFORMED (`from_sexp` rejects the string) is unreachable: the PDA admits only
well-formed sexps, content is pinned to the source cursor, and `(`/`)` in it are
escaped. Reaching it means the constraints are broken, so it raises rather than
fabricating a single-EDU tree that would score as if the model produced it.

DEEP is different and does happen: a legal action sequence can build a tree past
CPython's recursion limit (see test_sr_deep_tree_degrades.py). The mask and the
automaton agree there -- it is a bad parse, not a bug -- so it still degrades to
a single-EDU tree, flagged `_from_sexp_failed`, with `stash_meta` nulling the
action-derived ranges that the one-EDU tree would contradict.
"""

import os

import pytest

pytest.importorskip("transformers")

from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.gen.configuration_gen import GenConfig
from iudex.rst.parsers.gen.errors import DecodeInvariantError
from iudex.rst.parsers.gen.modeling_gen import GenParser

SMALL_CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
SMALL_SEQ2SEQ = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")


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


@pytest.mark.parametrize("backbone", ["decoder_only", "seq2seq"])
def test_sexp_build_tree_raises_on_malformed_output(backbone):
    """A malformed (empty) action stream -> empty sexp string, which
    RstTree.from_sexp rejects. That must surface, not become a single-EDU tree."""
    ser = _gen(backbone).serialization
    with pytest.raises(DecodeInvariantError, match="from_sexp rejected"):
        ser.build_tree([], [0, 1, 2])


def _toy_tree():
    return RstTree.from_shift_reduce(
        [Shift(edu_text="a"), Shift(edu_text="b"), Reduce(nuc="NS", rel="elaboration")],
        relation_types=[("elaboration", "rst")],
    )


@pytest.mark.parametrize("backbone", ["decoder_only", "seq2seq"])
def test_sexp_stash_meta_keeps_ranges_for_a_real_tree(backbone):
    ser = _gen(backbone).serialization
    tree = _toy_tree()
    ser.stash_meta(tree, [(0, 1), (1, 2)], [0, 1, 2])
    assert tree._pred_edu_source_ranges == [(0, 1), (1, 2)]


@pytest.mark.parametrize("backbone", ["decoder_only", "seq2seq"])
def test_sexp_stash_meta_nulls_ranges_for_a_degraded_tree(backbone):
    """A degraded (pathologically-deep) tree has one EDU, so the action-tracked
    ranges disagree with it and must be nulled."""
    ser = _gen(backbone).serialization
    tree = _toy_tree()
    tree._from_sexp_failed = True
    ser.stash_meta(tree, [(0, 1), (1, 2)], [0, 1, 2])
    assert tree._pred_edu_source_ranges == []


def test_use_copy_false_is_constructible():
    """`use_copy=False` is the no-COPY mode (Hu and Wan 2023 mirror). Both sexp
    backbones' configs should accept it without raising."""
    for backbone in ("decoder_only", "seq2seq"):
        GenConfig.from_dict(
            dict(
                backbone=backbone,
                serialization="sexp",
                train_dir="<unused>",
                dev_dir="<unused>",
                relation_types=[("elaboration", "rst")],
                use_copy=False,
            )
        )
