"""Over-length is a hard crash at eval/predict, never a silent truncation or
best-effort repair. A truncated/repaired tree scores as if the model produced it
(e.g. a single-EDU fallback craters recall), corrupting the metric invisibly, so
the whole run must stop. The training path already raises (train_gen.py); this
pins the eval/predict mirror on both axes (source > max_input_length, and decode
exhausting max_output_length without completing). See
dipo/rst/parsers/gen/errors.py::OverLengthError.
"""

import os

import pytest

pytest.importorskip("transformers")

from dipo.rst.parsers.gen.configuration_gen import GenConfig
from dipo.rst.parsers.gen.errors import OverLengthError
from dipo.rst.parsers.gen.modeling_gen import GenParser

SMALL_CAUSAL = os.environ.get("DIPO_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
SMALL_SEQ2SEQ = os.environ.get("DIPO_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")


def _gen(backbone, *, serialization="sr", max_input_length=128, max_output_length=256) -> GenParser:
    model = SMALL_CAUSAL if backbone == "decoder_only" else SMALL_SEQ2SEQ
    d = dict(
        backbone=backbone,
        serialization=serialization,
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=model,
        relation_types=[("elaboration", "rst")],
        gradient_checkpointing=False,
        amp=False,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        min_edu_length=1,
        traversal_order="postorder",
        use_copy=True,
    )
    try:
        return GenParser(GenConfig.from_dict(d))
    except Exception as e:  # pragma: no cover - environment/model availability
        pytest.skip(f"Could not load {model}: {e!r}")


@pytest.mark.parametrize("backbone", ["decoder_only", "seq2seq"])
def test_input_overflow_raises(backbone):
    """Source longer than max_input_length crashes, at the tokenizer and end-to-end."""
    parser = _gen(backbone, max_input_length=64)
    long_text = " ".join(["word"] * 500)  # far more than 64 subwords
    with pytest.raises(OverLengthError):
        parser.backbone.tokenize_source(long_text)
    with pytest.raises(OverLengthError):
        parser.predict_from_text(long_text)


@pytest.mark.parametrize("backbone", ["decoder_only", "seq2seq"])
def test_output_overflow_raises_greedy(backbone):
    """A doc that fits the input but needs more than max_output_length decode steps
    crashes rather than emitting a best-effort-repaired (truncated) tree."""
    parser = _gen(backbone, max_input_length=256, max_output_length=4)
    text = "Alpha beta gamma delta epsilon zeta eta theta."
    with pytest.raises(OverLengthError):
        parser.predict_from_text(text, num_beams=1)


@pytest.mark.parametrize("backbone", ["decoder_only", "seq2seq"])
def test_output_overflow_raises_beam(backbone):
    parser = _gen(backbone, max_input_length=256, max_output_length=4)
    text = "Alpha beta gamma delta epsilon zeta eta theta."
    with pytest.raises(OverLengthError):
        parser.predict_from_text(text, num_beams=3)


def test_sexp_output_overflow_crashes_instead_of_empty_tree_fallback():
    """The wsj_1146 scenario: sexp used to truncate -> from_sexp fails -> silent
    single-EDU empty-tree fallback. That fallback must never fire from decode now;
    over-length is a crash before build_tree is reached."""
    parser = _gen("decoder_only", serialization="sexp", max_input_length=256, max_output_length=4)
    text = "Alpha beta gamma delta epsilon zeta eta theta."
    with pytest.raises(OverLengthError):
        parser.predict_from_text(text, num_beams=1)


def test_fitting_doc_does_not_raise():
    """Generous caps: a short doc decodes to completion without raising."""
    parser = _gen("seq2seq", max_input_length=256, max_output_length=512)
    tree = parser.predict_from_text("Alpha beta gamma. Delta epsilon zeta.", num_beams=1)
    assert tree is not None
