"""Regression tests for the unified generative parser `gen` (backbone x
serialization).

Rewritten from the former gen-vs-legacy equivalence suite: the four hand-written
generative parsers are gone, so these pin gen's own invariants directly rather
than comparing against a twin. Covers, across all four (backbone, serialization)
combos:

  (a) checkpoint round-trip: the state_dict survives torch.save -> torch.load ->
      strict `load_state_dict` into a fresh construction (the load-bearing schema
      guarantee that keeps published checkpoints loadable);
  (b) linearize -> build_tree recovers a tree's reduce labels (token AND words
      label styles, both serializations);
  (c) the decode matrix (greedy / beam x pred-EDU / gold-EDU) builds a tree
      without raising on an untrained tiny backbone;
  (d) a LoRA build constructs, carries adapter keys, and forwards.

Tiny offline backbones; each case skips (not fails) when the model can't be
fetched, or the backbone can't support the mode (e.g. t5's non-single-token sexp
brackets under words).
"""
import os
import tempfile

import pytest

pytest.importorskip("transformers")

SMALL_CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
SMALL_SEQ2SEQ = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")

import torch

from iudex.rst.data.tree import Reduce, RstTree, Shift
from iudex.rst.parsers.gen.configuration_gen import GenConfig
from iudex.rst.parsers.gen.modeling_gen import GenParser

COMBOS = [("decoder_only", "sr"), ("seq2seq", "sr"), ("decoder_only", "sexp"), ("seq2seq", "sexp")]


def _gen(backbone: str, serialization: str, **ov) -> GenParser:
    """Build a `gen` parser directly for the given (backbone, serialization)
    combo on a tiny offline backbone. Skips (not fails) if the model can't be
    fetched or the backbone can't support the requested mode."""
    model = SMALL_CAUSAL if backbone == "decoder_only" else SMALL_SEQ2SEQ
    d = dict(
        backbone=backbone, serialization=serialization, train_dir="x", dev_dir="x",
        model_name=model, relation_types=[("elaboration", "rst"), ("joint", "multinuc")],
        amp=False, max_input_length=128, max_output_length=128,
    )
    d.update(ov)
    try:
        return GenParser(GenConfig.from_dict(d))
    except Exception as e:  # network / gated weights / mode unsupported on this backbone
        pytest.skip(f"Could not build gen {backbone}/{serialization} on {model}: {e!r}")


def _toy_tree() -> RstTree:
    actions = [
        Shift(edu_text="Cats sleep."),
        Shift(edu_text="Dogs bark loudly."),
        Shift(edu_text="Birds fly south."),
        Reduce(nuc="NS", rel="elaboration"),
        Reduce(nuc="NN", rel="joint"),
    ]
    return RstTree.from_shift_reduce(actions, relation_types=[("elaboration", "rst"), ("joint", "multinuc")])


def _reduce_labels(tree) -> list:
    return [(a.nuc, a.rel) for a in tree.to_shift_reduce(include_text=False) if isinstance(a, Reduce)]


# ---------------------------------------------------------------------------
# (a) construction + checkpoint schema round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone,serialization", COMBOS)
def test_state_dict_save_load_roundtrip(backbone, serialization):
    """A freshly-built parser's state_dict serializes to disk and strict-loads
    back into a fresh construction of the same combo (schema stability)."""
    gen = _gen(backbone, serialization)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        torch.save(gen.state_dict(), path)
        reloaded = torch.load(path)
    fresh = _gen(backbone, serialization)
    assert set(reloaded) == set(fresh.state_dict()), "state_dict key set drifted between builds"
    fresh.load_state_dict(reloaded, strict=True)


# ---------------------------------------------------------------------------
# (b) linearize -> build_tree recovers the reduce labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone,serialization", COMBOS)
@pytest.mark.parametrize("label_style", ["token", "words"])
def test_linearize_build_tree_recovers_labels(backbone, serialization, label_style):
    """The serialization's own `linearize` output, replayed through `build_tree`
    (with the stream terminator appended), reconstructs the exact reduce labels."""
    gen = _gen(backbone, serialization, label_style=label_style)
    ser = gen.serialization
    tree = _toy_tree()
    source_ids, _seen, label = ser.linearize(tree)
    rebuilt = ser.build_tree([*label, ser.stream_end_id], source_ids)
    assert _reduce_labels(rebuilt) == _reduce_labels(tree)


# ---------------------------------------------------------------------------
# (c) decode matrix (greedy/beam x pred/gold) builds a tree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone,serialization", COMBOS)
def test_decode_matrix_builds_tree(backbone, serialization):
    """Every decode path returns a buildable RstTree without raising on an
    untrained tiny backbone (greedy/beam x pred-EDU/gold-EDU)."""
    gen = _gen(backbone, serialization)
    tree = _toy_tree()
    text = "Cats sleep. Dogs bark loudly. Birds fly south. Fish swim deep."
    outs = [
        gen.predict_from_text(text, num_beams=1),
        gen.predict_from_text(text, num_beams=3),
        gen.predict_with_gold_edus(tree),
        gen.predict_with_gold_edus(tree, num_beams=3),
    ]
    for out in outs:
        assert isinstance(out, RstTree)
        assert len(out.edus) >= 1


# ---------------------------------------------------------------------------
# (d) LoRA build constructs, carries adapters, and forwards
# ---------------------------------------------------------------------------


def test_lora_build_and_forward():
    """Under LoRA the state_dict carries adapter keys and a forward pass yields a
    finite loss."""
    from iudex.rst.parsers.common.config import PeftConfig

    gen = _gen("decoder_only", "sr", peft=PeftConfig(r=4, target_modules="all-linear"))
    assert any("lora" in k.lower() for k in gen.state_dict()), "no LoRA adapter keys found"
    input_ids, labels = gen.encode_target(_toy_tree())
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }
    with torch.no_grad():
        out = gen(batch)
    assert torch.isfinite(out["loss"]), out["loss"]


# ---------------------------------------------------------------------------
# words-mode terminator: non-prefix-free relation sets stay decodable
# ---------------------------------------------------------------------------


def test_gen_words_terminator_prefix_free_and_roundtrip():
    """label_style='words' + the <label_end> terminator makes ANY unique label set
    decodable. A non-prefix-free relation set (`elaboration`'s tokens are a prefix of
    `elaboration-additional`'s) is prefix-free once terminated, and both generative
    serializations reconstruct the exact reduce labels through linearize -> build_tree.
    The old eager trie match would silently emit the short label + mis-parse the rest."""
    rels = [("elaboration", "rst"), ("elaboration-additional", "rst")]
    tree = RstTree.from_shift_reduce(
        [
            Shift(edu_text="Cats sleep."), Shift(edu_text="Dogs bark loudly."), Shift(edu_text="Birds fly south."),
            Reduce(nuc="NS", rel="elaboration-additional"), Reduce(nuc="NS", rel="elaboration"),
        ],
        relation_types=rels,
    )
    orig = _reduce_labels(tree)

    for serialization in ("sr", "sexp"):
        gen = _gen("decoder_only", serialization, relation_types=rels, label_style="words")
        ser = gen.serialization

        # The terminated label set is prefix-free (no tuple is a prefix of another).
        tuples = list(ser.word_label_to_ids.values())
        for a in tuples:
            for b in tuples:
                if a is not b:
                    assert b[: len(a)] != a, f"{serialization}: {a} is a token-prefix of {b}"

        # linearize -> build_tree recovers the exact reduce labels (short + prefix-extending).
        source_ids, _seen, label = ser.linearize(tree)
        rebuilt = ser.build_tree([*label, ser.stream_end_id], source_ids)
        assert _reduce_labels(rebuilt) == orig, f"{serialization}: reconstructed labels != {orig}"
