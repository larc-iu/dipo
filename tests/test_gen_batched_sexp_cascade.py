"""Why batched greedy sexp decode diverges MORE than SR on real bf16 weights.

Batched greedy is bit-identical to the per-document path when the backbone forward
is deterministic (`test_gen_batched_greedy_equivalence.py`, CPU/fp32). On a real GPU
in bf16 the batched forward reduces its matmuls in a different order than the batch-1
forward, so an occasional argmax TIE flips. For `sr` that flip perturbs at most an
action or two; for `sexp` an early flip at a leaf's CONTENT-vs-CLOSE decision can
cascade to a much shorter but STILL VALID tree (a leaf that keeps eating swallows the
rest of the source, collapsing segmentation).

This test does not reproduce the bf16 numerics (impossible on CPU). It pins the two
properties that make that cascade BENIGN rather than a bug:

  1. Legality is enforced independently of the logits: no perturbation, however large,
     can make `greedy_decode_batch` take an off-grammar step or retire a row early. A
     biased argmax only ever moves among the PDA's legal actions, and the row still
     terminates on a real EOS with a complete tree.
  2. A perturbation toward CONTENT collapses sexp segmentation to a valid shorter tree
     (the observed 159->43 pattern), it does not raise or produce a malformed parse.

If batching ever corrupted a row's logits (a real bug), `sr` would diverge as much as
`sexp` -- they share `Backbone.seed_rows`/`advance_rows` verbatim through the
serialization-agnostic `greedy_decode_batch`. It does not, which is the evidence the
divergence is downstream sensitivity, not a batching bug.
"""

import os

import pytest

pytest.importorskip("transformers")

import torch

from dipo.rst.data.tree import Reduce, RstTree, Shift
from dipo.rst.parsers.gen.configuration_gen import GenConfig
from dipo.rst.parsers.gen.modeling_gen import GenParser

SMALL_CAUSAL = os.environ.get("DIPO_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
RELATION_TYPES = [("elaboration", "rst"), ("joint", "multinuc"), ("contrast", "rst")]


def _gen(**ov) -> GenParser:
    d = dict(
        backbone="decoder_only", serialization="sexp", train_dir="x", dev_dir="x",
        model_name=SMALL_CAUSAL, relation_types=RELATION_TYPES,
        amp=False, max_input_length=512, max_output_length=512, batched_decode=True,
    )
    d.update(ov)
    try:
        parser = GenParser(GenConfig.from_dict(d))
    except Exception as e:
        pytest.skip(f"Could not build gen decoder_only/sexp on {SMALL_CAUSAL}: {e!r}")
    torch.manual_seed(0)
    parser.eval()
    return parser


def _tree(edu_texts, reduces) -> RstTree:
    actions = [Shift(edu_text=t) for t in edu_texts]
    for nuc, rel in reduces:
        actions.append(Reduce(nuc=nuc, rel=rel))
    return RstTree.from_shift_reduce(actions, relation_types=RELATION_TYPES)


def _short() -> RstTree:
    return _tree(["Cats sleep all day.", "Dogs bark loudly."], [("NS", "elaboration")])


def _long() -> RstTree:
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
        ("NS", "elaboration"), ("NN", "contrast"), ("NS", "elaboration"), ("NS", "elaboration"),
        ("NN", "joint"), ("NS", "elaboration"), ("NN", "joint"),
    ]
    return _tree(edus, reduces)


def _n_edus(tree) -> int:
    return len(getattr(tree, "_pred_edu_source_ranges", []) or [])


def _bias_content(parser, delta: float) -> None:
    """Add `delta` to the CONTENT (copy) column of every batched-decode logit row, so
    greedy prefers to keep eating source content wherever the PDA still allows it. The
    bias is applied to the raw logits BEFORE the decode loop masks them, so at any step
    where CONTENT is illegal (cursor == source_len, or an internal-node slot) the mask
    still zeroes it out -- the bias can only ever tip a choice among LEGAL actions."""
    ser = parser.serialization
    assert ser.uses_small_head(), "test assumes the default use_copy=True token head"
    col = ser.copy_head_idx
    bb = parser.backbone
    orig_seed, orig_adv = bb.seed_rows, bb.advance_rows

    def nudge(logits):
        logits = logits.clone()
        logits[:, col] += delta
        return logits

    def seed_rows(prefixes):
        logits, cache = orig_seed(prefixes)
        return nudge(logits), cache

    def advance_rows(cache, next_inputs):
        logits, cache2 = orig_adv(cache, next_inputs)
        return nudge(logits), cache2

    bb.seed_rows = seed_rows
    bb.advance_rows = advance_rows


def test_content_bias_cascades_to_valid_shorter_tree_without_error():
    """A large CONTENT bias on the batched path -- the extreme of the tie-flips real
    bf16 introduces -- collapses each sexp document to a single-EDU tree. The decode
    raises nothing (every step stays legal), completes on a real EOS, and yields a
    well-formed parse whose segmentation is a subset of the unbiased one. This is the
    159->43 collapse reproduced deterministically: a legal greedy path, not a bug."""
    batch = [_short(), _long()]

    # Unbiased batched decode (the reference the bf16 batched path would drift from).
    base = _gen()
    base_trees = base.predict_batch(batch, num_beams=1)
    assert len(base_trees) == 2
    for t in base_trees:
        assert not getattr(t, "_from_sexp_failed", False), "unbiased batched decode produced a malformed parse"
    base_long_edus = _n_edus(base_trees[1])

    # Same batch, but every row now greedily over-eats content.
    biased = _gen()
    _bias_content(biased, delta=1e4)
    biased_trees = biased.predict_batch(batch, num_beams=1)  # must not raise

    assert len(biased_trees) == 2
    for k, t in enumerate(biased_trees):
        # Completeness: a real EOS-terminated, well-formed tree (never the malformed
        # fallback and never an OverLengthError -- both would signal the loop failed to
        # drive the PDA to a legal terminal state).
        assert not getattr(t, "_from_sexp_failed", False), f"biased row {k} produced a malformed parse"
        # Collapse: over-eating content merges leaves, so segmentation can only shrink.
        assert _n_edus(t) >= 1

    # The long document, unbiased, segments into several EDUs; the content bias collapses
    # it strictly -- the divergence is a real (and much shorter) alternative parse.
    assert _n_edus(biased_trees[1]) <= base_long_edus
    assert base_long_edus > 1, "fixture too weak: unbiased long doc did not segment, cannot show a collapse"
    assert _n_edus(biased_trees[1]) < base_long_edus, "content bias did not collapse segmentation"


def test_content_bias_never_forces_an_illegal_step_or_early_retire():
    """Sweep the bias magnitude, including the sign that pushes AWAY from content. No
    setting makes `greedy_decode_batch` raise DecodeInvariantError (an off-grammar or
    mask/automaton-mismatch step) or OverLengthError (a row that failed to terminate):
    the mask + PDA gate legality independently of the logits, so batching -- which only
    perturbs the logits -- can never turn a valid decode into an invalid one."""
    batch = [_short(), _long()]
    for delta in (-1e4, -1e2, 1e2, 1e4):
        parser = _gen()
        _bias_content(parser, delta=delta)
        trees = parser.predict_batch(batch, num_beams=1)  # raises on any invariant break
        assert len(trees) == 2
        for t in trees:
            assert not getattr(t, "_from_sexp_failed", False)
