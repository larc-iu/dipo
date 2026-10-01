"""Regression tests for two generative-parser paths the runtime smoke sweep
missed: (1) checkpoint save -> reload-from-disk -> predict (strict
`load_state_dict` of the carved embedding Parameter, the small action head, and
the resized action vocab), and (2) ragged multi-document `predict_batch`.

Tiny models, CPU. Skips gracefully if a model can't be fetched.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch

pytest.importorskip("transformers")

from iudex.common.training import save_checkpoint  # noqa: E402
from iudex.rst.parsers.common.inference import load_parser_from_checkpoint  # noqa: E402
from iudex.rst.parsers.gen.configuration_gen import GenConfig  # noqa: E402
from iudex.rst.parsers.gen.modeling_gen import GenParser  # noqa: E402

T5 = os.environ.get("IUDEX_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")
CAUSAL = os.environ.get("IUDEX_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")


def _make_parser(backbone: str, serialization: str, model: str, extra: dict) -> GenParser:
    d = {
        "backbone": backbone,
        "serialization": serialization,
        "train_dir": "<unused>",
        "dev_dir": "<unused>",
        "model_name": model,
        "relation_types": [["elaboration", "rst"], ["joint", "multinuc"]],
        "amp": False,
        "max_input_length": 256,
        "max_output_length": 512,
        "num_beams": 1,
        **extra,
    }
    try:
        return GenParser(GenConfig.from_dict(d))
    except Exception as e:  # network / gated model / arch mismatch on this host
        pytest.skip(f"Could not construct gen {backbone}/{serialization} with {model}: {e!r}")


# (id, backbone, serialization, model, extra) -- covers carve+small-head (use_copy)
# and the full-head path (use_copy=False), both backbones.
ROUNDTRIP_CASES = [
    ("seq2seq_sr", "seq2seq", "sr", T5, {}),
    ("decoder_only_sr", "decoder_only", "sr", CAUSAL, {}),
    (
        "seq2seq_sexp_copy",
        "seq2seq",
        "sexp",
        T5,
        {"traversal_order": "postorder", "use_copy": True},
    ),
    (
        "seq2seq_sexp_nocopy",
        "seq2seq",
        "sexp",
        T5,
        {"traversal_order": "postorder", "use_copy": False},
    ),
    (
        "decoder_only_sexp_copy",
        "decoder_only",
        "sexp",
        CAUSAL,
        {"traversal_order": "preorder", "use_copy": True},
    ),
]


@pytest.mark.parametrize(
    "name,backbone,serialization,model,extra", ROUNDTRIP_CASES, ids=[c[0] for c in ROUNDTRIP_CASES]
)
def test_checkpoint_roundtrip_then_predict(tmp_path, name, backbone, serialization, model, extra):
    """Save a freshly-built parser, reload it from disk, and predict. Catches
    state_dict drift in the carved embedding / small head / resized vocab."""
    parser = _make_parser(backbone, serialization, model, extra)
    opt = torch.optim.SGD([p for p in parser.parameters() if p.requires_grad], lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)
    ckpt = str(tmp_path / "best_model.pt")
    save_checkpoint(
        ckpt,
        parser,
        opt,
        sched,
        config=dataclasses.asdict(parser.config),
        config_hash="test",
        global_step=0,
        epoch=0,
        best_val=0.0,
        parser_kind="gen",
    )
    # Reload from disk (the real from_pretrained path).
    loaded = load_parser_from_checkpoint(ckpt, torch.device("cpu"), GenConfig, GenParser)
    tree = loaded.predict_from_text("The plan was clear. The result was not.")
    assert len(tree.edus) >= 1


def test_predict_batch_ragged():
    """predict_batch over 2 ragged documents returns one tree per doc. gen decodes
    per-document (there is no batched-greedy ShiftReduceDecodeState path like the
    old seq2seq_sr's), so this only pins the per-doc fan-out and ragged handling."""
    parser = _make_parser("seq2seq", "sr", T5, {})
    texts = ["Short doc here.", "A noticeably longer document with several more tokens to force ragged lengths."]
    trees = parser.predict_batch_from_texts(texts, num_beams=1)
    assert len(trees) == 2
    assert all(len(t.edus) >= 1 for t in trees)
