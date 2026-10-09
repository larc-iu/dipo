"""Smoke coverage for the gen training machinery: the backbone collator, the
new-row shadow scheme (only newly-added token rows train, for the input embedding
and, on an untied lm_head under LoRA in words mode, the lm_head too), and a
forward/backward/optimizer step. The full train() loop (curriculum, eval,
checkpointing) is a faithful port of the hand-written loops and needs data dirs,
so it is exercised end-to-end at real-run time, not here.
"""
import os

import pytest

pytest.importorskip("transformers")

SMALL_CAUSAL = os.environ.get("DIPO_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
SMALL_SEQ2SEQ = os.environ.get("DIPO_TEST_SEQ2SEQ_MODEL", "google-t5/t5-small")

import torch

from dipo.rst.data.tree import Reduce, RstTree, Shift
from dipo.rst.parsers.gen.configuration_gen import GenConfig
from dipo.rst.parsers.gen.modeling_gen import GenParser


def _gen(backbone: str, serialization: str, **ov) -> GenParser:
    """Build a `gen` parser directly for the given (backbone, serialization) combo
    on a tiny offline backbone. Skips (not fails) if the model can't be fetched."""
    model = SMALL_CAUSAL if backbone == "decoder_only" else SMALL_SEQ2SEQ
    d = dict(
        backbone=backbone, serialization=serialization, train_dir="x", dev_dir="x",
        model_name=model, relation_types=[("elaboration", "rst"), ("joint", "multinuc")],
        amp=False, max_input_length=128, max_output_length=128,
    )
    d.update(ov)
    try:
        return GenParser(GenConfig.from_dict(d))
    except Exception as e:
        pytest.skip(f"Could not build gen {backbone}/{serialization} on {model}: {e!r}")


def _toy_tree() -> RstTree:
    return RstTree.from_shift_reduce(
        [
            Shift(edu_text="Cats sleep."), Shift(edu_text="Dogs bark loudly."), Shift(edu_text="Birds fly south."),
            Reduce(nuc="NS", rel="elaboration"), Reduce(nuc="NN", rel="joint"),
        ],
        relation_types=[("elaboration", "rst"), ("joint", "multinuc")],
    )


def _train_step_and_check(gen: GenParser, n_total: int, n_new: int):
    """One forward/backward/stage/step/commit; assert EVERY installed new-row shadow
    (input embedding, and the untied lm_head when present) carries the new-row gradient,
    its full matrix is excluded from the optimizer, and only the new rows move (the
    pretrained rows stay fixed)."""
    gen.train()
    trees = [_toy_tree(), _toy_tree()]
    examples = [gen.encode_target(t) for t in trees]
    assert all(e is not None for e in examples)
    pad_id = int(gen.backbone.tokenizer.pad_token_id)
    batch = gen.backbone.collate(examples, pad_id)

    out = gen(batch)
    assert torch.isfinite(out["loss"]), out["loss"]
    out["loss"].backward()

    n_old = n_total - n_new
    pairs = gen.backbone._shadow_pairs()
    assert pairs, "token/full-head mode must install at least the input-embedding shadow"
    befores = [(w, s, w.detach().clone()) for w, s in pairs]
    # backward materializes the full (unmasked) matrix grads; stage moves each new-row
    # slice onto its shadow and frees the full grad.
    for weight, _shadow in pairs:
        assert weight.grad is not None, "backward must materialize the full matrix grad to slice"
    gen.stage_new_embedding_row_grads()
    for weight, shadow in pairs:
        assert weight.grad is None, "stage did not free the full matrix grad"
        assert shadow.grad is not None and shadow.grad.abs().max().item() > 0.0, "shadow missing new-row grad"

    params = gen.optimizer_parameters()
    for weight, shadow in pairs:
        assert all(p is not weight for p in params), "a shadowed full matrix must be excluded from the optimizer"
        assert any(p is shadow for p in params), "the shadow must be in the optimizer params"

    opt = torch.optim.AdamW(params, lr=1e-1)
    opt.step()
    opt.zero_grad()
    gen.commit_new_embedding_rows()
    for weight, _shadow, before in befores:
        after = weight.detach()
        assert torch.equal(after[:n_old], before[:n_old]), "pretrained rows must not move"
        assert not torch.equal(after[n_old:], before[n_old:]), "new rows must update via the shadow"


def test_gen_train_step_decoder_only_sr():
    gen = _gen("decoder_only", "sr")
    masked = gen.configure_new_row_training()
    assert masked is not None, "token mode must freeze old embedding rows"
    _train_step_and_check(gen, *masked)


def test_gen_train_step_seq2seq_sr():
    gen = _gen("seq2seq", "sr")
    masked = gen.configure_new_row_training()
    assert masked is not None
    _train_step_and_check(gen, *masked)


def _force_untie(gen) -> None:
    """Break the fixture's lm_head/embedding weight tie (every supported backbone
    ties, so the untied reject path needs a synthetic untied model)."""
    base = gen.backbone.underlying_model()
    head = getattr(base, "lm_head", None)
    if head is None or not isinstance(getattr(head, "weight", None), torch.Tensor):
        pytest.skip("fixture lm_head is not a plain weighted Linear; cannot force-untie")
    head.weight = torch.nn.Parameter(head.weight.detach().clone())
    assert not gen.backbone.lm_head_tied_to_embeddings(), "force-untie failed"


def test_gen_train_step_untied_lm_head_words_lora():
    """Finding-B fix: words mode + LoRA on an UNTIED lm_head trains the new label/copy
    OUTPUT rows via a SECOND shadow, so both the input embedding and the lm_head new rows
    learn. The tiny causal fixture ties its lm_head to the input embedding, so force-untie
    it (deep-copy the lm_head weight into its own storage) to exercise the untied path
    deterministically without depending on a specific untied-backbone download."""
    peft = dict(r=4, alpha=8, dropout=0.0, target_modules="all-linear", bias="none", dora=False)
    gen = _gen("decoder_only", "sr", label_style="words", peft=peft)
    _force_untie(gen)

    masked = gen.configure_new_row_training()
    assert masked is not None
    assert len(gen.backbone._new_row_shadow_holder) == 2, "words + LoRA + untied lm_head must install input + lm_head shadows"
    # Both shadows must map to distinct matrices (input embedding and the untied lm_head).
    pair_weights = [w for w, _ in gen.backbone._shadow_pairs()]
    assert pair_weights[0].data_ptr() != pair_weights[1].data_ptr(), "the two shadows must target distinct matrices"
    _train_step_and_check(gen, *masked)


def test_new_row_lr_shadows_untied_head_under_ft():
    """new_row_lr installs the untied lm_head shadow even under full FT (peft=null), so the
    new OUTPUT rows can be trained at the separate lr. Force-untie the tied fixture."""
    gen = _gen("decoder_only", "sr", label_style="words", new_row_lr=1e-3)  # peft=null -> FT
    _force_untie(gen)
    masked = gen.configure_new_row_training()
    assert masked is not None
    assert len(gen.new_row_params()) == 2, "new_row_lr + untied FT must shadow input + lm_head"
    _train_step_and_check(gen, *masked)


def test_new_row_lr_optimizer_group():
    """_build_optimizer puts the new-row shadow(s) in their own group at new_row_lr (wd 0),
    with no shadow leaking into the base group."""
    from dipo.rst.parsers.gen.train_gen import _build_optimizer

    gen = _gen("decoder_only", "sr", new_row_lr=1e-3, optimizer="adamw", lr=2e-5)
    gen.configure_new_row_training()
    shadows = gen.new_row_params()
    assert shadows, "token mode installs the input-embedding shadow"
    opt = _build_optimizer(gen, gen.config)
    assert len(opt.param_groups) == 2, "new_row_lr must split into two optimizer groups"
    sid = {id(p) for p in shadows}
    row_grp = next(g for g in opt.param_groups if all(id(p) in sid for p in g["params"]))
    base_grp = next(g for g in opt.param_groups if g is not row_grp)
    assert row_grp["lr"] == 1e-3 and row_grp["weight_decay"] == 0.0
    assert base_grp["lr"] == 2e-5
    assert not any(id(p) in sid for p in base_grp["params"]), "shadow must not also be in the base group"
