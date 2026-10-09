"""A trainable-only checkpoint carries the new EMBEDDING ROWS, not the whole matrix.

The new-row shadow flags the entire embedding `requires_grad` purely so backward
materializes a gradient to slice from; `optimizer_parameters()` then drops the matrix
and trains the small shadow in its stead. `save_checkpoint(trainable_only=True)` used
to select by `requires_grad` alone, so it asked "what has a grad flag?" when it meant
"what does the optimizer train?" -- and shipped the full embedding.

Measured on a real archived run (paper-dec-sr-qwen36-27b-post-chat, LoRA r=16): one
2.37GB `embed_tokens.weight` entry inside a 2.83GB checkpoint whose every other entry
was ~1MB of adapter. The trained part was a handful of rows.

The matrix cannot simply be dropped -- it is what carries the trained new rows -- so
the shadows are persisted under `_new_rows.<param name>` and overlaid onto the freshly
constructed matrices at load, via the same `commit_new_embedding_rows` write the
training loop does after every step.
"""

import dataclasses
import os

import pytest

pytest.importorskip("transformers")

import torch

from dipo.common.training import load_model_state, save_checkpoint
from dipo.rst.parsers.common.inference import load_parser_from_checkpoint
from dipo.rst.parsers.gen.configuration_gen import GenConfig
from dipo.rst.parsers.gen.modeling_gen import NEW_ROW_KEY_PREFIX, GenParser

SMALL_CAUSAL = os.environ.get("DIPO_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")

EMB_NAME = "backbone.model.model.embed_tokens.weight"


def _cfg(**over) -> GenConfig:
    d = dict(
        backbone="decoder_only",
        serialization="sr",
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_CAUSAL,
        relation_types=[("elaboration", "rst"), ("contrast", "rst")],
        label_style="words",
        peft=None,
        gradient_checkpointing=False,
        amp=False,
    )
    d.update(over)
    return GenConfig.from_dict(d)


def _gen(**over) -> GenParser:
    """A gen parser with its new-row shadow installed. Seeded: the tiny fixture is
    randomly initialized on every construction (its BASE rows differ build to build),
    so an unseeded rebuild could not tell "the overlay disturbed the frozen rows" from
    "this is simply a different random model"."""
    torch.manual_seed(1234)
    try:
        gen = GenParser(_cfg(**over))
    except Exception as e:  # noqa: BLE001 -- network / gated weights
        pytest.skip(f"Could not build gen on {SMALL_CAUSAL}: {e!r}")
    gen.configure_new_row_training()
    return gen


def _untie_lm_head(gen: GenParser) -> None:
    """Break the weight tie so the untied second shadow (dormant on every grid backbone,
    which all tie) is actually exercised."""
    model = gen.backbone.underlying_model()
    head = model.get_output_embeddings()
    if head is None:
        pytest.skip("backbone has no locatable lm_head to untie")
    head.weight = torch.nn.Parameter(head.weight.detach().clone())
    model.config.tie_word_embeddings = False


def _nbytes(state: dict) -> int:
    return sum(v.numel() * v.element_size() for v in state.values())


def _old_style_trainable_state(gen: GenParser) -> dict:
    """What save_checkpoint used to produce: filter the state_dict by requires_grad."""
    trainable = {n for n, p in gen.named_parameters() if p.requires_grad}
    return {k: v for k, v in gen.state_dict().items() if k in trainable}


def test_full_matrix_is_excluded_and_new_rows_carried():
    gen = _gen()
    n_old = gen.backbone._original_vocab_size
    n_new = gen.backbone.embedding_weight().shape[0] - n_old
    state = gen.trainable_state_dict()

    assert EMB_NAME not in state, "the full embedding is still in the trainable-only state"
    key = NEW_ROW_KEY_PREFIX + EMB_NAME
    assert key in state, f"missing compact new rows; got {sorted(state)[:5]}"
    assert tuple(state[key].shape) == (n_new, gen.backbone.embedding_weight().shape[1])


def test_new_state_is_orders_of_magnitude_smaller():
    """The actual claim. Fails against the old requires_grad filter."""
    gen = _gen()
    old_bytes = _nbytes(_old_style_trainable_state(gen))
    new_bytes = _nbytes(gen.trainable_state_dict())
    assert new_bytes * 100 < old_bytes, f"expected a big shrink, got {old_bytes} -> {new_bytes} bytes"


def test_roundtrip_restores_new_rows_and_leaves_base_rows_pretrained(tmp_path):
    gen = _gen()
    n_old = gen.backbone._original_vocab_size

    # "Train" the new rows: write a recognizable pattern through the shadow, exactly as
    # a step would (shadow -> commit -> live matrix).
    shadow = gen.backbone.shadow_pairs()[0][1]
    with torch.no_grad():
        shadow.data = torch.full_like(shadow.data, 0.4242)
    gen.backbone.commit_new_embedding_rows()
    trained_rows = gen.backbone.embedding_weight()[n_old:].detach().clone()
    base_rows = gen.backbone.embedding_weight()[:n_old].detach().clone()

    opt = torch.optim.AdamW(gen.optimizer_parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(
        path,
        gen,
        opt,
        sched,
        trainable_only=True,
        epoch=1,
        config=dataclasses.asdict(gen.config),
    )

    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    assert ckpt["trainable_only"] is True
    # Exercise the public standalone-inference path: its freshly constructed parser
    # has no shadows installed before load. The checkpoint loader must install them.
    torch.manual_seed(1234)
    fresh = load_parser_from_checkpoint(path, torch.device("cpu"), GenConfig, GenParser)

    got = fresh.backbone.embedding_weight()
    assert torch.allclose(got[n_old:], trained_rows), "trained new rows did not survive the round trip"
    assert torch.allclose(got[:n_old], base_rows), "overlay disturbed the frozen pretrained rows"
    # the shadow must track the matrix it mirrors, or the next step would resume from stale rows
    assert torch.allclose(fresh.backbone.shadow_pairs()[0][1].detach(), trained_rows)


def test_untied_head_shadows_and_restores_both_matrices(tmp_path):
    gen = _gen(peft=None, new_row_lr=1e-3)
    _untie_lm_head(gen)
    installed = gen.backbone.install_new_row_shadow(include_untied_lm_head=True)
    if installed is None or len(gen.backbone.shadow_pairs()) != 2:
        pytest.skip("this fixture does not produce a second (untied lm_head) shadow")

    n_old = gen.backbone._original_vocab_size
    for _w, shadow in gen.backbone.shadow_pairs():
        with torch.no_grad():
            shadow.data = torch.full_like(shadow.data, 0.31)
    gen.backbone.commit_new_embedding_rows()
    expected = [w[n_old:].detach().clone() for w, _ in gen.backbone.shadow_pairs()]

    state = gen.trainable_state_dict()
    new_row_keys = [k for k in state if k.startswith(NEW_ROW_KEY_PREFIX)]
    assert len(new_row_keys) == 2, f"expected both matrices shadowed, got {new_row_keys}"
    for k in new_row_keys:
        assert k[len(NEW_ROW_KEY_PREFIX) :] not in state, "a shadowed full matrix leaked into the state"

    fresh = _gen(peft=None, new_row_lr=1e-3)
    _untie_lm_head(fresh)
    fresh.backbone.install_new_row_shadow(include_untied_lm_head=True)
    fresh.load_trainable_state_dict(state)
    for (w, _), want in zip(fresh.backbone.shadow_pairs(), expected, strict=True):
        assert torch.allclose(w[n_old:].detach(), want)


def test_old_format_checkpoint_still_loads(tmp_path):
    """Checkpoints written before this change carry the full matrix and no `_new_rows.`
    keys. The in-flight big runs are all old-format, so they must keep loading."""
    gen = _gen()
    n_old = gen.backbone._original_vocab_size
    shadow = gen.backbone.shadow_pairs()[0][1]
    with torch.no_grad():
        shadow.data = torch.full_like(shadow.data, 0.77)
    gen.backbone.commit_new_embedding_rows()
    trained_rows = gen.backbone.embedding_weight()[n_old:].detach().clone()

    old_state = _old_style_trainable_state(gen)  # includes the full embedding
    assert EMB_NAME in old_state
    assert not any(k.startswith(NEW_ROW_KEY_PREFIX) for k in old_state)

    fresh = _gen()
    load_model_state(fresh, {"model_state_dict": old_state, "trainable_only": True})
    assert torch.allclose(fresh.backbone.embedding_weight()[n_old:], trained_rows)


def test_new_rows_for_an_unknown_matrix_are_rejected():
    gen = _gen()
    state = gen.trainable_state_dict()
    state[NEW_ROW_KEY_PREFIX + "backbone.model.nonexistent.weight"] = torch.zeros(2, 2)
    with pytest.raises(RuntimeError, match="no .*shadow installed"):
        gen.load_trainable_state_dict(state)


def test_new_rows_with_the_wrong_shape_are_rejected():
    gen = _gen()
    state = gen.trainable_state_dict()
    state[NEW_ROW_KEY_PREFIX + EMB_NAME] = torch.zeros(99, 3)
    with pytest.raises(RuntimeError, match="Vocab/architecture mismatch"):
        gen.load_trainable_state_dict(state)
