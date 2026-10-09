"""The inert PeftConfig fields are rejected, not silently ignored.

`modules_to_save` / `train_only_new_embedding_rows` are read by nobody: no parser
touches them and no LoraConfig is built from them. The four hand-written generative
parsers did (`modules_to_save=['embed_tokens']` + a retie, so the full lm_head could
emit source subwords under `use_copy=false`); `gen` replaced that with the new-row
shadow and rejects `use_copy=false` under LoRA outright. So a user setting them got
silence.

They cannot simply be deleted, which is why they are rejected instead of removed:
  * every archived run's config.json carries them and tonga rejects unknown keys, so
    deleting them would make those configs unloadable (`dipo gen eval` reads a run's
    frozen config.json);
  * `peft` is hashed as a nested blob -- HASH_EXCLUDE only filters TOP-LEVEL keys --
    so deleting them would change every LoRA run's id and orphan its run dir.
Both are pinned below.
"""

import os
from dataclasses import asdict

import pytest

from dipo.rst.parsers.common.config import PeftConfig


def test_defaults_are_accepted():
    cfg = PeftConfig()
    assert cfg.modules_to_save == ["embed_tokens"]
    assert cfg.train_only_new_embedding_rows is True


def test_changed_modules_to_save_is_rejected():
    with pytest.raises(ValueError, match="modules_to_save is inert"):
        PeftConfig(modules_to_save=["lm_head"])
    with pytest.raises(ValueError, match="modules_to_save is inert"):
        PeftConfig(modules_to_save=[])


def test_changed_train_only_new_embedding_rows_is_rejected():
    with pytest.raises(ValueError, match="train_only_new_embedding_rows is inert"):
        PeftConfig(train_only_new_embedding_rows=False)


def test_rejection_survives_config_parsing():
    with pytest.raises(Exception, match="modules_to_save is inert"):
        PeftConfig.from_params({"r": 8, "modules_to_save": ["embed_tokens", "lm_head"]})


def test_default_instances_do_not_share_the_mutable_list():
    """The default is a list; two configs must not alias it (mutating one would
    silently 'change' the other, and defeat the equality check above)."""
    a, b = PeftConfig(), PeftConfig()
    assert a.modules_to_save == b.modules_to_save
    assert a.modules_to_save is not b.modules_to_save


# --- why they are kept rather than deleted ---------------------------------


def test_an_archived_config_carrying_the_fields_still_loads():
    """Deleting the fields would make every archived run's config.json unloadable."""
    cfg = PeftConfig.from_params(
        {"r": 16, "alpha": 32, "modules_to_save": ["embed_tokens"], "train_only_new_embedding_rows": True}
    )
    assert cfg.r == 16


def test_the_fields_stay_in_the_hashed_blob():
    """`peft` is hashed whole, so the run id depends on these keys being present."""
    d = asdict(PeftConfig())
    assert "modules_to_save" in d and "train_only_new_embedding_rows" in d


# --- the accepted option's actual effect on the trainable set --------------

SMALL_CAUSAL = os.environ.get("DIPO_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")


def test_lora_trainable_set_has_no_modules_to_save_duplicate():
    """The accepted default must NOT put the embedding in LoRA's modules_to_save: that
    would duplicate the whole vocab x hidden matrix to train a handful of new rows,
    which is exactly what the new-row shadow exists to avoid. So the trainable set is
    LoRA adapters + the shadow, with no `modules_to_save` copy anywhere."""
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from dipo.rst.parsers.gen.configuration_gen import GenConfig
    from dipo.rst.parsers.gen.modeling_gen import GenParser

    d = dict(
        backbone="decoder_only",
        serialization="sr",
        train_dir="<unused>",
        dev_dir="<unused>",
        model_name=SMALL_CAUSAL,
        relation_types=[("elaboration", "rst")],
        peft={"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        gradient_checkpointing=False,
        amp=False,
    )
    try:
        gen = GenParser(GenConfig.from_dict(d))
    except Exception as e:  # noqa: BLE001 -- network / gated weights
        pytest.skip(f"Could not build a LoRA gen on {SMALL_CAUSAL}: {e!r}")
    gen.configure_new_row_training()

    trainable = [n for n, p in gen.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n in trainable), "no LoRA adapter is trainable"
    assert not any("modules_to_save" in n for n in trainable), (
        f"peft duplicated a module via modules_to_save: {[n for n in trainable if 'modules_to_save' in n]}"
    )
    # the optimizer trains the compact shadow, never the full embedding
    opt_params = gen.optimizer_parameters()
    emb = gen.backbone.embedding_weight()
    assert all(p is not emb for p in opt_params), "the full embedding leaked into the optimizer"
    assert any(p is s for _w, s in gen.backbone.shadow_pairs() for p in opt_params), "the shadow is not trained"
