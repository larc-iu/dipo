"""gen's `seg` serialization: the shift-reduce stream without reduces, an EDU segmenter."""

import os

import pytest

pytest.importorskip("transformers")

import torch

from dipo.rst.data.tree import Reduce, RstTree, Shift
from dipo.rst.parsers.common.seqgen import ShiftReduceDecodeState
from dipo.rst.parsers.gen.configuration_gen import GenConfig
from dipo.rst.parsers.gen.modeling_gen import GenParser

SMALL_CAUSAL = os.environ.get("DIPO_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
RELATION_TYPES = [("elaboration", "rst"), ("joint", "multinuc")]
EDUS = ["Cats sleep.", "Dogs bark loudly.", "Birds fly south."]


def _toy_tree() -> RstTree:
    return RstTree.from_shift_reduce(
        [Shift(edu_text=e) for e in EDUS] + [Reduce(nuc="NS", rel="elaboration"), Reduce(nuc="NN", rel="joint")],
        relation_types=RELATION_TYPES,
    )


def _seg_parser() -> GenParser:
    d = dict(
        backbone="decoder_only",
        serialization="seg",
        train_dir="x",
        dev_dir="x",
        model_name=SMALL_CAUSAL,
        relation_types=RELATION_TYPES,
        amp=False,
        max_input_length=128,
        max_output_length=128,
    )
    try:
        return GenParser(GenConfig.from_dict(d))
    except OSError as e:
        pytest.skip(f"cannot load {SMALL_CAUSAL}: {e}")


def test_segment_only_automaton_never_reduces():
    st = ShiftReduceDecodeState(source_len=3, segment_only=True)
    st.step_copy()
    st.step_shift()
    st.step_copy()
    st.step_shift()
    assert not st.reduce_ok and not st.eos_ok  # a token is left
    st.step_copy()
    st.step_shift()
    assert st.stack_size == 3 and not st.reduce_ok and st.eos_ok
    assert st.clone().segment_only


def test_config_rejects_seg_without_copy_or_with_word_labels():
    base = dict(backbone="decoder_only", serialization="seg", train_dir="x", dev_dir="x")
    with pytest.raises(ValueError, match="use_copy"):
        GenConfig.from_dict({**base, "use_copy": False})
    with pytest.raises(ValueError, match="label_style"):
        GenConfig.from_dict({**base, "label_style": "words"})


def test_linearize_is_copies_and_shifts_only():
    gen = _seg_parser()
    ser = gen.serialization
    source_ids, seen, label = ser.linearize(_toy_tree())
    assert set(label) == {ser.copy_token_id, ser.shift_token_id}
    assert label.count(ser.shift_token_id) == len(EDUS)
    assert label.count(ser.copy_token_id) == len(source_ids)
    assert [t for t in seen if t != ser.shift_token_id] == list(source_ids)
    assert ser.head_vocab_size == 3  # copy, shift, end of stream


def test_predicted_edus_tile_the_text():
    gen = _seg_parser()
    text = " ".join(EDUS)
    pred = gen.predict_from_text(text)
    assert "".join(pred.edu_strings).replace(" ", "") == text.replace(" ", "")  # every token, once, in order
    assert len(pred.edus) >= 1
    gold = gen.predict_with_gold_edus(_toy_tree())  # forced to the gold boundaries
    assert [e.strip() for e in gold.edu_strings] == EDUS


def test_train_step_has_a_finite_loss_and_gradients():
    gen = _seg_parser()
    gen.configure_new_row_training()
    gen.train()
    examples = [gen.encode_target(_toy_tree()) for _ in range(2)]
    batch = gen.backbone.collate(examples, int(gen.backbone.tokenizer.pad_token_id))
    out = gen(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gen.parameters())
