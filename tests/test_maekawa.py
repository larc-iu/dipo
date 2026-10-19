"""Maekawa et al. (2024) port: prompts, gold decisions, tree assembly, and the parser on a
tiny offline causal LM (CPU, unquantized)."""

import os
from pathlib import Path

import pytest
import torch

from dipo.rst.data.metrics import evaluate_parseval
from dipo.rst.data.tree import RstTree
from dipo.rst.parsers.maekawa.configuration_maekawa import MaekawaConfig
from dipo.rst.parsers.maekawa.decisions import (
    QUEUE_EMPTY,
    Labels,
    gold_decisions,
    gold_nodes,
    joint_label_frequencies,
    joint_prompt,
    middle_split,
    nuc_prompt,
    rel_with_nuc_prompt,
    resolve_unified_bu,
    resolve_unified_td,
    span_prompt,
    top_down_prompt,
    tree_from_nodes,
    unified_bu_prompt,
    unified_td_prompt,
)

SMALL_CAUSAL = os.environ.get("DIPO_TEST_CAUSAL_MODEL", "hf-internal-testing/tiny-random-Gemma3ForCausalLM")
EDUS = [" The cat sat.", "It was tired ", "so it slept.", "Then it woke."]
RELATION_TYPES = [("elaboration", "rst"), ("joint", "multinuc"), ("cause", "rst")]
# ((e0 e1)_NS:elaboration e2)_NS:cause, then that with e3 under NN:joint
ACTIONS = [(3, "NN", "joint"), (2, "NS", "cause"), (1, "NS", "elaboration")]


def toy_tree() -> RstTree:
    return RstTree.from_parsing_actions(ACTIONS, EDUS, relation_types=RELATION_TYPES)


def labels() -> Labels:
    return Labels(["elaboration", "joint", "cause"], {"elaboration": "Elaboration"})


def test_prompts_match_their_templates():
    edus = [e.strip() for e in EDUS]
    assert span_prompt(edus, (0, 2), (2, 3), 3) == (
        "Stack2: The cat sat. It was tired\nStack1: so it slept.\nQueue1: Then it woke.\nAction (shift or reduce): "
    )
    assert nuc_prompt(edus, (0, 1), (1, 2)) == (
        "Span1: The cat sat.\nSpan2: It was tired\n"
        "Nucleus label (nucleus-nucleus, nucleus-satellite, or satellite-nucleus): "
    )
    assert rel_with_nuc_prompt(edus, (0, 1), (1, 2), "nucleus-satellite", ["Elaboration", "joint", "cause"]) == (
        "Span1: The cat sat.\nSpan2: It was tired\nNucleus label: nucleus-satellite\n"
        "Relation label (Elaboration, joint, or cause): "
    )
    assert top_down_prompt(EDUS, 0, 3) == "Input: The cat sat. [0] It was tired [1] so it slept.\nSplit point (0–1): "


def test_gold_decisions_cover_what_their_generator_emits():
    tree = toy_tree()
    ds = gold_decisions(tree, labels(), ("span", "nuc", "rel_with_nuc", "top_down"))
    by_task = {t: [d for d in ds if d.task == t] for t in ("span", "nuc", "rel_with_nuc", "top_down")}
    assert len(by_task["nuc"]) == len(by_task["rel_with_nuc"]) == 3  # n - 1 internal nodes
    assert sorted(d.key for d in by_task["top_down"]) == [(0, 2, 3), (0, 3, 4)]  # spans of > 2 EDUs
    assert {d.answer for d in by_task["rel_with_nuc"]} == {"Elaboration", "joint", "cause"}
    # shift/reduce decisions only where both are legal (stack >= 2, queue non-empty): this
    # left-branching tree never has two items stacked before a shift, so both are reduces
    assert [(d.key, d.answer) for d in by_task["span"]] == [((0, 1, 1, 2, 2), "reduce"), ((0, 2, 2, 3, 3), "reduce")]


def test_nodes_reassemble_the_gold_tree():
    tree = toy_tree()
    nodes = [(b, s, e, nuc, rel) for (b, e), (s, nuc, rel) in gold_nodes(tree).items()]
    rebuilt = tree_from_nodes(nodes, tree, RELATION_TYPES)
    assert evaluate_parseval([tree], [rebuilt])["full_f1"] == 1.0


def test_config_rejects_adapters_for_unused_tasks():
    with pytest.raises(ValueError, match="top_down"):
        MaekawaConfig(
            train_dir="x",
            dev_dir="y",
            parse_type="bottom_up",
            label_mode="separate",
            pretrained_adapters={"top_down": "a/b"},
        )


def test_config_defaults_are_the_unified_top_down_setup():
    cfg = MaekawaConfig(train_dir="x", dev_dir="y")
    assert (cfg.parse_type, cfg.label_mode, cfg.tasks) == ("top_down", "unified", ("unified_td",))
    assert cfg.model_name == "Qwen/Qwen3.8-27B" and not cfg.peft.load_in_4bit and cfg.gradient_checkpointing
    assert (cfg.peft.r, cfg.peft.alpha, cfg.peft.dropout) == (64, 16, 0.1)


@pytest.mark.parametrize(
    "bad",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"max_microbatch_tokens": 0},
        {"relation_labels": ["a"], "default_relation": "b"},
        {"relation_labels": ["a"], "relation_surface": {"b": "B"}},
        {"label_mode": "separate", "only_tasks": ["nuc"], "pretrained_adapters": {"nuc": "x"}},
    ],
)
def test_config_rejects_inconsistent_settings(bad):
    with pytest.raises(ValueError):
        MaekawaConfig(train_dir="x", dev_dir="y", **bad)


def test_adapter_names_never_contain_one_another():
    from dipo.rst.parsers.maekawa.modeling_maekawa import ADAPTER

    names = set(ADAPTER.values())
    assert not any(a != b and a in b for a in names for b in names)


_PARSERS = {}


JOINT_LABELS = [("NS", "elaboration"), ("NN", "joint"), ("NS", "cause"), ("SN", "cause")]


def make_parser(parse_type: str = "bottom_up", label_mode: str = "separate"):
    from dipo.rst.parsers.common.config import PeftConfig
    from dipo.rst.parsers.maekawa.modeling_maekawa import MaekawaParser

    if (parse_type, label_mode) in _PARSERS:
        return _PARSERS[(parse_type, label_mode)]
    cfg = MaekawaConfig(
        parse_type=parse_type,
        label_mode=label_mode,
        joint_labels=JOINT_LABELS if label_mode == "joint" else None,
        train_dir="x",
        dev_dir="y",
        relation_types=RELATION_TYPES,
        relation_labels=["elaboration", "joint", "cause"],
        model_name=SMALL_CAUSAL,
        peft=PeftConfig(r=4, alpha=8, dropout=0.0, load_in_4bit=False),
        gradient_checkpointing=False,
        record_distributions=True,
        segmentation=None,
    )
    try:
        _PARSERS[(parse_type, label_mode)] = MaekawaParser(cfg, training=True)
    except OSError as e:  # offline without the cached fixture
        pytest.skip(f"cannot load {SMALL_CAUSAL}: {e}")
    return _PARSERS[(parse_type, label_mode)]


@pytest.fixture
def tiny_parser():
    return make_parser()


@pytest.mark.parametrize("parse_type", ["bottom_up", "top_down"])
@pytest.mark.parametrize("decode", ["generate", "score"])
def test_predict_builds_a_tree_over_the_given_edus(parse_type, decode, monkeypatch):
    parser = make_parser(parse_type)
    monkeypatch.setattr(parser.config, "decode", decode)
    pred = parser.predict(toy_tree())
    assert pred.edu_strings == toy_tree().edu_strings
    assert len(pred.spans_with_ranges()) == len(EDUS) - 1
    for r in parser.last_decisions:
        assert r["answer"] in r["candidates"]
        assert len(r["logprobs"]) == len(r["candidates"])


def test_score_matches_a_full_forward(tiny_parser):
    """The shared-prefix KV-cache scoring equals scoring each prompt+answer from scratch."""
    p = tiny_parser
    ids = torch.tensor([p._prompt_ids("Span1: a\nSpan2: b\nNucleus label: ")], device=p.device)
    cands = ["nucleus-satellite", "nucleus-nucleus", "satellite-nucleus"]
    p.activate("nuc")
    fast = p.score(ids, cands, chunk=2)
    for c, got in zip(cands, fast, strict=True):
        a = p._answer_ids(c)
        full = torch.tensor([ids[0].tolist() + a], device=p.device)
        lp = torch.log_softmax(p.lm(input_ids=full).logits[0].float(), -1)
        want = sum(lp[ids.shape[1] - 1 + j, t].item() for j, t in enumerate(a))
        assert got == pytest.approx(want, abs=1e-3)


def test_training_loss_reaches_only_the_active_adapter(tiny_parser):
    p = tiny_parser
    p.train()
    p.activate("nuc")
    batch = [d for d in gold_decisions(toy_tree(), p.labels, ("nuc",))]
    loss = p(batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = {n for n, w in p.lm.named_parameters() if w.grad is not None and w.grad.abs().sum() > 0}
    assert grads and all(".nuc." in n for n in grads)
    p.lm.zero_grad()


def test_trainable_state_round_trips(tiny_parser):
    state = tiny_parser.trainable_state_dict()
    assert {k.split("::")[0] for k in state} == {"span", "nuc", "rel_with_nuc"}
    per_task = {t: sum(1 for k in state if k.startswith(t + "::")) for t in ("span", "nuc", "rel_with_nuc")}
    assert len(set(per_task.values())) == 1  # no adapter's weights leak into another's
    before = {k: v.clone() for k, v in state.items()}
    with torch.no_grad():
        for n, w in tiny_parser.lm.named_parameters():
            if "lora_" in n:
                w.add_(1.0)
    tiny_parser.load_trainable_state_dict(before)
    after = tiny_parser.trainable_state_dict()
    assert all(torch.equal(after[k], before[k]) for k in before)
    with pytest.raises(RuntimeError, match="does not train"):
        tiny_parser.load_trainable_state_dict({"top_down::x": torch.zeros(1)})


def test_batched_loss_equals_accumulated_single_losses(tiny_parser):
    """A micro-batch of k gives the same loss as the mean of k micro-batches of 1, padding
    and all (their recipe is 1 x 16 accumulated)."""
    p = tiny_parser
    p.eval()
    p.activate("rel_with_nuc")
    batch = gold_decisions(toy_tree(), p.labels, ("rel_with_nuc",))
    with torch.no_grad():
        together = p(batch)["loss"].item()
        apart = sum(p([d])["loss"].item() for d in batch) / len(batch)
    assert together == pytest.approx(apart, abs=1e-4)


def test_answer_only_loss_equals_a_full_logits_loss(tiny_parser):
    """forward() runs the LM head at answer-predicting positions only; it must equal the
    loss computed from the full model's logits at every position."""
    p = tiny_parser
    p.eval()
    p.activate("rel_with_nuc")
    batch = gold_decisions(toy_tree(), p.labels, ("rel_with_nuc",))
    with torch.no_grad():
        got = p(batch)["per_decision"]
        for i, d in enumerate(batch):
            prompt, answer = p._prompt_ids(d.prompt), p._answer_ids(d.answer)
            ids = torch.tensor([prompt + answer], device=p.device)
            lp = torch.log_softmax(p.lm(input_ids=ids).logits[0].float(), -1)
            want = -sum(lp[len(prompt) - 1 + j, t].item() for j, t in enumerate(answer)) / len(answer)
            assert got[i].item() == pytest.approx(want, abs=1e-4)


def test_optimizer_steps_cover_16_decisions_under_the_caps(tiny_parser):
    import dataclasses
    import random

    from dipo.rst.parsers.maekawa.train_maekawa import optimizer_steps

    p = tiny_parser
    decisions = gold_decisions(toy_tree(), p.labels, ("nuc", "rel_with_nuc", "span")) * 20
    cfg = dataclasses.replace(p.config, batch_size=8, grad_accum=2, max_microbatch_tokens=200)
    steps = optimizer_steps(p, cfg, decisions, random.Random(0))
    assert sorted(id(d) for s in steps for m in s for d in m) == sorted(id(d) for d in decisions)
    assert all(sum(len(m) for m in s) == 16 for s in steps[:-1])
    for s in steps:
        for m in s:
            width = max(len(p._prompt_ids(d.prompt)) + len(p._answer_ids(d.answer)) for d in m)
            assert len(m) <= 8 and (len(m) == 1 or width * len(m) <= 200)
    faithful = optimizer_steps(
        p, dataclasses.replace(p.config, batch_size=1, grad_accum=16), decisions, random.Random(0)
    )
    assert all(len(m) == 1 for s in faithful for m in s)


def _legal_answer(prompt: str, labels: Labels) -> str:
    """A legal answer to `prompt` that depends on its text, so answers routed to the wrong
    document show up (the tiny random model only produces text that falls back)."""
    import re
    import zlib

    pick = lambda options: options[zlib.crc32(prompt.encode()) % len(options)]  # noqa: E731
    question = prompt.rsplit("\n", 1)[-1]
    split = re.search(r"Split point \(0–([0-9]+)\)", question)
    if question.startswith("Action (shift or reduce)"):
        return pick(["shift", "reduce"])
    if question.startswith("Action (shift, or reduce"):
        reduce = "reduce " + pick(labels.joint_answers)
        return reduce if prompt.split("\n")[2] == "Queue1:" else pick(["shift", reduce])
    if question.startswith("Nucleus label"):
        return pick(["nucleus-satellite", "nucleus-nucleus", "satellite-nucleus"])
    if question.startswith("Relation label"):
        return pick(labels.answers)
    if question.startswith("Nucleus and relation"):
        return pick(labels.joint_answers)
    k = pick([str(i) for i in range(int(split.group(1)) + 1)])
    return f"{k} {pick(labels.joint_answers)}" if "nuclearity" in question else k


@pytest.mark.parametrize("label_mode", ["separate", "joint", "unified"])
@pytest.mark.parametrize("parse_type", ["bottom_up", "top_down"])
def test_lockstep_parsing_matches_one_document_at_a_time(parse_type, label_mode, monkeypatch):
    """predict_batch answers each round's decisions in batches; it must make exactly the
    decisions predict makes document by document, each document getting its own answers."""
    parser = make_unified_parser(parse_type) if label_mode == "unified" else make_parser(parse_type, label_mode)
    monkeypatch.setattr(parser.config, "decode", "generate")
    tok = parser.tokenizer

    def generate(prompts, max_new_tokens):
        return [_legal_answer(tok.decode(p, skip_special_tokens=True), parser.labels) for p in prompts]

    monkeypatch.setattr(parser, "_generate", generate)
    docs = [toy_tree(), RstTree.from_parsing_actions([(1, "NS", "cause")], EDUS[:2], relation_types=RELATION_TYPES)]
    fields = ("task", "key", "generated", "answer")
    one_by_one = []
    for t in docs:
        parser.predict(t)
        one_by_one.append([tuple(str(r[f]) for f in fields) for r in parser.last_decisions])
        assert all(not r["fallback"] for r in parser.last_decisions)
    parser.predict_batch(docs)
    together = [[tuple(str(r[f]) for f in fields) for r in recs] for recs in parser.last_decisions_by_doc]
    assert together == one_by_one


def test_only_tasks_trains_a_subset_and_skips_parsing():
    import dataclasses

    from dipo.rst.parsers.maekawa.modeling_maekawa import MaekawaParser

    base = make_parser()
    cfg = dataclasses.replace(base.config, only_tasks=["nuc"])
    p = MaekawaParser(cfg, training=True)
    assert p.trained_tasks == ("nuc",) and p.installed_tasks == ("nuc",) and not p.can_parse


def test_exported_adapter_loads_back_as_pretrained(tmp_path):
    import dataclasses

    from dipo.rst.parsers.maekawa.modeling_maekawa import MaekawaParser
    from dipo.rst.parsers.maekawa.train_maekawa import export_adapter

    p = make_parser()
    paths = {t: export_adapter(p, t, str(tmp_path)) for t in p.tasks}
    assert set(p.tasks) <= set(os.listdir(tmp_path / "adapters"))
    q = MaekawaParser(dataclasses.replace(p.config, pretrained_adapters=paths))
    a, b = p.trainable_state_dict(), {}
    from peft import get_peft_model_state_dict

    from dipo.rst.parsers.maekawa.modeling_maekawa import adapter_name

    for t in q.tasks:
        for k, v in get_peft_model_state_dict(q.lm, adapter_name=adapter_name(t)).items():
            b[f"{t}::{k}"] = v
    assert a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


# ---- joint nuclearity + relation --------------------------------------------------


def test_joint_prompt_and_gold_decisions():
    edus = [e.strip() for e in EDUS]
    labels = Labels(["elaboration", "joint", "cause"], {"elaboration": "Elaboration"}, joint=JOINT_LABELS)
    assert labels.joint_answers == ["NS Elaboration", "NN joint", "NS cause", "SN cause"]
    assert joint_prompt(edus, (0, 1), (1, 2), labels.joint_answers) == (
        "Span1: The cat sat.\nSpan2: It was tired\n"
        "Nucleus and relation label (NS Elaboration, NN joint, NS cause, or SN cause): "
    )
    ds = gold_decisions(toy_tree(), labels, ("joint",))
    assert sorted((d.key, d.answer) for d in ds) == [
        ((0, 1, 2), "NS Elaboration"),
        ((0, 2, 3), "NS cause"),
        ((0, 3, 4), "NN joint"),
    ]
    assert joint_label_frequencies([toy_tree()]) == {("NS", "elaboration"): 1, ("NS", "cause"): 1, ("NN", "joint"): 1}
    # a gold pair outside the offered labels (unseen in training) stays a decision nobody can answer
    narrow = Labels(["elaboration", "joint", "cause"], joint=[("NS", "cause")])
    answers = {d.answer for d in gold_decisions(toy_tree(), narrow, ("joint",))}
    assert answers == {"NS elaboration", "NS cause", "NN joint"} and answers - set(narrow.joint_answers) == {
        "NS elaboration",
        "NN joint",
    }


def test_joint_config_replaces_the_two_label_adapters():
    cfg = MaekawaConfig(train_dir="x", dev_dir="y", label_mode="joint", parse_type="bottom_up")
    assert cfg.tasks == ("span", "joint")
    assert MaekawaConfig(train_dir="x", dev_dir="y", label_mode="joint", parse_type="top_down").tasks == (
        "top_down",
        "joint",
    )
    with pytest.raises(ValueError, match="label_mode"):
        MaekawaConfig(train_dir="x", dev_dir="y", label_mode="both")


@pytest.mark.parametrize("parse_type", ["bottom_up", "top_down"])
@pytest.mark.parametrize("decode", ["generate", "score"])
def test_joint_parser_predicts_trees_with_legal_labels(parse_type, decode, monkeypatch):
    parser = make_parser(parse_type, "joint")
    monkeypatch.setattr(parser.config, "decode", decode)
    assert parser.installed_tasks == ("span" if parse_type == "bottom_up" else "top_down", "joint")
    pred = parser.predict(toy_tree())
    assert len(pred.spans_with_ranges()) == len(EDUS) - 1
    joint = [r for r in parser.last_decisions if r["task"] == "joint"]
    assert len(joint) == len(EDUS) - 1 and not any(r["task"] in ("nuc", "rel_with_nuc") for r in parser.last_decisions)
    assert all(r["answer"] in parser.labels.joint_answers for r in joint)
    for _, nuc, rel in pred.spans_with_ranges():
        assert (nuc, rel) in JOINT_LABELS


def test_joint_training_loss_reaches_only_the_joint_adapter():
    p = make_parser("bottom_up", "joint")
    p.train()
    p.activate("joint")
    batch = gold_decisions(toy_tree(), p.labels, ("joint",))
    loss = p(batch)["loss"]
    loss.backward()
    grads = {n for n, w in p.lm.named_parameters() if w.grad is not None and w.grad.abs().sum() > 0}
    assert grads and all(".joint." in n for n in grads)
    p.lm.zero_grad()


# ---- unified: one adapter for structure and labels ----------------------------------

GUM_TRAIN = Path(__file__).resolve().parent.parent / "data" / "gum_12.1.0_notok" / "train"


def joint_labels() -> Labels:
    return Labels(["elaboration", "joint", "cause"], {"elaboration": "Elaboration"}, joint=JOINT_LABELS)


def test_unified_prompts():
    edus = [e.strip() for e in EDUS]
    assert unified_bu_prompt(edus, (0, 2), (2, 3), 3) == (
        "Stack2: The cat sat. It was tired\nStack1: so it slept.\nQueue1: Then it woke.\n"
        "Action (shift, or reduce with nuclearity and relation): "
    )
    assert unified_bu_prompt(edus, (0, 3), (3, 4), None) == (
        "Stack2: The cat sat. It was tired so it slept.\nStack1: Then it woke.\nQueue1:\n"
        "Action (shift, or reduce with nuclearity and relation): "
    )
    assert unified_td_prompt(EDUS, 0, 3) == (
        "Input: The cat sat. [0] It was tired [1] so it slept.\nSplit point (0–1), nuclearity and relation: "
    )
    assert (
        unified_td_prompt(EDUS, 1, 3)
        == "Input: It was tired [0] so it slept.\nSplit point (0–0), nuclearity and relation: "
    )


def _check_unified_coverage(tree: RstTree, labels: Labels) -> None:
    nodes = gold_nodes(tree)
    n = len(tree.edu_strings)
    td = gold_decisions(tree, labels, ("unified_td",))
    # top-down: one decision per internal node, 2-EDU spans included
    assert sorted((d.key[0], d.key[2]) for d in td) == sorted(nodes)
    for d in td:
        b, s, e = d.key
        nuc, rel = nodes[(b, e)][1:]
        assert d.answer == f"{s - b - 1} {labels.joint_answer(nuc, rel)}"
    # bottom-up: every reduce (= internal node) once, shifts only where a reduce was legal too
    bu = gold_decisions(tree, labels, ("unified_bu",))
    reduced = sorted((d.key[0], d.key[3]) for d in bu if d.answer.startswith("reduce "))
    assert reduced == sorted(nodes)
    for d in bu:
        if d.answer == "shift":
            assert d.key[4] != QUEUE_EMPTY
    assert sum(d.answer == "shift" for d in bu) == len(gold_decisions(tree, labels, ("span",))) - sum(
        d.answer == "reduce" for d in gold_decisions(tree, labels, ("span",))
    )
    forced = [d for d in bu if d.key[4] == QUEUE_EMPTY]
    assert forced and all(d.answer.startswith("reduce ") for d in forced)
    assert len(bu) - len(forced) == len(gold_decisions(tree, labels, ("span",)))
    assert n - 1 == len(nodes)


def test_unified_gold_decisions_label_every_node_once():
    _check_unified_coverage(toy_tree(), joint_labels())
    ds = gold_decisions(toy_tree(), joint_labels(), ("unified_bu",))
    # left-branching: reduce, reduce, shift, then the forced (empty-queue) final reduce
    assert [(d.key, d.answer) for d in ds] == [
        ((0, 1, 1, 2, 2), "reduce NS Elaboration"),
        ((0, 2, 2, 3, 3), "reduce NS cause"),
        ((0, 3, 3, 4, QUEUE_EMPTY), "reduce NN joint"),
    ]
    td = gold_decisions(toy_tree(), joint_labels(), ("unified_td",))
    assert sorted((d.key, d.answer) for d in td) == [
        ((0, 1, 2), "0 NS Elaboration"),
        ((0, 2, 3), "1 NS cause"),
        ((0, 3, 4), "2 NN joint"),
    ]


def test_unified_resolution_falls_back_part_by_part():
    lab = joint_labels()
    assert resolve_unified_bu("shift", True, lab) == ("shift", [])
    assert resolve_unified_bu("reduce SN cause", True, lab) == ("reduce SN cause", [])
    assert resolve_unified_bu("reduce SN bogus", True, lab) == ("reduce NS Elaboration", ["label"])
    assert resolve_unified_bu("reduce", False, lab) == ("reduce NS Elaboration", ["label"])
    assert resolve_unified_bu("shift", False, lab) == ("reduce NS Elaboration", ["action", "label"])
    assert resolve_unified_bu("", True, lab) == ("shift", ["action"])
    assert resolve_unified_bu("NN joint", False, lab) == ("reduce NS Elaboration", ["action", "label"])
    assert middle_split(0, 2) == 0 and middle_split(0, 5) == 1 and middle_split(3, 7) == 1
    assert resolve_unified_td("1 NN joint", 0, 3, lab) == ("1 NN joint", [])
    assert resolve_unified_td("2 NN joint", 0, 3, lab) == ("0 NN joint", ["split"])
    assert resolve_unified_td("-1 NN joint", 0, 5, lab) == ("1 NN joint", ["split"])
    assert resolve_unified_td("x", 0, 5, lab) == ("1 NS Elaboration", ["split", "label"])
    assert resolve_unified_td("0 NS joint", 0, 2, lab) == ("0 NS Elaboration", ["label"])


def test_unified_config():
    assert MaekawaConfig(train_dir="x", dev_dir="y", label_mode="unified", parse_type="bottom_up").tasks == (
        "unified_bu",
    )
    cfg = MaekawaConfig(train_dir="x", dev_dir="y", label_mode="unified", parse_type="top_down")
    assert cfg.tasks == ("unified_td",)
    with pytest.raises(ValueError, match="generate"):
        MaekawaConfig(train_dir="x", dev_dir="y", label_mode="unified", decode="score")
    with pytest.raises(ValueError, match="record_distributions"):
        MaekawaConfig(train_dir="x", dev_dir="y", label_mode="unified", record_distributions=True)


@pytest.fixture(scope="module")
def gum_train():
    if not GUM_TRAIN.is_dir():
        pytest.skip("No GUM data under data/gum_12.1.0_notok/train.")
    from dipo.rst.data.reader import infer_relation_types, read_rst_dir

    rels = infer_relation_types([str(GUM_TRAIN)])
    pairs = sorted(read_rst_dir(str(GUM_TRAIN), relation_types=rels), key=lambda p: len(p[1].edu_strings))
    trees = [t for _, t in pairs]
    picked = [trees[0], trees[len(trees) // 2], trees[-1]]  # shortest, median, longest
    pairs_seen = [p for p, _ in joint_label_frequencies(trees).most_common()]
    relations = sorted({r for _, r in pairs_seen})
    return rels, picked, Labels(relations, joint=pairs_seen)


def test_unified_gold_decisions_cover_real_gum_trees(gum_train):
    _, trees, lab = gum_train
    for t in trees:
        _check_unified_coverage(t, lab)


class _OracleStub:
    """The parse loops' state without a model, for oracle round trips."""

    def __init__(self, parse_type: str, label_mode: str, labels: Labels):
        from types import SimpleNamespace

        self.config = SimpleNamespace(parse_type=parse_type)
        self.tasks = MaekawaConfig(train_dir="x", dev_dir="y", label_mode=label_mode, parse_type=parse_type).tasks
        self.labels = labels

    def _label_steps(self, *args):
        from dipo.rst.parsers.maekawa.modeling_maekawa import MaekawaParser

        return MaekawaParser._label_steps(self, *args)


def _state(d) -> tuple:
    """A decision's parser state; a top-down split's key leaves the split out when parsing."""
    return (d.task, d.key[0], d.key[2]) if d.task in ("top_down", "unified_td") else (d.task, d.key)


@pytest.mark.parametrize("label_mode", ["separate", "joint", "unified"])
@pytest.mark.parametrize("parse_type", ["bottom_up", "top_down"])
def test_oracle_round_trip_reproduces_gold_trees(gum_train, parse_type, label_mode):
    """Run the parse loop with gold answers (looked up from gold_decisions by parser state):
    every prompt must be its gold decision's, every gold decision asked once, and the tree
    the gold tree."""
    from dipo.rst.parsers.maekawa.modeling_maekawa import MaekawaParser

    rels, trees, lab = gum_train
    stub = _OracleStub(parse_type, label_mode, lab)
    steps_fn = MaekawaParser._bottom_up_steps if parse_type == "bottom_up" else MaekawaParser._top_down_steps
    for tree in trees:
        gold = gold_decisions(tree, lab, stub.tasks)
        by_state = {_state(d): d for d in gold}
        assert len(by_state) == len(gold)
        gen = steps_fn(stub, [e.strip() for e in tree.edu_strings])
        asked = 0
        try:
            d = next(gen)
            while True:
                g = by_state[_state(d)]
                assert d.prompt == g.prompt
                asked += 1
                answer, fallbacks = g.answer, []
                if d.task == "unified_bu":
                    answer, fallbacks = resolve_unified_bu(g.answer, d.key[4] != QUEUE_EMPTY, lab)
                elif d.task == "unified_td":
                    answer, fallbacks = resolve_unified_td(g.answer, d.key[0], d.key[2], lab)
                assert answer == g.answer and not fallbacks
                d = gen.send(answer)
        except StopIteration as stop:
            nodes = stop.value
        assert asked == len(gold)
        rebuilt = tree_from_nodes(nodes, tree, rels)
        scores = evaluate_parseval([tree], [rebuilt])
        assert scores["full_f1"] == 1.0 and scores["span_f1"] == 1.0


def make_unified_parser(parse_type: str):
    import dataclasses

    from dipo.rst.parsers.maekawa.modeling_maekawa import MaekawaParser

    key = (parse_type, "unified")
    if key not in _PARSERS:
        base = make_parser()
        cfg = dataclasses.replace(
            base.config,
            parse_type=parse_type,
            label_mode="unified",
            joint_labels=JOINT_LABELS,
            record_distributions=False,
            decode="generate",
        )
        _PARSERS[key] = MaekawaParser(cfg, training=True)
    return _PARSERS[key]


def test_generation_budgets_fit_the_longest_legal_answer(monkeypatch):
    from dipo.rst.parsers.maekawa.decisions import Decision

    def n_tokens(p, text):
        return len(p.tokenizer(text, add_special_tokens=False)["input_ids"])

    p = make_unified_parser("bottom_up")
    assert p._max_new_tokens("unified_bu") >= max(n_tokens(p, f"reduce {a}") for a in p.labels.joint_answers) + 1
    wide = [Decision("unified_td", (0, -1, 120), "", "")]
    assert p._max_new_tokens("unified_td", wide) >= max(n_tokens(p, f"118 {a}") for a in p.labels.joint_answers) + 1
    sep = make_parser()
    assert sep._max_new_tokens("span") == 5
    assert sep._max_new_tokens("rel_with_nuc") >= max(n_tokens(sep, a) for a in sep.labels.answers) + 1
    monkeypatch.setattr(p.config, "decode", "score")
    with pytest.raises(ValueError, match="not enumerated"):
        p.predict(toy_tree())


def test_training_refuses_cut_answers_and_warns_on_cut_prompts(monkeypatch):
    import dataclasses

    from dipo.rst.parsers.maekawa import train_maekawa

    p = make_parser()
    ds = gold_decisions(toy_tree(), p.labels, ("rel_with_nuc",))
    train_maekawa.check_answer_lengths(p, p.config, "rel_with_nuc", ds)
    with pytest.raises(ValueError, match="max_target_tokens"):
        train_maekawa.check_answer_lengths(p, dataclasses.replace(p.config, max_target_tokens=1), "rel_with_nuc", ds)
    warned = []
    monkeypatch.setattr(train_maekawa, "warn", warned.append)
    train_maekawa.check_prompt_lengths(p, dataclasses.replace(p.config, max_source_tokens=5), "rel_with_nuc", ds)
    assert warned and "3 of 3" in warned[0]
    with pytest.raises(ValueError, match="no dev decisions"):
        train_maekawa.oracle_accuracy(p, [])


@pytest.mark.parametrize("parse_type", ["bottom_up", "top_down"])
def test_unified_parser_predicts_and_logs_fallbacks(parse_type):
    p = make_unified_parser(parse_type)
    assert p.installed_tasks == p.tasks and len(p.tasks) == 1
    docs = [toy_tree(), RstTree.from_parsing_actions([(1, "NS", "cause")], EDUS[:2], relation_types=RELATION_TYPES)]
    preds = p.predict_batch(docs)
    for doc, pred, recs in zip(docs, preds, p.last_decisions_by_doc, strict=True):
        assert pred.edu_strings == doc.edu_strings
        assert len(pred.spans_with_ranges()) == len(doc.edu_strings) - 1
        assert all(r["task"] == p.tasks[0] and "generated" in r and "fallback" in r for r in recs)
        for _, nuc, rel in pred.spans_with_ranges():
            assert (nuc, rel) in JOINT_LABELS
        if parse_type == "top_down":  # every node asked, 2-EDU spans included
            assert len(recs) == len(doc.edu_strings) - 1
        else:  # every reduce asked; a random model may shift too
            assert sum(r["answer"].startswith("reduce ") for r in recs) == len(doc.edu_strings) - 1


@pytest.mark.parametrize("parse_type", ["bottom_up", "top_down"])
def test_unified_training_loss_reaches_only_the_unified_adapter(parse_type):
    p = make_unified_parser(parse_type)
    p.train()
    p.activate(p.tasks[0])
    batch = gold_decisions(toy_tree(), p.labels, p.tasks)
    loss = p(batch)["loss"]
    loss.backward()
    grads = {n for n, w in p.lm.named_parameters() if w.grad is not None and w.grad.abs().sum() > 0}
    assert grads and all(".unified." in n for n in grads)
    p.lm.zero_grad()


def test_top_down_reuses_bottom_up_label_adapters(tmp_path):
    """Top-down with Maekawa's separate adapters, training only `top_down` and loading nuc
    and rel_with_nuc exported by a bottom-up run."""
    import dataclasses

    from peft import get_peft_model_state_dict

    from dipo.rst.parsers.maekawa.modeling_maekawa import MaekawaParser, adapter_name
    from dipo.rst.parsers.maekawa.train_maekawa import export_adapter

    bu = make_parser()
    paths = {t: export_adapter(bu, t, str(tmp_path)) for t in ("nuc", "rel_with_nuc")}
    cfg = dataclasses.replace(bu.config, parse_type="top_down", pretrained_adapters=paths)
    td = MaekawaParser(cfg, training=True)
    assert td.trained_tasks == ("top_down",) and td.can_parse
    assert set(td.trainable_state_dict()) and {k.split("::")[0] for k in td.trainable_state_dict()} == {"top_down"}
    for t in ("nuc", "rel_with_nuc"):
        want = get_peft_model_state_dict(bu.lm, adapter_name=adapter_name(t))
        got = get_peft_model_state_dict(td.lm, adapter_name=adapter_name(t))
        assert want.keys() == got.keys() and all(torch.equal(want[k].cpu(), got[k].cpu()) for k in want)
    td.train()
    td.activate("top_down")
    loss = td(gold_decisions(toy_tree(), td.labels, ("top_down",)))["loss"]
    loss.backward()
    grads = {n for n, w in td.lm.named_parameters() if w.grad is not None and w.grad.abs().sum() > 0}
    assert grads and all(".top_down." in n for n in grads)
    td.lm.zero_grad()
    pred = td.predict(toy_tree())
    assert len(pred.spans_with_ranges()) == len(EDUS) - 1
    assert {r["task"] for r in td.last_decisions} == {"top_down", "nuc", "rel_with_nuc"}


# ---- the bundled segmenter: gen's `seg` serialization as one more adapter ----------------


def make_segmenting_parser():
    import dataclasses

    from dipo.rst.parsers.maekawa.configuration_maekawa import SegmentationConfig
    from dipo.rst.parsers.maekawa.modeling_maekawa import MaekawaParser

    if "segmenting" not in _PARSERS:
        base = make_parser("top_down", "joint")
        cfg = dataclasses.replace(
            base.config, segmentation=SegmentationConfig(epochs=1, docs_per_step=2, max_input_tokens=256)
        )
        _PARSERS["segmenting"] = MaekawaParser(cfg, training=True)
    return _PARSERS["segmenting"]


def test_segmenter_is_another_adapter_on_the_same_model():
    p = make_segmenting_parser()
    assert p.installed_tasks == ("top_down", "joint", "segment") and p.segmenter is not None
    assert p.segmenter.backbone.model is p.lm
    head = p.lm.get_base_model().get_output_embeddings()
    with p.segmenter.active():
        assert p.lm.active_adapter == "segment"
        assert p.lm.get_base_model().get_output_embeddings() is p.segmenter.head
    assert p.lm.get_base_model().get_output_embeddings() is head  # restored


def test_predict_from_text_segments_then_parses():
    p = make_segmenting_parser()
    text = " ".join(e.strip() for e in EDUS)
    pred = p.predict_from_text(text)
    assert "".join(pred.edu_strings).replace(" ", "") == text.replace(" ", "")  # every token, once, in order
    assert len(pred.spans_with_ranges()) == len(pred.edus) - 1
    assert pred._pred_edu_source_ranges and pred._source_ids


def test_segmenter_state_round_trips_and_exports(tmp_path):
    import dataclasses

    from dipo.rst.parsers.maekawa.modeling_maekawa import MaekawaParser
    from dipo.rst.parsers.maekawa.train_maekawa import export_adapter

    p = make_segmenting_parser()
    state = p.trainable_state_dict()
    tasks = {k.split("::")[0] for k in state}
    assert tasks == {"top_down", "joint", "segment"}
    assert any(k.startswith("segment::head.") for k in state) and "segment::new_rows" in state
    p.load_trainable_state_dict(state)
    paths = {t: export_adapter(p, t, str(tmp_path)) for t in p.installed_tasks}
    q = MaekawaParser(dataclasses.replace(p.config, pretrained_adapters=paths))
    want = {k: v for k, v in state.items() if k.startswith("segment::")}
    got = {f"segment::{k}": v for k, v in q.segmenter.segment_state_dict().items()}
    assert want.keys() == got.keys() and all(torch.equal(want[k], got[k]) for k in want)


def test_segmenter_training_touches_only_its_own_weights():
    p = make_segmenting_parser()
    seg = p.segmenter
    with seg.active():
        seg.train()
        batch = seg.backbone.collate([seg.encode_target(toy_tree())], int(seg.tokenizer.pad_token_id))
        seg({k: v.to(p.device) for k, v in batch.items()})["loss"].backward()
    lora = {n for n, w in p.lm.named_parameters() if "lora" in n and w.grad is not None and w.grad.abs().sum() > 0}
    assert lora and all(".segment." in n for n in lora)
    assert seg.head.weight.grad is not None
    p.lm.zero_grad()
    seg.head.zero_grad()


def test_train_segmenter_and_end_to_end_evaluation():
    from dipo.rst.parsers.common.generative_eval import evaluate_on_dev
    from dipo.rst.parsers.maekawa.modeling_maekawa import EndToEnd
    from dipo.rst.parsers.maekawa.train_maekawa import train_segmenter

    p = make_segmenting_parser()
    pairs = [("toy.rs4", toy_tree())]
    scores = train_segmenter(p, p.config, [toy_tree(), toy_tree()], pairs)
    assert len(scores) == 1 and 0.0 <= scores[0] <= 1.0
    metrics = evaluate_on_dev(EndToEnd(p), pairs, batch_size=2)
    assert {"seg_f1", "e2e_span_f1", "e2e_full_f1"} <= set(metrics)


def test_parser_without_segmenter_refuses_text():
    with pytest.raises(RuntimeError, match="no segmenter"):
        make_parser().predict_from_text("Some text.")
