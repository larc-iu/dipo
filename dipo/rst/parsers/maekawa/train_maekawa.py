"""Train the Maekawa et al. (2024) parser: each LoRA adapter on its own prompts for `epochs`
epochs, keeping the epoch with the best dev decision accuracy under gold parser states
(their test_with_oracle.py), then the bundled segmenter (if any), keeping its best dev
segmentation F1. Then parse dev and test with gold EDUs, and end to end from raw text when
there is a segmenter, and write final metrics.

Tasks listed in `pretrained_adapters` are loaded, not trained, so a config that names
every task only evaluates (e.g. their released RST-DT adapters).

Resuming restarts an interrupted task from scratch, keeping the adapters already done.
"""

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import time

import torch
from tonga import Params

from dipo.common.log import console, dim, rule, setup_logging, success, warn, wrote
from dipo.common.training import (
    config_panel,
    gpu_mem_gb,
    load_model_state,
    prepare_run_dir,
    save_checkpoint,
    set_seeds,
    write_run_config,
)
from dipo.rst import HASH_EXCLUDE
from dipo.rst.data.metrics import evaluate_parseval, metrics_table
from dipo.rst.data.reader import infer_relation_types, read_rst_dir
from dipo.rst.parsers.maekawa.configuration_maekawa import MaekawaConfig
from dipo.rst.parsers.maekawa.decisions import Labels, gold_decisions, joint_label_frequencies, relation_frequencies
from dipo.rst.parsers.maekawa.modeling_maekawa import SEGMENTER_FILE, EndToEnd, MaekawaParser, adapter_name

setup_logging()
logger = logging.getLogger(__name__)

CHECKPOINT = "best_model.pt"


def _save(path: str, model: MaekawaParser, cfg_dict: dict, completed: list[str], dev_accuracy: dict) -> None:
    save_checkpoint(
        path,
        model,
        None,
        None,
        trainable_only=True,
        config=cfg_dict,
        parser_kind="maekawa",
        completed_tasks=completed,
        dev_accuracy=dev_accuracy,
    )


def oracle_accuracy(model: MaekawaParser, decisions: list) -> float:
    """Decision accuracy under gold parser states. The decisions are independent, so they
    run as inference batches."""
    if not decisions:
        raise ValueError("no dev decisions to select epochs on (dev_dir empty, or dev_max_docs too small)")
    model.eval()
    model.last_decisions = []
    answers = model.decide_batch(decisions)
    return sum(a == d.answer for a, d in zip(answers, decisions, strict=True)) / len(decisions)


def check_answer_lengths(model: MaekawaParser, cfg: MaekawaConfig, task: str, decisions: list) -> None:
    """Training truncates answers (+ EOS) to max_target_tokens; a cut answer loses its EOS
    and teaches the adapter never to stop, so refuse to train on one."""
    longest = max(decisions, key=lambda d: len(model._answer_ids(d.answer)))
    n = len(model._answer_ids(longest.answer))
    if n > cfg.max_target_tokens:
        raise ValueError(
            f"{task}: the answer {longest.answer!r} is {n} tokens with EOS, over max_target_tokens="
            f"{cfg.max_target_tokens}; raise max_target_tokens"
        )
    dim(f"{task}: longest gold answer {n} tokens with EOS (max_target_tokens={cfg.max_target_tokens})")


def check_prompt_lengths(model: MaekawaParser, cfg: MaekawaConfig, task: str, decisions: list) -> None:
    """Training keeps the first max_source_tokens tokens of a prompt (theirs), which cuts
    off its final question line; inference never truncates. Warn when that happens."""
    lengths = [len(model._prompt_ids(d.prompt)) for d in decisions]
    over = sum(n > cfg.max_source_tokens for n in lengths)
    if over:
        warn(
            f"{task}: {over} of {len(decisions)} training prompts are longer than max_source_tokens="
            f"{cfg.max_source_tokens} (longest {max(lengths)}) and lose their end in training"
        )


def optimizer_steps(model: MaekawaParser, cfg: MaekawaConfig, decisions: list, rng: random.Random) -> list[list[list]]:
    """One epoch as optimizer steps, each a list of micro-batches covering exactly
    batch_size x grad_accum decisions (the last step may be short).

    With batch_size=1 the order is a plain shuffle (their train.sh: group_by_length is a
    no-op at batch size 1). Otherwise steps hold decisions of similar length, like
    transformers' group_by_length: shuffle, cut into megabatches of 50 steps, sort each by
    length, cut into steps, shuffle the steps. Each step is then split into micro-batches
    of at most `batch_size` decisions and `max_microbatch_tokens` padded tokens, longest
    first. The loss is summed per decision and divided by the step's decision count, so
    the split does not change what is optimized."""
    per_step = cfg.batch_size * cfg.grad_accum
    order = list(decisions)
    rng.shuffle(order)
    if cfg.batch_size == 1:
        return [[[d] for d in order[i : i + per_step]] for i in range(0, len(order), per_step)]
    length = {
        id(d): len(model._prompt_ids(d.prompt, truncate=True)) + len(model._answer_ids(d.answer, truncate=True))
        for d in order
    }
    steps = []
    for i in range(0, len(order), 50 * per_step):
        chunk = sorted(order[i : i + 50 * per_step], key=lambda d: length[id(d)], reverse=True)
        steps += [chunk[j : j + per_step] for j in range(0, len(chunk), per_step)]
    rng.shuffle(steps)
    out = []
    for step in steps:
        micro, cur = [], []
        for d in step:  # already longest first
            longest = length[id(cur[0])] if cur else length[id(d)]
            over_tokens = cfg.max_microbatch_tokens is not None and longest * (len(cur) + 1) > cfg.max_microbatch_tokens
            if cur and (len(cur) == cfg.batch_size or over_tokens):
                micro.append(cur)
                cur = []
            cur.append(d)
        micro.append(cur)
        out.append(micro)
    return out


def train_task(model: MaekawaParser, cfg: MaekawaConfig, task: str, train: list, dev: list) -> list[float]:
    """Train one adapter, leaving its best epoch's weights in place. Returns the dev
    accuracy of every epoch."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import get_cosine_schedule_with_warmup

    model.activate(task)
    name = adapter_name(task)
    params = [p for n, p in model.lm.named_parameters() if p.requires_grad and f".{name}." in n]
    if not params:
        raise RuntimeError(f"adapter {task!r} has no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train) / (cfg.batch_size * cfg.grad_accum))
    total = steps_per_epoch * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(cfg.warmup_ratio * total), total)
    dim(f"{task}: {len(train)} train / {len(dev)} dev decisions, {steps_per_epoch} steps/epoch, {total} steps")

    rng = random.Random(cfg.seed)
    device = model.device
    best_state, best_acc, step, accuracies = None, -1.0, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running = time.monotonic(), []
        for micro_batches in optimizer_steps(model, cfg, train, rng):
            n = sum(len(m) for m in micro_batches)
            step_loss = 0.0
            for batch in micro_batches:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(batch)["per_decision"].sum() / n
                loss.backward()
                step_loss += loss.item()
            running.append(step_loss)
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % cfg.log_every == 0:
                mem = gpu_mem_gb(device)
                console.print(
                    f"  [step]{task} step {step}/{total}[/step] loss=[loss]{sum(running) / len(running):.4f}[/loss] "
                    f"lr=[dim]{scheduler.get_last_lr()[0]:.2e}[/dim]"
                    + (f" mem=[dim]{mem[1]:.1f}GB[/dim]" if mem else "")
                )
                running = []
        acc = oracle_accuracy(model, dev)
        accuracies.append(acc)
        console.print(
            f"  [epoch]{task} epoch {epoch}/{cfg.epochs}[/epoch] ({time.monotonic() - t0:.0f}s) dev accuracy {acc:.4f}"
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model.lm, adapter_name=name).items()
            }
            success(f"  New best {task} dev accuracy {acc:.4f}")
    set_peft_model_state_dict(model.lm, best_state, adapter_name=name)
    return accuracies


def _segmenter_micro_batches(seg, docs: list, cap: int | None) -> list[list]:
    """A step's training examples (longest first) as micro-batches under `cap` padded tokens."""
    docs = sorted(docs, key=lambda ex: -len(ex[0]))
    out, cur = [], []
    for ex in docs:
        if cur and cap is not None and len(cur[0][0]) * (len(cur) + 1) > cap:
            out.append(cur)
            cur = []
        cur.append(ex)
    return out + [cur]


def train_segmenter(model: MaekawaParser, cfg: MaekawaConfig, train_trees: list, dev_pairs: list) -> list[float]:
    """Train the segment adapter, its head and the segmenter's new token rows with gen's
    loss, leaving the epoch with the best dev segmentation F1 in place. Returns the dev
    F1 of every epoch."""
    from transformers import get_cosine_schedule_with_warmup

    from dipo.rst.parsers.common.generative_eval import evaluate_on_dev

    seg, scfg = model.segmenter, cfg.segmentation
    if not dev_pairs:
        raise ValueError("no dev documents to select the segmenter on")
    with seg.active():
        seg.configure_new_row_training()
        examples = [ex for ex in (seg.encode_target(t) for t in train_trees) if ex is not None]
        if len(examples) < len(train_trees):
            warn(f"segment: {len(train_trees) - len(examples)} training documents exceed max_input_tokens, skipped")
        lora = [p for n, p in model.lm.named_parameters() if p.requires_grad and f".{adapter_name('segment')}." in n]
        params = lora + list(seg.head.parameters()) + seg.new_row_params()
        optimizer = torch.optim.AdamW(params, lr=scfg.lr, betas=(0.9, cfg.adam_beta2), weight_decay=cfg.weight_decay)
        steps_per_epoch = math.ceil(len(examples) / scfg.docs_per_step)
        total = steps_per_epoch * scfg.epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(cfg.warmup_ratio * total), total)
        dim(f"segment: {len(examples)} train / {len(dev_pairs)} dev documents, {steps_per_epoch} steps/epoch")
        rng = random.Random(cfg.seed)
        device = model.device
        pad = int(seg.tokenizer.pad_token_id)
        best_state, best_f1, scores = None, -1.0, []
        for epoch in range(1, scfg.epochs + 1):
            seg.train()
            t0, order = time.monotonic(), list(examples)
            rng.shuffle(order)
            for start in range(0, len(order), scfg.docs_per_step):
                step = order[start : start + scfg.docs_per_step]
                for micro in _segmenter_micro_batches(seg, step, cfg.max_microbatch_tokens):
                    batch = {k: v.to(device) for k, v in seg.backbone.collate(micro, pad).items()}
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        loss = seg(batch)["loss"] * len(micro) / len(step)
                    loss.backward()
                seg.stage_new_embedding_row_grads()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                seg.commit_new_embedding_rows()
            f1 = evaluate_on_dev(seg, dev_pairs, batch_size=scfg.inference_batch_size)["seg_f1"]
            scores.append(f1)
            console.print(
                f"  [epoch]segment epoch {epoch}/{scfg.epochs}[/epoch] ({time.monotonic() - t0:.0f}s) dev seg F1 {f1:.4f}"
            )
            if f1 > best_f1:
                best_f1, best_state = f1, seg.segment_state_dict()
                success(f"  New best segment dev F1 {f1:.4f}")
        seg.load_segment_state_dict(best_state)
    return scores


def evaluate(model: MaekawaParser, pairs: list, out_dir: str) -> dict[str, float]:
    """Parse every document (lockstep batches) with the EDUs it carries and score with gold."""
    os.makedirs(out_dir, exist_ok=True)
    golds = [gold for _, gold in pairs]
    preds = model.predict_batch(golds)
    decisions = []
    for (path, _), pred, records in zip(pairs, preds, model.last_decisions_by_doc, strict=True):
        doc_id = os.path.basename(path).split(".")[0]
        with open(os.path.join(out_dir, doc_id + ".rs4"), "w", encoding="utf-8") as f:
            f.write(pred.to_rs4_string())
        decisions += [{"doc_id": doc_id, "step": i, **r} for i, r in enumerate(records)]
    path = os.path.join(out_dir, "decisions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in decisions)
    wrote(path)
    return evaluate_parseval(golds, preds)


def export_adapter(model: MaekawaParser, task: str, run_dir: str) -> str:
    """Save one trained adapter as a standard PEFT dir, loadable via `pretrained_adapters`
    (the segmenter's also carries its head and new token rows in `segmenter.pt`)."""
    out = os.path.join(run_dir, "adapters")
    model.lm.save_pretrained(out, selected_adapters=[adapter_name(task)])
    final = os.path.join(out, task)
    src = os.path.join(out, adapter_name(task))
    if src != final:
        if os.path.exists(final):
            import shutil

            shutil.rmtree(final)
        os.replace(src, final)
    if task == "segment":
        extras = {k: v for k, v in model.segmenter.segment_state_dict().items() if not k.startswith("lora.")}
        torch.save(extras, os.path.join(final, SEGMENTER_FILE))
    wrote(final)
    return final


def train(cfg: MaekawaConfig) -> None:
    set_seeds(cfg.seed)
    cfg.relation_types = infer_relation_types([cfg.train_dir, cfg.dev_dir], relation_map=cfg.relation_map)
    read = lambda d: read_rst_dir(d, relation_types=cfg.relation_types, relation_map=cfg.relation_map)  # noqa: E731
    train_trees = [t for _, t in read(cfg.train_dir)]
    dev_pairs = read(cfg.dev_dir)
    test_pairs = read(cfg.test_dir) if cfg.test_dir else None
    if cfg.relation_labels is None:
        cfg.relation_labels = [r for r, _ in relation_frequencies(train_trees).most_common()]
        dim(f"Relations offered in prompts, most frequent first: {cfg.relation_labels}")
    if cfg.label_mode in ("joint", "unified") and cfg.joint_labels is None:
        cfg.joint_labels = [pair for pair, _ in joint_label_frequencies(train_trees).most_common()]
        dim(f"{len(cfg.joint_labels)} joint labels offered in prompts, most frequent first")
    run_dir, _ = prepare_run_dir(dataclasses.asdict(cfg), cfg.checkpoint_dir, cfg.run_name, hash_exclude=HASH_EXCLUDE)
    cfg_dict = dataclasses.asdict(cfg)
    write_run_config(run_dir, cfg_dict)
    console.print(config_panel(cfg_dict))

    model = MaekawaParser(cfg, training=bool(set(cfg.all_tasks) - set(cfg.pretrained_adapters or {})))
    labels = Labels(cfg.relation_labels, cfg.relation_surface, cfg.default_relation, cfg.joint_labels)
    ckpt_path = os.path.join(run_dir, CHECKPOINT)
    completed: list[str] = []
    dev_accuracy: dict[str, float] = {}
    dev_by_epoch: dict[str, list[float]] = {}
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_model_state(model, ckpt)
        completed, dev_accuracy = ckpt["completed_tasks"], ckpt["dev_accuracy"]
        dim(f"Resuming: adapters already trained: {completed}")

    dev_trees = [t for _, t in dev_pairs][: cfg.dev_max_docs]
    for task in model.trained_tasks:
        if task in completed:
            continue
        rule(f"Training adapter: {task}")
        if task == "segment":
            by_epoch = train_segmenter(model, cfg, train_trees, dev_pairs[: cfg.dev_max_docs])
            dev_accuracy[task], dev_by_epoch[task] = max(by_epoch), by_epoch
            completed.append(task)
            _save(ckpt_path, model, cfg_dict, completed, dev_accuracy)
            export_adapter(model, task, run_dir)
            continue
        train_d = [d for t in train_trees for d in gold_decisions(t, labels, (task,))]
        dev_d = [d for t in dev_trees for d in gold_decisions(t, labels, (task,))]
        check_answer_lengths(model, cfg, task, train_d)
        check_prompt_lengths(model, cfg, task, train_d)
        by_epoch = train_task(model, cfg, task, train_d, dev_d)
        dev_accuracy[task] = max(by_epoch)
        dev_by_epoch[task] = by_epoch
        completed.append(task)
        _save(ckpt_path, model, cfg_dict, completed, dev_accuracy)
        export_adapter(model, task, run_dir)

    if not model.can_parse:
        missing = sorted(set(cfg.tasks) - set(model.installed_tasks))
        success(f"Trained {list(model.trained_tasks)}; skipping final evaluation (no adapters for {missing}).")
        return
    rule("Final Evaluation")
    final = {}
    for split, pairs in (("dev", dev_pairs), ("test", test_pairs)):
        if pairs is None:
            continue
        final[split] = evaluate(model, pairs, os.path.join(run_dir, f"{split}_predictions", "final"))
        if model.segmenter is not None:
            from dipo.rst.parsers.common.generative_eval import evaluate_on_dev

            e2e = evaluate_on_dev(
                EndToEnd(model),
                pairs,
                batch_size=cfg.segmentation.inference_batch_size,
                output_dir=os.path.join(run_dir, f"{split}_predictions", "final_e2e"),
            )
            final[split].update({k: v for k, v in e2e.items() if k.startswith(("seg_", "e2e_"))})
        console.print(metrics_table(final[split], title=f"Final {split.capitalize()} Results"))
    final["dev_decision_accuracy"] = dev_accuracy
    if dev_by_epoch:  # tasks trained in this process (a resumed run lacks the earlier ones)
        final["dev_decision_accuracy_by_epoch"] = dev_by_epoch
    path = os.path.join(run_dir, "final_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    wrote(path)
    if not os.path.exists(ckpt_path):  # all adapters pretrained: still leave a loadable run
        _save(ckpt_path, model, cfg_dict, completed, dev_accuracy)


def main():
    parser = argparse.ArgumentParser(description="Train the maekawa parser")
    parser.add_argument("config", help="Path to a jsonnet config file")
    args = parser.parse_args()
    train(MaekawaConfig.from_dict(Params.from_file(args.config).as_dict(quiet=True)))


if __name__ == "__main__":
    main()
