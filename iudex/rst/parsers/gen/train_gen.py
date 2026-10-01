"""Train the unified generative parser `gen`.

One training loop for every backbone x serialization combo. The per-backbone
differences (single causal stream vs encoder/decoder split) are confined to
`backbone.collate`; new-token embedding rows train via the new-row shadow
(`GenParser.configure_new_row_training`), which the loop stages/commits around
each optimizer step over `model.optimizer_parameters()`. Evaluation goes
through the shared `evaluate_on_dev`. Curriculum phases, grad-accum, per-epoch
validation + early stopping, final dev/test eval.
"""

import argparse
import dataclasses
import json
import logging
import math
import os
import time
from collections import deque

import torch
from tonga import Params
from torch.utils.data import DataLoader

from iudex.common.log import console, dim, rule, setup_logging, success, warn, wrote
from iudex.common.training import (
    TBLogger,
    config_panel,
    device_panel,
    edu_count_loss_weights,
    gpu_mem_gb,
    install_abort_handler,
    load_model_state,
    make_progress_bar,
    make_scheduler,
    make_wsd_scheduler,
    model_panel,
    prepare_run_dir,
    resume_or_init,
    save_checkpoint,
    schedule_panel,
    set_seeds,
    write_run_config,
)
from iudex.rst import HASH_EXCLUDE
from iudex.rst.data.metrics import metrics_table
from iudex.rst.data.reader import infer_relation_types, read_rst_dir
from iudex.rst.parsers.common.generative_eval import evaluate_on_dev
from iudex.rst.parsers.gen.configuration_gen import GenConfig
from iudex.rst.parsers.gen.modeling_gen import GenParser

setup_logging()
logger = logging.getLogger(__name__)


def _build_dataset(model: GenParser, trees) -> tuple[list[dict], int]:
    """Encode each tree once, returning the item list a DataLoader consumes
    directly (a plain list is a valid map-style dataset). The encoded example is
    the backbone's `pack_example` output (a (input_ids, labels) tuple for
    decoder_only, a dict for seq2seq); `backbone.collate` batches whichever
    shape. Overflowing trees encode to None."""
    items: list[dict] = []
    dropped = 0
    for tree in trees:
        encoded = model.encode_target(tree)
        if encoded is None:
            dropped += 1
            continue
        items.append({"example": encoded, "n_edus": len(tree.edus)})
    return items, dropped


def _build_optimizer(model: GenParser, cfg: GenConfig):
    params = model.optimizer_parameters()
    if cfg.new_row_lr is not None:
        # Split the new-row shadow params into their own group at new_row_lr (no weight
        # decay: these are embeddings we specifically want to grow from init). The LR
        # schedule scales each group's base lr, so both warm up and decay together.
        shadows = model.new_row_params()
        if shadows:
            sid = {id(p) for p in shadows}
            base = [p for p in params if id(p) not in sid]
            params = [{"params": base}, {"params": shadows, "lr": cfg.new_row_lr, "weight_decay": 0.0}]
    if cfg.optimizer == "adafactor":
        from transformers import Adafactor

        return Adafactor(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            scale_parameter=False, relative_step=False, warmup_init=False,
        )
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}. Expected 'adamw' or 'adafactor'.")


def _make_collator(model: GenParser, pad_id: int, weight_table: dict[int, float] | None):
    """`(model_batch, meta)`; `meta["weights"]` is the per-document EDU-count loss
    weight (all ones without `edu_loss_weight_exponent`). Padding is backbone-owned."""

    def collate(batch: list[dict]):
        model_batch = model.backbone.collate([b["example"] for b in batch], pad_id)
        if weight_table is None:
            weights = torch.ones(len(batch), dtype=torch.float)
        else:
            weights = torch.tensor([weight_table.get(b["n_edus"], 1.0) for b in batch], dtype=torch.float)
        return model_batch, {"weights": weights}

    return collate


def _weighted_document_loss(loss_per_example: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Apply each per-document weight before reducing the batch."""
    if loss_per_example.ndim != 1 or weights.ndim != 1 or loss_per_example.shape != weights.shape:
        raise ValueError(
            f"Expected matching 1-D document losses and weights, got "
            f"{tuple(loss_per_example.shape)} and {tuple(weights.shape)}."
        )
    return (loss_per_example * weights.to(device=loss_per_example.device, dtype=loss_per_example.dtype)).mean()


def _normalize_window_grads(params, window_docs: int) -> None:
    """Turn an accumulation window's summed gradient into its mean, in place.

    The loop backwards the weighted SUM of each batch and counts documents, so the
    window's gradient is normalized by the documents it actually accumulated. Dividing
    each batch by `cfg.grad_accum` instead would under-scale the epoch's short final
    window (`len(loader) % grad_accum`, which steps early) by `window/grad_accum`
    while the scheduler still counted a full step, and would weight documents in
    uneven batches unequally.

    Call AFTER `stage_new_embedding_row_grads` (so the full embedding's freed grad is
    never touched; staging is a linear slice, so scaling the shadow is equivalent) and
    BEFORE clipping (so the norm sees the true mean-scale gradient).
    """
    if window_docs <= 1:
        return
    inv = 1.0 / window_docs
    for p in params:
        if p.grad is not None:
            p.grad.mul_(inv)


def train(cfg: GenConfig) -> None:
    if cfg.width_band_loss is not None:
        raise NotImplementedError(
            "width_band_loss is not yet supported in gen training (the shared loss_terms has no "
            "per-position weighting). Unset it, or wire per-position weights into loss_terms first."
        )
    set_seeds(cfg.seed)

    run_dir, cfg_hash = prepare_run_dir(
        dataclasses.asdict(cfg), cfg.checkpoint_dir, cfg.run_name, hash_exclude=HASH_EXCLUDE
    )

    if cfg.relation_map is not None:
        dim(f"Applying `relation_map` ({len(cfg.relation_map)} entries) to all read trees.")
    cfg.relation_types = infer_relation_types([cfg.train_dir, cfg.dev_dir], relation_map=cfg.relation_map)
    dim(
        f"Inferred {len(cfg.relation_types)} (relation, kind) pairs from "
        f"{cfg.train_dir} + {cfg.dev_dir}" + (" (after relation_map)." if cfg.relation_map is not None else ".")
    )

    cfg_dict = dataclasses.asdict(cfg)
    write_run_config(run_dir, cfg_dict)
    tb = TBLogger(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = cfg.amp and device.type == "cuda"
    model = GenParser(cfg).to(device)
    masked = model.configure_new_row_training()
    if masked is not None:
        n_total, n_new = masked
        lr_note = f" at new_row_lr={cfg.new_row_lr:.1e}" if cfg.new_row_lr is not None else f" at lr={cfg.lr:.1e}"
        dim(f"Training only the {n_new} new token rows (of {n_total}) via the new-row shadow{lr_note}; pretrained rows frozen.")

    train_pairs = read_rst_dir(cfg.train_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    dev_pairs = read_rst_dir(cfg.dev_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    test_pairs = (
        read_rst_dir(cfg.test_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
        if cfg.test_dir is not None
        else None
    )

    train_trees = [t for _, t in train_pairs]
    phases = cfg.curriculum.plan()
    total_epochs = sum(p.epochs for p in phases)

    pad_id = model.backbone.tokenizer.pad_token_id
    rng_seed = torch.Generator()
    rng_seed.manual_seed(cfg.seed)

    # Build each phase's loader up front so the LR schedule spans the whole run
    # (total_steps summed across phases). SimpleCurriculum yields a single full
    # phase == the pre-curriculum loop.
    phase_loaders: list[tuple] = []
    total_steps = 0
    for phase in phases:
        phase_trees = cfg.curriculum.train_trees(train_trees, phase)
        ds, dropped = _build_dataset(model, phase_trees)
        if dropped > 0:
            raise ValueError(
                f"[phase cap={phase.cap}] {dropped}/{len(phase_trees)} training trees do not fit "
                f"(source > max_input_length={cfg.max_input_length}, target > max_output_length="
                f"{cfg.max_output_length}, or the combined stream over the model's positional cap; "
                f"see per-tree warnings). Silently dropping the longest documents corrupts the run, "
                f"so this is a hard error: raise the length caps or remove them on purpose."
            )
        wtab = (
            edu_count_loss_weights([it["n_edus"] for it in ds], exponent=cfg.edu_loss_weight_exponent)
            if cfg.edu_loss_weight_exponent
            else None
        )
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=_make_collator(model, pad_id, wtab), generator=rng_seed,
        )
        spe = max(1, math.ceil(len(loader) / cfg.grad_accum))
        phase_loaders.append((phase, loader, spe))
        total_steps += spe * phase.epochs

    warmup = cfg.num_warmup_steps if cfg.num_warmup_steps is not None else min(200, max(1, int(0.1 * total_steps)))

    # WSD schedule boundaries: hold peak across the subtree phases, re-warmup into
    # the full-document phase, then decay to `min_lr_frac` over the rest of it, so
    # the whole anneal lands inside the validated final phase. The final phase is the
    # full-document (cap=None) phase (curriculum validator enforces this).
    final_phase, _, spe_final = phase_loaders[-1]
    fulldoc_steps = spe_final * final_phase.epochs
    subtree_steps = total_steps - fulldoc_steps
    if subtree_steps > warmup:
        hold_end = subtree_steps
        fdwu = min(cfg.fulldoc_warmup_steps, max(1, fulldoc_steps // 4))
        decay_start = hold_end + fdwu
    else:
        # single-phase (SimpleCurriculum): no hold/re-warmup, just warmup -> decay.
        hold_end = decay_start = warmup
    decay_end = total_steps

    if len(phases) > 1:
        dim(
            "Curriculum phases (cap/epochs/examples): "
            + ", ".join(
                f"{p.cap if p.cap is not None else 'full'}/{p.epochs}/{len(loader.dataset)}"
                for p, loader, _ in phase_loaders
            )
        )

    console.print(config_panel(cfg_dict))
    console.print(device_panel(device, seed=cfg.seed, checkpoint_dir=run_dir))
    console.print(model_panel(model, num_train_trees=len(train_trees), grad_accum=cfg.grad_accum))
    console.print(
        schedule_panel(
            steps_per_epoch=phase_loaders[0][2], total_steps=total_steps,
            warmup_steps=warmup, lr=cfg.lr, encoder_lr=None,
        )
    )

    optimizer = _build_optimizer(model, cfg)
    scheduler = make_wsd_scheduler(optimizer, warmup, hold_end, decay_start, decay_end, cfg.min_lr_frac)

    resumed = resume_or_init(run_dir, model=model, optimizer=optimizer, scheduler=scheduler, expected_hash=cfg_hash)
    global_step = resumed["global_step"]
    start_epoch = resumed["epoch"]
    best_val = resumed["best_val"]
    stale = resumed["stale_validations"]
    # Noise-robust stop signal: trailing mean of recent dev scores + its best.
    # Not persisted across resume (window refills in <= patience_window epochs;
    # resume breakage is acceptable per project policy).
    recent_dev: deque = deque(maxlen=cfg.patience_window)
    best_smoothed = -1.0

    def _save(path: str, epoch: int) -> None:
        save_checkpoint(
            path, model, optimizer, scheduler,
            trainable_only=cfg.checkpoint_trainable_only, config=cfg_dict, config_hash=cfg_hash,
            global_step=global_step, epoch=epoch, best_val=best_val, stale_validations=stale, parser_kind="gen",
        )

    dev_beams = 1 if cfg.eval_decode_greedy else cfg.num_beams

    def _validate(epoch: int, epoch_in_phase: int, dev_set: list) -> None:
        nonlocal best_val, stale, best_smoothed
        if epoch_in_phase < cfg.begin_validation_epoch or not dev_set:
            return
        if epoch % cfg.validate_every != 0 and epoch != total_epochs:
            return
        per_epoch_dev = dev_set if cfg.dev_max_docs is None else dev_set[: cfg.dev_max_docs]
        pred_dir = os.path.join(run_dir, "dev_predictions", f"epoch{epoch}_step{global_step}")
        metrics = evaluate_on_dev(
            model, per_epoch_dev, num_beams=dev_beams, batch_size=cfg.dev_batch_size, output_dir=pred_dir,
        )
        tb.log_scalars("dev", metrics, global_step)
        console.print(metrics_table(metrics, title=f"Dev @ step {global_step}"))
        if cfg.val_metric_name not in metrics:
            raise KeyError(
                f"val_metric_name={cfg.val_metric_name!r} not in dev metrics (have: {sorted(metrics)})."
            )
        score = metrics[cfg.val_metric_name]
        # Checkpoint SELECTION: raw argmax on the metric (best_model.pt), unchanged.
        if score > best_val:
            best_val = score
            _save(os.path.join(run_dir, "best_model.pt"), epoch)
            success(f"  New best! {cfg.val_metric_name}={best_val:.4f}")
        # STOPPING: trailing mean of the last `patience_window` scores, so patience
        # counts real plateaus rather than per-epoch noise. `stale` only advances
        # once the window is full.
        recent_dev.append(score)
        if len(recent_dev) < cfg.patience_window:
            dim(f"  Stop-window filling ({len(recent_dev)}/{cfg.patience_window})")
        else:
            smoothed = sum(recent_dev) / len(recent_dev)
            if smoothed > best_smoothed:
                best_smoothed = smoothed
                stale = 0
            else:
                stale += 1
            dim(f"  Smoothed {smoothed:.4f} (best {best_smoothed:.4f}, stale {stale}/{cfg.patience})")
        model.train()

    aborted = install_abort_handler()
    training_complete = start_epoch >= total_epochs or stale >= cfg.patience
    if training_complete:
        reason = "all epochs completed" if start_epoch >= total_epochs else "patience exhausted"
        dim(f"Skipping training: {reason} on prior run. Jumping to final evaluation.")

    recent_losses: deque = deque(maxlen=200)
    recent_action_losses: deque = deque(maxlen=200)
    recent_copy_losses: deque = deque(maxlen=200)
    if not training_complete:
        rule("Training")
    training_start = time.monotonic()

    phase_start = 0
    for phase_idx, (phase, loader, spe) in enumerate(phase_loaders):
        p_start, p_end = phase_start, phase_start + phase.epochs
        phase_start = p_end
        if stale >= cfg.patience or aborted.value:
            break
        if start_epoch >= p_end:
            continue
        dev_set = cfg.curriculum.dev_pairs(dev_pairs, phase)
        cap_desc = "full documents" if phase.cap is None else f"subtrees <= {phase.cap} EDUs"
        rule(
            f"Curriculum phase {phase_idx + 1}/{len(phase_loaders)}: {cap_desc} | "
            f"epochs {p_start + 1}-{p_end} | {len(loader.dataset)} examples | "
            f"{'validating' if dev_set else 'no dev (warmup)'}"
        )
        if dev_set and max(start_epoch, p_start) == p_start:
            best_val, stale = -1.0, 0
            best_smoothed = -1.0
            recent_dev.clear()
        n_batches_total = len(loader)

        for epoch in range(max(start_epoch, p_start), p_end):
            if stale >= cfg.patience or aborted.value:
                break
            epoch_start = time.monotonic()
            model.train()
            total_loss = 0.0
            num_batches = 0
            epoch_step = 0
            window_docs = 0  # documents accumulated since the last optimizer step

            with make_progress_bar() as progress:
                task = progress.add_task(
                    "training", total=spe, epoch=f"{epoch + 1}/{total_epochs}",
                    loss_str="loss=-.----", lr_str="", mem_str="", total_elapsed="0:00:00",
                )

                for batch_idx, (batch, meta) in enumerate(loader):
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                        out = model(batch)
                    n_docs = int(out["loss_per_example"].numel())
                    loss = _weighted_document_loss(out["loss_per_example"], meta["weights"])
                    raw_loss = float(loss.item())
                    # Backward the weighted SUM (mean * n is exactly the sum) and count
                    # the documents; the window's gradient is normalized by the real
                    # document count at the step below. Dividing by cfg.grad_accum here
                    # would under-scale a short final window (len(loader) % grad_accum)
                    # and would mis-weight uneven batches. grad_accum only decides WHEN
                    # to step.
                    (loss * n_docs).backward()
                    window_docs += n_docs
                    recent_losses.append(raw_loss)
                    if "action_loss" in out:
                        recent_action_losses.append(float(out["action_loss"].item()))
                        recent_copy_losses.append(float(out["copy_loss"].item()))
                    total_loss += raw_loss
                    num_batches += 1

                    is_step = (batch_idx + 1) % cfg.grad_accum == 0 or (batch_idx + 1) == n_batches_total
                    if not is_step:
                        continue

                    # New-row shadow: slice the embedding's new-row grad onto the
                    # shadow and free the full grad, clip/step over optimizer_parameters
                    # (which swaps the embedding for the shadow), write the stepped
                    # shadow back into the live rows. No-op without new tokens.
                    model.stage_new_embedding_row_grads()
                    step_params = model.optimizer_parameters()
                    _normalize_window_grads(step_params, window_docs)
                    grad_norm = torch.nn.utils.clip_grad_norm_(step_params, cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    window_docs = 0
                    model.commit_new_embedding_rows()
                    scheduler.step()
                    global_step += 1
                    epoch_step += 1

                    avg_loss = sum(recent_losses) / len(recent_losses)
                    lr_display = "/".join(f"{lr:.1e}" for lr in sorted(set(scheduler.get_last_lr())))
                    mem = gpu_mem_gb(device)
                    mem_str = f"[gpu]max_mem={mem[1]:.1f}GB[/gpu]" if mem else ""
                    secs = int(time.monotonic() - training_start)
                    progress.update(
                        task, advance=1,
                        loss_str=f"loss=[bold orange1]{avg_loss:.4f}[/bold orange1]",
                        lr_str=f"lr=[dim]{lr_display}[/dim]", mem_str=mem_str,
                        total_elapsed=f"{secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d}",
                    )

                    if epoch_step % cfg.log_every == 0:
                        tb_train = {"loss": avg_loss, "lr": max(scheduler.get_last_lr()), "grad_norm": float(grad_norm)}
                        if recent_action_losses:
                            tb_train["action_loss"] = sum(recent_action_losses) / len(recent_action_losses)
                            tb_train["copy_loss"] = sum(recent_copy_losses) / len(recent_copy_losses)
                        if mem:
                            tb_train["gpu_mem_gb"] = mem[1]
                        tb.log_scalars("train", tb_train, global_step)

            if num_batches > 0:
                console.print(
                    f"  [epoch]Epoch {epoch + 1}/{total_epochs}[/epoch] "
                    f"[dim]({time.monotonic() - epoch_start:.1f}s)[/dim]  "
                    f"loss=[loss]{total_loss / num_batches:.4f}[/loss]"
                )
                _validate(epoch + 1, epoch + 1 - p_start, dev_set)
                _save(os.path.join(run_dir, "last.pt"), epoch + 1)
                if stale >= cfg.patience or aborted.value:
                    warn(f"\nEarly stopping after {cfg.patience} validations without improvement")
                    break

    rule("Final Evaluation")
    best_path = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        load_model_state(model, checkpoint)
    else:
        warn("No best_model.pt found (no validation ran). Evaluating the final-epoch model.")
    model.eval()
    dev_m = evaluate_on_dev(
        model, dev_pairs, num_beams=cfg.num_beams, batch_size=cfg.dev_batch_size,
        output_dir=os.path.join(run_dir, "dev_predictions", "final"), eval_gold_edu=True,
    )
    console.print(metrics_table(dev_m, title="Final Dev Results"))
    final_metrics: dict[str, dict[str, float]] = {"dev": dev_m}
    if test_pairs is not None:
        test_m = evaluate_on_dev(
            model, test_pairs, num_beams=cfg.num_beams, batch_size=cfg.dev_batch_size,
            output_dir=os.path.join(run_dir, "test_predictions", "final"), eval_gold_edu=True,
        )
        console.print(metrics_table(test_m, title="Final Test Results"))
        final_metrics["test"] = test_m
        tb.log_scalars("test", test_m, global_step)
    final_metrics["decode"] = {"num_beams": cfg.num_beams}
    metrics_path = os.path.join(run_dir, "final_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    wrote(metrics_path)
    tb.close()


def main():
    parser = argparse.ArgumentParser(description="Train the unified generative parser `gen`")
    parser.add_argument("config", help="Path to a jsonnet config file")
    args = parser.parse_args()
    cfg = GenConfig.from_dict(Params.from_file(args.config).as_dict(quiet=True))
    train(cfg)


if __name__ == "__main__":
    main()
