"""Shared dev/test evaluation for the generative RST parser `gen`.

The evaluation path is one shape for every backbone x serialization combo: batched
end-to-end decode then Parseval + segmentation F1 (token-range keyed, since predicted
EDU counts can differ from gold so an EDU-index-keyed Parseval would crash), plus an
optional gold-EDU-forced Parseval. Every combo difference (SR vs sexp serialization,
encoder-decoder vs causal) lives inside `model.predict_batch` /
`model.predict_with_gold_edus`, so this orchestration is combo-agnostic.

This is a plain function call, NOT inversion of control: `train_gen.py` owns its
training loop top-to-bottom and CALLS `evaluate_on_dev(model, dev_pairs, ...)`,
getting metrics back. `model` is duck-typed (see `GenerativeParser`): anything
exposing `eval()`, `tokenizer`, `predict_batch(...)`, and `predict_with_gold_edus(...)`.
"""

import os
import time
from typing import Protocol

import torch

from iudex.common.log import console, dim
from iudex.rst.data.metrics import compute_parseval_metrics, f1
from iudex.rst.data.seg_metrics import evaluate_seg_and_e2e
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.common.seqgen import align_edus_to_tokens, reconstruct_text


class GenerativeParser(Protocol):
    """The slice of the generative-parser interface this eval needs."""

    tokenizer: object

    def eval(self) -> object: ...
    def predict_batch(self, trees: list[RstTree], *, num_beams: int | None = None) -> list[RstTree]: ...
    def predict_with_gold_edus(self, tree: RstTree, *, num_beams: int | None = None) -> RstTree: ...


def _decodes_in_batches(model: GenerativeParser, num_beams: int | None) -> bool:
    """Whether `model.predict_batch` decodes its argument in ONE batched pass rather
    than a document at a time. Optional on the protocol (duck-typed `model`), so a
    parser that never batches simply doesn't declare it."""
    hook = getattr(model, "decodes_in_batches", None)
    return bool(hook(num_beams=num_beams)) if callable(hook) else False


def _chunks(dev_pairs: list[tuple[str, RstTree]], batch_size: int, bucket_by_length: bool) -> list[list[int]]:
    """Index chunks of at most `batch_size` documents.

    `bucket_by_length` groups documents of similar length together (longest first)
    instead of taking them in corpus order. It matters only when the chunk really is
    one batched decode: the batch runs until its LAST row retires, so a 185-EDU
    document mixed in with short ones would hold the whole batch open for its own
    length. Bucketing keeps that waste to the spread within a bucket. Chunks are
    index lists, so the caller still reports results in corpus order.
    """
    order = list(range(len(dev_pairs)))
    if bucket_by_length:
        order.sort(key=lambda i: -len(dev_pairs[i][1].edus))
    return [order[s : s + batch_size] for s in range(0, len(order), batch_size)]


def _gold_edu_token_mapping(model: GenerativeParser, tree: RstTree) -> tuple[list[int], list[tuple[int, int]]]:
    """EDU end-positions and per-EDU `(start, end_exclusive)` token-position
    ranges in the ENCODER'S whole-document tokenization space, the same space as
    the pred mappings the inference loop produces by cursor tracking. Delegates
    to `align_edus_to_tokens` (see it for why whole-doc tiling, not per-EDU
    tokenization, is required)."""
    text = reconstruct_text(tree)
    _, mapping = align_edus_to_tokens(model.tokenizer, text, tree.edus)
    edu_ends = [end - 1 for _, end in mapping]
    return edu_ends, mapping


def _pred_edu_token_mapping(pred_tree: RstTree) -> tuple[list[int], list[tuple[int, int]]]:
    """Pull the per-EDU source-position ranges that the inference loop
    stashed on the tree. Already in the encoder's source-id token space,
    so no re-tokenization needed."""
    ranges = getattr(pred_tree, "_pred_edu_source_ranges", None)
    if ranges is None:
        # Fallback: degenerate single-EDU empty tree.
        return [], []
    edu_ends = [end - 1 for _, end in ranges]
    return edu_ends, list(ranges)


def _write_rs4(tree: RstTree, output_dir: str, basename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, basename), "w", encoding="utf-8") as f:
        f.write(tree.to_rs4_string())


@torch.no_grad()
def _evaluate_gold_edu(
    model: GenerativeParser,
    dev_pairs: list[tuple[str, RstTree]],
    output_dir: str | None = None,
    *,
    num_beams: int | None = None,
    batch_size: int = 1,
) -> dict[str, float]:
    """Run the gold-EDU-forced predict path over every dev pair and
    aggregate Parseval. Per-tree shifts equal gold EDU counts by
    construction (forced segmentation), so `compute_parseval_metrics`
    sees aligned span counts and every document scores.

    Nothing is skipped: a drifting EDU count used to be absorbed here (it
    meant input truncation dropped gold EDUs), but over-length is a hard
    crash since 356bff2 and decode no longer fabricates a degenerate tree
    on failure, so a mismatch now means a bug and `compute_parseval_metrics`
    raising is the correct outcome. Dropping documents would silently make
    these numbers optimistic.

    `num_beams` is forwarded to `predict_with_gold_edus` so the gold-EDU
    condition decodes at the same beam width as the e2e condition (None/1
    = greedy forced decode, >1 = beam analogue). `batch_size` is the e2e
    condition's, so a parser that batches its greedy decode batches this
    pass at the same width -- forcing changes the mask, not the token
    count, so an unbatched gold-EDU pass would otherwise cost as much as
    the whole e2e eval again. `output_dir`, when set,
    saves each gold-EDU-forced pred tree as `{basename}.rs4` (checkpoints
    are deleted once a run's numbers are final, so predictions not written
    here are unrecoverable for error analysis). Output keys:
    `gold_edu_{span,nuc,rel,full}_f1`.
    """
    totals = {f"{m}_{x}_count": 0 for m in ("span", "nuc", "rel", "full") for x in ("p", "r")}
    totals["num_spans"] = 0
    eval_t0 = time.monotonic()
    batched = _decodes_in_batches(model, num_beams) and batch_size > 1
    for chunk in _chunks(dev_pairs, batch_size, bucket_by_length=batched):
        golds = [dev_pairs[i][1] for i in chunk]
        preds = (
            model.predict_batch_with_gold_edus(golds, num_beams=num_beams)
            if batched
            else [model.predict_with_gold_edus(g, num_beams=num_beams) for g in golds]
        )
        for i, pred in zip(chunk, preds, strict=True):
            if output_dir is not None:
                basename = os.path.splitext(os.path.basename(dev_pairs[i][0]))[0] + ".rs4"
                _write_rs4(pred, output_dir, basename)
            per_tree = compute_parseval_metrics(dev_pairs[i][1], pred)
            for k in totals:
                totals[k] += per_tree[k]
    dim(f"  gold-EDU eval: {time.monotonic() - eval_t0:.1f}s over {len(dev_pairs)} docs")
    num_spans = totals["num_spans"]
    if num_spans == 0:
        return {f"gold_edu_{m}_f1": 0.0 for m in ("span", "nuc", "rel", "full")}
    return {
        f"gold_edu_{m}_f1": f1(totals[f"{m}_p_count"] / num_spans, totals[f"{m}_r_count"] / num_spans)
        for m in ("span", "nuc", "rel", "full")
    }


@torch.no_grad()
def evaluate_on_dev(
    model: GenerativeParser,
    dev_pairs: list[tuple[str, RstTree]],
    *,
    num_beams: int | None = None,
    batch_size: int = 1,
    output_dir: str | None = None,
    eval_gold_edu: bool = False,
) -> dict[str, float]:
    """End-to-end Parseval + segmentation F1. No gold-EDU Parseval here:
    these parsers always use their own segmentation, so EDU counts can differ
    from gold and `evaluate_parseval` (which keys spans by EDU index) would
    crash. Token-range keyed `evaluate_seg_and_e2e` handles the alignment.

    `num_beams` overrides the parser's configured beam width. Per-epoch dev
    eval passes 1 (greedy) when `cfg.eval_decode_greedy`; final test eval
    passes the full `cfg.num_beams`. `batch_size > 1` groups dev documents
    into one `predict_batch` call per chunk; whether that chunk is one shared
    forward or a per-document loop is the parser's call (`decodes_in_batches`),
    and when it batches, chunks are length-bucketed. `output_dir`, when set,
    writes each pred tree as
    `{basename}.rs4` for later inspection. `eval_gold_edu` adds the
    gold-EDU-forced Parseval (`gold_edu_*` keys), decoded at the same
    `num_beams` as the e2e condition (so final eval beam-searches both).

    Per-batch wall-time + EDU counts are printed via `dim()` so a slow or
    pathological prediction is visible in real time.
    """
    model.eval()
    # A batched-decode parser runs each chunk as one shared forward, so keep
    # similar-length documents together (a long doc otherwise holds its whole batch
    # open); an unbatched parser decodes doc-by-doc, so corpus order is fine. Results
    # are collected by document index either way, so metrics stay order-independent.
    batched = _decodes_in_batches(model, num_beams) and batch_size > 1
    chunks = _chunks(dev_pairs, batch_size, bucket_by_length=batched)
    seg_by_index: dict[int, dict] = {}
    eval_t0 = time.monotonic()
    done = 0
    for chunk in chunks:
        chunk_t0 = time.monotonic()
        preds = model.predict_batch([dev_pairs[i][1] for i in chunk], num_beams=num_beams)
        chunk_dt = time.monotonic() - chunk_t0
        done += len(chunk)
        # Per-batch summary: one line covering all docs in the chunk.
        names = ",".join(os.path.basename(dev_pairs[i][0]) for i in chunk)
        gold_counts = [len(dev_pairs[i][1].edus) for i in chunk]
        pred_counts = [len(p.edus) for p in preds]
        dim(
            f"  dev {done}/{len(dev_pairs)} "
            f"[{names}]: gold_edus={gold_counts} pred_edus={pred_counts} {chunk_dt:.1f}s"
        )
        for i, pred in zip(chunk, preds, strict=True):
            filepath, gold = dev_pairs[i]
            gold_ends, gold_map = _gold_edu_token_mapping(model, gold)
            pred_ends, pred_map = _pred_edu_token_mapping(pred)
            seg_by_index[i] = {
                "gold_edu_ends": gold_ends,
                "pred_edu_ends": pred_ends,
                "e2e_pred": pred,
                "gold_edu_mapping": gold_map,
                "pred_edu_mapping": pred_map,
            }
            if output_dir is not None:
                basename = os.path.splitext(os.path.basename(filepath))[0] + ".rs4"
                _write_rs4(pred, output_dir, basename)
    # Reassemble in corpus order (bucketing may have reordered the chunks).
    gold_trees = [gold for _, gold in dev_pairs]
    seg_data = [seg_by_index[i] for i in range(len(dev_pairs))]
    dim(f"  dev eval total: {time.monotonic() - eval_t0:.1f}s over {len(dev_pairs)} documents")
    if output_dir is not None:
        console.print(f"[dim]Wrote {len(dev_pairs)} predictions under[/dim] [path]{os.path.abspath(output_dir)}[/path]")
    metrics = evaluate_seg_and_e2e(gold_trees, seg_data)
    if eval_gold_edu:
        gold_edu_dir = output_dir + "_gold_edu" if output_dir is not None else None
        metrics.update(
            _evaluate_gold_edu(model, dev_pairs, output_dir=gold_edu_dir, num_beams=num_beams, batch_size=batch_size)
        )
    return metrics
