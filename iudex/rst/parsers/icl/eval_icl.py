"""`iudex icl eval` -- run the ICL pipeline over a labeled split and score it.

This is the paper-number command. Metric depends on the pipeline mode:
  - gold_edu -> lockstep gold-EDU Parseval (structure + labeling).
  - e2e / two_call -> segmentation F1 + token-range end-to-end Parseval (the
    model produces its own EDUs).
Docs whose response fails to parse are counted and skipped (reported, never
silently dropped).
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from iudex.common.log import console, dim, rule, success, warn
from iudex.rst.data.metrics import compute_parseval_metrics, f1, metrics_table
from iudex.rst.data.reader import read_rst_dir
from iudex.rst.data.seg_metrics import evaluate_seg_and_e2e
from iudex.rst.parsers.icl._cli_common import build_parser, load_config, write_json, write_text
from iudex.rst.parsers.icl.eval_metrics import failed_seg_data, seg_data_for_doc
from iudex.rst.parsers.icl.serialize import doc_text


def _gold_edu_metrics(scored_pairs, failed_golds) -> dict:
    """Aggregate gold-EDU Parseval, counting failed documents as zero (their gold
    spans go into the denominator with no matches) so the metric is over ALL docs,
    comparable to the trained parsers which never fail."""
    totals = {f"{m}_{x}_count": 0 for m in ("span", "nuc", "rel", "full") for x in ("p", "r")}
    totals["num_spans"] = 0
    for gold, pred in scored_pairs:
        per = compute_parseval_metrics(gold, pred)
        for k in totals:
            totals[k] += per[k]
    for gold in failed_golds:
        totals["num_spans"] += max(len(gold.edus) - 1, 0)
    ns = totals["num_spans"]
    if ns == 0:
        return {}
    return {
        f"{m}_f1": f1(totals[f"{m}_p_count"] / ns, totals[f"{m}_r_count"] / ns)
        for m in ("span", "nuc", "rel", "full")
    }


def _resolve_eval_dir(cfg, args) -> str:
    if args.input:
        return args.input
    if args.split == "dev":
        return cfg.dev_dir
    if args.split == "test":
        if not cfg.test_dir:
            raise SystemExit("--split test requested but config has no test_dir")
        return cfg.test_dir
    raise SystemExit("provide --input DIR or --split {dev,test}")


def _predict_doc(parser, mode, gold, out: Path, did: str) -> dict:
    """Predict one document. Pure w.r.t. shared state (safe to run in a worker
    thread): returns an outcome dict, the caller does all aggregation."""
    t0 = time.time()
    gold_edus = list(gold.edu_strings)
    parser.begin_doc()  # resets this thread's forced-call / token counters
    try:
        if mode == "gold_edu":
            pred = parser.predict_with_gold_edus(gold)
            if len(pred.edu_strings) != len(gold_edus):
                raise ValueError(f"gold-EDU count drift pred={len(pred.edu_strings)} gold={len(gold_edus)}")
        else:
            pred = parser.predict_from_text(doc_text(gold_edus))
    except Exception as e:
        # Truncate: a litellm error repr can embed the whole request body.
        return {"did": did, "gold": gold, "pred": None, "error": repr(e)[:500],
                "elapsed": time.time() - t0, **parser.doc_stats()}
    try:
        (out / "preds" / f"{did}.rs4").write_text(pred.to_rs4_string())
    except Exception as e:
        return {"did": did, "gold": gold, "pred": pred, "error": None, "elapsed": time.time() - t0,
                "write_error": repr(e), **parser.doc_stats()}
    return {"did": did, "gold": gold, "pred": pred, "error": None, "elapsed": time.time() - t0,
            **parser.doc_stats()}


def main() -> None:
    argp = argparse.ArgumentParser(prog="iudex icl eval", description="Evaluate the ICL parser on a split")
    argp.add_argument("--config", required=True, help="jsonnet IclConfig")
    src = argp.add_mutually_exclusive_group()
    src.add_argument("--input", help="RS3/RS4 directory to evaluate on")
    src.add_argument("--split", choices=("dev", "test"), help="use cfg.dev_dir / cfg.test_dir")
    argp.add_argument("--output-dir", help="write predictions + results.json here")
    argp.add_argument("--limit", type=int, default=None, help="only the first N docs")
    argp.add_argument("--concurrency", type=int, default=1, help="number of documents to predict in parallel")
    argp.add_argument("--dry-run", action="store_true", help="assemble+print the first doc's prompt, no API calls")
    args = argp.parse_args()

    cfg = load_config(args.config)
    eval_dir = _resolve_eval_dir(cfg, args)
    pairs = read_rst_dir(eval_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"no RS3/RS4 documents found in {eval_dir}")

    parser = build_parser(cfg)
    mode = cfg.pipeline_mode
    rule(f"ICL eval: {cfg.serialization}/{mode} via {parser.provider.describe()}")
    dim(f"eval dir: {eval_dir}  docs: {len(pairs)}  k={cfg.k}  relation types: {len(cfg.relation_types)}")

    if args.dry_run:
        _, gold = pairs[0]
        probe = list(gold.edu_strings) if mode == "gold_edu" else doc_text(gold.edu_strings)
        console.print(parser.preview_prompt(probe))
        dim("[dry-run: no API calls made]")
        return

    if not args.output_dir:
        argp.error("--output-dir is required unless --dry-run")
    out = Path(args.output_dir)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    write_text(out / "prompt_example.txt", parser.preview_prompt(
        list(pairs[0][1].edu_strings) if mode == "gold_edu" else doc_text(pairs[0][1].edu_strings)
    ))

    gold_trees = []  # e2e/two_call: golds in scoring order (failures included)
    seg_data = []  # e2e/two_call: one record per gold (a failure -> empty prediction)
    scored_pairs = []  # gold_edu: (gold, pred)
    failed_golds = []  # gold_edu: golds whose prediction failed
    per_doc = []
    failures = []

    parser.prewarm()  # build shared prompt prefixes once, before the workers race on them
    dim(f"predicting with concurrency={args.concurrency}")
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    outcomes: list[dict] = []
    with progress:
        task = progress.add_task("ICL eval", total=len(pairs))
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(_predict_doc, parser, mode, gold, out, Path(path).stem) for path, gold in pairs]
            for fut in as_completed(futs):
                outcomes.append(fut.result())
                progress.advance(task)
    outcomes.sort(key=lambda r: r["did"])  # deterministic output order regardless of completion order

    for r in outcomes:
        gold = r["gold"]
        if r.get("write_error"):
            warn(f"[{r['did']}] rs4 write failed: {r['write_error']} (still scored)")
        if r["error"]:
            # Keep the doc in the denominator (scored as zero), never drop it,
            # so the metric stays comparable to the ever-valid trained parsers.
            warn(f"[{r['did']}] failed: {r['error']}")
            failures.append({"doc_id": r["did"], "error": r["error"]})
            if mode == "gold_edu":
                failed_golds.append(gold)
            else:
                gold_trees.append(gold)
                seg_data.append(failed_seg_data(list(gold.edu_strings)))
            continue
        pred = r["pred"]
        if mode == "gold_edu":
            scored_pairs.append((gold, pred))
        else:
            gold_trees.append(gold)
            seg_data.append(seg_data_for_doc(list(gold.edu_strings), pred))
        per_doc.append(
            {
                "doc_id": r["did"],
                "gold_edus": len(gold.edu_strings),
                "pred_edus": len(pred.edu_strings),
                "elapsed_s": r["elapsed"],
                # Which documents were truncated, and how long each call actually
                # generated -- the two things needed to tell a bound that rescued a
                # runaway from one that cut off a document which would have finished.
                "forced": r.get("forced", 0),
                "call_tokens": r.get("call_tokens", []),
            }
        )

    if mode == "gold_edu":
        metrics = _gold_edu_metrics(scored_pairs, failed_golds)
    else:
        metrics = evaluate_seg_and_e2e(gold_trees, seg_data) if gold_trees else {}

    cfg_dict = dataclasses.asdict(cfg)
    if cfg_dict["provider"].get("api_key"):  # never persist an inline key
        cfg_dict["provider"]["api_key"] = "<redacted>"
    results = {
        "provider": parser.provider.describe(),
        "eval_dir": eval_dir,
        "num_docs": len(pairs),
        "num_scored": len(pairs) - len(failures),
        "num_failures": len(failures),
        # Calls (not docs -- two_call gives a doc two chances) whose reasoning was
        # cut off by grammar.think_budget. Their answers are still fully
        # grammar-constrained, but they came from truncated reasoning, so the
        # count is reported rather than blended into the naturally-completed ones.
        "num_forced_calls": parser.forced_calls,
        "forced_docs": sorted(r["doc_id"] for r in per_doc if r.get("forced")),
        "scored_over_all_docs": True,  # failures counted as zero, not dropped
        "failures": failures,
        "per_doc": per_doc,
        "metrics": metrics,
        "config": cfg_dict,
    }
    write_json(out / "results.json", results)
    write_json(out / "config.json", cfg_dict)  # lets `runs list` / archive see this run

    if metrics:
        console.print(metrics_table(metrics, title=f"{cfg.serialization}/{mode} via {cfg.provider.model}"))
    else:
        warn("no documents scored")
    if failures:
        warn(f"{len(failures)}/{len(pairs)} docs failed (counted as zero, kept in the denominator)")
    if parser.forced_calls:
        warn(f"{parser.forced_calls} call(s) hit grammar.think_budget and had reasoning truncated")
    success(f"Done. Results: {out / 'results.json'}")


if __name__ == "__main__":
    main()
