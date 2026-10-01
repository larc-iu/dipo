"""Emit the LaTeX table rows for the icl-c cells straight from results.json.

Transcribing eight-to-ten numbers per row by hand is exactly the step where a
digit goes missing, and a wrong number in a table is invisible once it is in the
paper. So read the run directories and print the rows.

Two tables, different column sets (see main.tex):
    tab:e2e      seg span nuc rel full   -- own segmentation, token-range matching
    tab:goldedu      span nuc rel full   -- gold segmentation, EDU-index matching
each with RST-DT columns then GUM columns.

Also prints failures and forced calls per cell, which the paper reports alongside
every ICL row and which must not be dropped when the numbers are copied.

Usage:
    python scripts/harvest_icl_rows.py <run-dir> [<run-dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

E2E_KEYS = ("seg_f1", "e2e_span_f1", "e2e_nuc_f1", "e2e_rel_f1", "e2e_full_f1")
GOLD_KEYS = ("span_f1", "nuc_f1", "rel_f1", "full_f1")


def load(d: Path) -> dict:
    r = json.loads((d / "results.json").read_text())
    cfg = r["config"]
    m = r["metrics"]
    gold = cfg["pipeline_mode"] == "gold_edu"
    keys = GOLD_KEYS if gold else E2E_KEYS
    return {
        "dir": d.name,
        "gold": gold,
        "corpus": "GUM" if "gum" in cfg["test_dir"] else "RST-DT",
        "endpoint": cfg["provider"]["base_url"],
        "cells": [m[k] for k in keys] if all(k in m for k in keys) else None,
        "docs": r["num_docs"],
        "failures": r["num_failures"],
        # older runs predate the counter; report as unknown rather than as zero
        "forced": r.get("num_forced_calls", "?"),
        "budget": cfg["grammar"].get("think_budget"),
        "k": cfg["k"],
    }


def main(dirs: list[str]) -> int:
    runs = []
    for d in dirs:
        p = Path(d)
        if not (p / "results.json").is_file():
            print(f"!! no results.json in {p}", file=sys.stderr)
            return 1
        runs.append(load(p))

    for table, gold in (("tab:e2e", False), ("tab:goldedu", True)):
        sel = [r for r in runs if r["gold"] == gold]
        if not sel:
            continue
        print(f"\n% ---- {table} ----")
        for r in sel:
            if r["cells"] is None:
                print(f"% {r['dir']}: metrics missing the expected keys, skipped")
                continue
            nums = " & ".join(f"{100 * v:.1f}" for v in r["cells"])
            print(f"% {r['dir']} ({r['corpus']}, k={r['k']}, {r['endpoint']})")
            print(f"%   {nums}")

    print("\n% ---- failure / bound accounting (report alongside every ICL row) ----")
    for r in runs:
        b = "unbounded" if r["budget"] is None else f"bound {r['budget']}"
        print(f"%   {r['dir']:44s} {r['failures']}/{r['docs']} failed, {r['forced']} forced calls, {b}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
