"""Assemble a browsable tree of every prediction behind the main tables.

    preds/{rstdt,gum}/{DOC}/{MODEL}.{e2e,goldedu}.rs3
    preds/{rstdt,gum,ert,pcc,prstc,gcdt}/{DOC}/gold.rs3

Cells are mapped to run dirs by MATCHING final_metrics.json against the printed
table values, never by run name: many cells have several candidate dirs
(superseded/duplicate hashes) and only the value match is safe. A cell that
does not match exactly one run is reported and skipped rather than guessed.

Prediction files are copied verbatim. `to_rs4_string` output is schema-identical
to the gold .rs3 (same tags, same attributes), so the .rs3 extension is honest;
the trees are written AS SCORED, i.e. binarized, so they carry more `group`
nodes than the n-ary gold. Pass --debinarize to flatten them for eyeballing
against gold (this changes the tree, so never use it to recompute scores).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iudex.common.log import wrote  # noqa: E402

TEX = os.path.expanduser("~/papers/serial_failures/main.tex")
ROOTS = [os.path.expanduser("~/iudex_ckpts"), os.path.expanduser("~/mnt/scratch/iudex_ckpts")]
JOBLOGS = os.path.expanduser("~/joblogs")
GOLD_DIRS = {
    "rstdt": "data/rstdt",
    "gum": "data/gum_12.1.0_notok",
    "ert": "data/ert",
    "pcc": "data/pcc",
    "prstc": "data/prstc",
    "gcdt": "data/gcdt",
}

# tab:multiling groups its rows under \multicolumn language headers rather than
# putting the corpus in the column layout, and it carries BOTH conditions in one
# row (Seg S N R F | S N R F). Run names are prefixed with the corpus key.
MULTILING = {"Basque": "ert", "German": "pcc", "Persian": "prstc", "Chinese": "gcdt"}

# NB: the printed Seg column is NO LONGER final_metrics.json's `seg_f1` -- it is
# re-scored with the official DISRPT 2025 scorer (scripts/disrpt_seg_rescore.py,
# commits 48690e1 / cc82722), which reads this very tree. So Seg cannot be part of
# the match key; cells are identified by the four Parseval columns alone.
E2E_KEYS = [["e2e_span_f1", "e2e_nuc_f1", "e2e_rel_f1", "e2e_full_f1"]]
GOLD_KEYS = [
    ["span_f1", "nuc_f1", "rel_f1", "full_f1"],
    ["gold_edu_span_f1", "gold_edu_nuc_f1", "gold_edu_rel_f1", "gold_edu_full_f1"],
]


def load_runs() -> dict:
    """Map run name -> (candidate dirs, test metrics).

    A run can exist under BOTH roots with COMPLEMENTARY contents -- e.g. the
    greedy `final/` predictions left on scratch while a later beam re-eval wrote
    `final_beam6/` to the local copy. Binding a run to the first root found
    silently loses whichever half lives in the other, so every candidate dir is
    kept and `pred_dir` searches them all.
    """
    out: dict = {}
    for r in ROOTS:
        for d in sorted(glob.glob(r + "/*")):
            fm = os.path.join(d, "final_metrics.json")
            if not os.path.isdir(d) or not os.path.exists(fm):
                continue
            try:
                t = json.load(open(fm)).get("test") or {}
            except Exception:
                continue
            name = os.path.basename(d)
            if name in out:
                out[name][0].append(d)
            else:
                out[name] = ([d], t)
    for f in sorted(glob.glob(JOBLOGS + "/*/results.json")):
        d = os.path.dirname(f)
        try:
            t = json.load(open(f)).get("metrics") or {}
        except Exception:
            continue
        out["icl:" + os.path.basename(d)] = ([d], t)
    return out


def sig(t, keys):
    try:
        return tuple(round(t[k] * 100, 1) for k in keys)
    except (KeyError, TypeError):
        return None


def table_rows(tex: str, label: str):
    body = tex.split("\\label{" + label + "}")[0]
    body = body[body.rfind("\\begin{table*}"):]
    for line in body.splitlines():
        line = line.strip()
        if not line.endswith("\\\\") or line.startswith("%") or "multicolumn" in line:
            continue
        if "\\textbf{System}" in line:
            continue
        parts = [p.strip() for p in line[:-2].split("&")]
        if len(parts) >= 3:
            yield parts[0], parts[1], parts[2:]


def multiling_rows(tex: str):
    """Yield (corpus, system, backbone, vals) from tab:multiling.

    Unlike the two main tables, the corpus lives in a \\multicolumn header that
    `table_rows` skips, so the current block has to be tracked as we scan.
    """
    body = tex.split("\\label{tab:multiling}")[0]
    body = body[body.rfind("\\begin{table*}"):]
    corpus = None
    for line in body.splitlines():
        line = line.strip()
        if not line.endswith("\\\\") or line.startswith("%"):
            continue
        if "multicolumn" in line:
            m = re.search(r"\\emph\{([A-Za-z]+)", line)
            if m:
                corpus = MULTILING.get(m.group(1))
            continue
        if "\\textbf{System}" in line:
            continue
        parts = [p.strip() for p in line[:-2].split("&")]
        if corpus and len(parts) >= 3:
            yield corpus, parts[0], parts[1], parts[2:]


def cells(vals, n):
    out = []
    for c in vals[:n]:
        c = c.strip()
        if c in ("\\pending", "--", "") or "todo" in c.lower():
            return None
        try:
            out.append(float(c))
        except ValueError:
            return None
    return tuple(out) if len(out) == n else None


def slug(system: str, backbone: str) -> str:
    s = re.sub(r"[\\{}]", "", system)
    s = re.sub(r"\s*\(.*?\)\s*", "", s).strip()
    b = re.sub(r"[\\{}]", "", backbone).strip()
    b = b.replace(" ", "-").replace("_", "-").lower()
    b = re.sub(r"[^a-z0-9.\-]", "", b)
    return f"{s}_{b}".strip("_")


def pred_dir(run_paths, is_icl: bool, cond: str) -> str | None:
    """Locate the directory of .rs4 predictions for one condition.

    Layouts differ by family:
      gen   test_predictions/{final,final_gold_edu}/
      dmrst test_predictions/final/{e2e,gold}/
      anchors (gold-EDU only)  test_predictions/final/
      icl   <joblogs run>/preds/
    """
    for run_path in run_paths:
        if is_icl:
            p = os.path.join(run_path, "preds")
            if os.path.isdir(p):
                return p
            continue
        tp = os.path.join(run_path, "test_predictions")
        if cond == "e2e":
            cands = (os.path.join(tp, "final", "e2e"), os.path.join(tp, "final"))
        else:
            cands = (
                os.path.join(tp, "final_gold_edu"),
                os.path.join(tp, "final", "gold"),
                os.path.join(tp, "final"),
            )
        for c in cands:
            if os.path.isdir(c) and any(f.endswith(".rs4") for f in os.listdir(c)):
                return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="preds")
    ap.add_argument("--debinarize", action="store_true", help="flatten predictions before writing")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="permit writing a MANIFEST with fewer cells than the existing one")
    args = ap.parse_args()

    tex = open(TEX).read()
    R = load_runs()
    manifest, problems = [], []

    for label, keys, width, cond in (
        ("tab:e2e", E2E_KEYS, 5, "e2e"),
        ("tab:goldedu", GOLD_KEYS, 4, "goldedu"),
    ):
        for system, backbone, vals in table_rows(tex, label):
            for corpus, off in (("rstdt", 0), ("gum", width)):
                printed = cells(vals[off:off + width], width)
                if printed is None:
                    continue
                if cond == "e2e":
                    printed = printed[1:]  # drop Seg (DISRPT-scored downstream)
                hits = []
                for n, (dirs, t) in R.items():
                    # The multilingual runs live under the same roots; without this
                    # an `ert-`/`pcc-` run could match an RST-DT cell by value alone.
                    if any(n.startswith(k + "-") for k in MULTILING.values()):
                        continue
                    is_gum = ("gum" in n.lower()) if n.startswith("icl:") else n.startswith("gum-")
                    if is_gum != (corpus == "gum"):
                        continue
                    if any(sig(t, ks) == printed for ks in keys):
                        hits.append(n)
                name = slug(system, backbone)
                if len(hits) != 1:
                    problems.append(f"{label} {name} {corpus}: {len(hits)} matches for {printed}")
                    continue
                run = hits[0]
                dirs, _ = R[run]
                src = pred_dir(dirs, run.startswith("icl:"), cond)
                if src is None:
                    problems.append(f"{label} {name} {corpus}: no {cond} predictions under {dirs}")
                    continue
                manifest.append((corpus, name, cond, run, src))

    for corpus, system, backbone, vals in multiling_rows(tex):
        for cond, off, width, keys in (("e2e", 0, 5, E2E_KEYS), ("goldedu", 5, 4, GOLD_KEYS)):
            printed = cells(vals[off:off + width], width)
            if printed is None:
                continue
            if cond == "e2e":
                printed = printed[1:]  # drop Seg (DISRPT-scored downstream)
            hits = [n for n, (dirs, t) in R.items()
                    if n.startswith(corpus + "-") and any(sig(t, ks) == printed for ks in keys)]
            name = slug(system, backbone)
            if len(hits) != 1:
                problems.append(f"tab:multiling {name} {corpus}: {len(hits)} matches for {printed}")
                continue
            run = hits[0]
            dirs, _ = R[run]
            src = pred_dir(dirs, False, cond)
            if src is None:
                problems.append(f"tab:multiling {name} {corpus}: no {cond} predictions under {dirs}")
                continue
            manifest.append((corpus, name, cond, run, src))

    # Documents available per corpus, so per-cell coverage is interpretable.
    # ICL cells are legitimately short: a document whose decode fails produces no
    # prediction file, and those documents still score zero in the denominator.
    expected = {}
    for corpus, base in GOLD_DIRS.items():
        d = os.path.join(base, "test")
        expected[corpus] = (
            len([f for f in os.listdir(d) if f.endswith((".rs3", ".rs4"))]) if os.path.isdir(d) else 0
        )

    n_files = 0
    counts: dict = {}
    for corpus, name, cond, run, src in manifest:
        for f in sorted(os.listdir(src)):
            if not f.endswith((".rs4", ".rs3")):
                continue
            counts[(corpus, name, cond)] = counts.get((corpus, name, cond), 0) + 1
            doc = f.rsplit(".", 1)[0]
            dest_dir = os.path.join(args.out, corpus, doc)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, f"{name}.{cond}.rs3")
            if args.debinarize:
                from iudex.rst.data.tree import RstTree

                t = RstTree.from_rs4_string(open(os.path.join(src, f)).read())
                open(dest, "w").write(t.debinarize().to_rs4_string())
            else:
                shutil.copyfile(os.path.join(src, f), dest)
            n_files += 1

    # gold, for every document that received at least one prediction
    n_gold = 0
    for corpus, base in GOLD_DIRS.items():
        cdir = os.path.join(args.out, corpus)
        if not os.path.isdir(cdir):
            continue
        for doc in sorted(os.listdir(cdir)):
            found = False
            # RST-DT ships gold as .rs3, GUM as .rs4; the schema is the same, so
            # both are normalized to gold.rs3 here.
            for split in ("test", "dev", "train"):
                for ext in (".rs3", ".rs4"):
                    g = os.path.join(base, split, doc + ext)
                    if os.path.exists(g):
                        shutil.copyfile(g, os.path.join(cdir, doc, "gold.rs3"))
                        n_gold += 1
                        found = True
                        break
                if found:
                    break

    mpath = os.path.join(args.out, "MANIFEST.tsv")
    # Refuse to shrink the manifest. Runs age off the 30-day scratch purge, and
    # once one does, its cell silently fails to map and a rebuild drops it --
    # which quietly discards the only record of predictions that are still
    # sitting right there in the tree. Losing cells is therefore a hard error,
    # not a diff. Pass --allow-shrink when the loss is genuinely intended.
    prev = 0
    if os.path.exists(mpath):
        with open(mpath) as fh:
            prev = max(0, sum(1 for _ in fh) - 1)
    if prev and len(manifest) < prev and not args.allow_shrink:
        print(f"\nREFUSING to overwrite {mpath}: it has {prev} cells, this run mapped only {len(manifest)}.")
        for p_ in problems:
            print(f"  {p_}")
        print("The existing manifest is left untouched. Update incrementally, or pass --allow-shrink.")
        return 1

    incomplete = []
    with open(mpath, "w") as fh:
        fh.write("corpus\tmodel\tcondition\tfiles\tdocs\trun\tsource\n")
        for corpus, name, cond, run, src in sorted(manifest):
            got = counts.get((corpus, name, cond), 0)
            exp = expected.get(corpus, 0)
            fh.write(f"{corpus}\t{name}\t{cond}\t{got}\t{exp}\t{run}\t{src}\n")
            if exp and got != exp:
                incomplete.append((corpus, name, cond, got, exp))
    wrote(os.path.abspath(mpath))

    rpath = os.path.join(args.out, "README.md")
    with open(rpath, "w") as fh:
        fh.write(
            "# Predictions behind the main tables\n\n"
            "`{corpus}/{DOCUMENT}/{MODEL}.{e2e,goldedu}.rs3`, plus `gold.rs3` per document.\n\n"
        "**This tree is the durable artifact; the run directories are not.** Cells are mapped by\n"
        "matching printed table values against each run's `final_metrics.json`, and those live on a\n"
        "30-day-purged scratch filesystem, so a full rebuild silently maps FEWER cells once a run\n"
        "ages out. Update incrementally rather than rebuilding, keep `MANIFEST.tsv` under version\n"
        "control, and note that a backbone rename leaves the superseded model's files behind --\n"
        "this script copies but never prunes.\n\n"
            "- **e2e** — the system segmented the document itself, so its EDUs need not match gold.\n"
            "- **goldedu** — gold segmentation was supplied, so EDU counts match gold exactly.\n"
            "- Systems that only run under gold EDUs (the two discriminative anchors) have no `e2e` file.\n\n"
            "Trees are written **as scored**, i.e. binarized, so they carry more `group` nodes than the\n"
            "n-ary gold. Rebuild with `--debinarize` to flatten them for reading against gold; do not\n"
            "score those.\n\n"
            "`MANIFEST.tsv` maps every cell to the run that produced it, with a `files`/`docs` coverage\n"
            "pair. **Coverage below `docs` is expected for the in-context rows**: a document whose decode\n"
            "fails produces no prediction file, and those documents still score zero in the denominator\n"
            "of the reported metric. Every trained system is complete.\n\n"
            "`disrpt_seg_scores.tsv` re-scores every e2e cell's segmentation with the official\n"
            "DISRPT 2025 scorer (per-token BeginSeg boundary P/R/F1); regenerate it with\n"
            "`scripts/disrpt_seg_rescore.py` (which also writes the intermediate `disrpt_tok/`).\n\n"
            "Regenerate with `scripts/build_preds.py` in the iudex repo.\n"
        )
    wrote(os.path.abspath(rpath))

    models = sorted({(c, m) for c, m, _, _, _ in manifest})
    print(f"\n{len(manifest)} cells mapped, {len(models)} corpus/model pairs")
    print(f"{n_files} prediction files + {n_gold} gold files under {os.path.abspath(args.out)}")
    if incomplete:
        print(f"\n{len(incomplete)} cell(s) with partial coverage (expected for in-context rows):")
        for corpus, name, cond, got, exp in incomplete:
            print(f"  {corpus:5s} {name:28s} {cond:8s} {got}/{exp}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  " + p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
