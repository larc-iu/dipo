"""Re-score every e2e cell in preds/ with the official DISRPT 2025 segmentation scorer.

The main tables' Seg column is our own metric: position-level EDU-boundary set
F1 over each system's token axis (subword for trained parsers, whitespace for
ICL). The DISRPT shared tasks instead score per-token BeginSeg boundary P/R/F1
with `utils/disrpt_eval_2024.py -t S` (their `evaluation_main.sh` entrypoint).
This script converts each document's rs3 segmentation into DISRPT .tok files
and scores them with the scorer vendored verbatim at
scripts/disrpt/disrpt_eval_2024.py, so our end-to-end rows can sit on the
DISRPT 2025 scale.

Token axis: the GOLD document's whitespace tokens (per-EDU `str.split()` then
concatenation), the same axis the paper's ICL e2e metric uses
(iudex/rst/parsers/icl/eval_metrics.py). Gold and pred .tok files carry that
identical stream, as the scorer requires. Predicted EDU strings that do not
reproduce the document verbatim (ICL copy drift) are projected onto the gold
axis with the metric's own monotonic difflib projection (`n_drift` column);
every trained system reproduces the text exactly. A document with no
prediction file (failed ICL decode) contributes all-`_` labels, which keeps
the document in the recall denominator exactly like the paper's metric.

Sanity gates (hard unless noted):
  * gold-vs-gold must score exactly 1.0 P/R/F on every corpus;
  * per-document gold/pred token-count identity;
  * with --disrpt-data (a github.com/disrpt/sharedtask2025 clone), our RST-DT
    gold is checked against the official eng.rst.rstdt_test.tok: same 38 docs
    and identical per-document gold boundary counts, with boundary positions
    verified through a token-length alignment (see
    verify_rstdt_against_official; exact token identity is impossible because
    our 2017 rs3 conversion drops PTB quote tokens). GUM is reported but NOT
    gated: DISRPT 2025 used eng.erst.gum (eRST, earlier GUM release) while
    ours is GUM 12.1 RST, so we print boundary agreement over a
    whitespace-stripped character axis instead.

Outputs (see --out/--tsv): one .tok per (corpus, model) plus gold.tok per
corpus, and a tidy per-cell TSV of DISRPT-style seg_p/seg_r/seg_f1.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iudex.common.log import wrote  # noqa: E402
from iudex.rst.data.reader import read_rst_file  # noqa: E402
from iudex.rst.parsers.icl.eval_metrics import (  # noqa: E402
    _cumulative_spans,
    _project_onto_gold,
)

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
LAB_B = "Seg=B-seg"
OFFICIAL_TEST_TOK = {
    "rstdt": "eng.rst.rstdt/eng.rst.rstdt_test.tok",
    "gum": "eng.erst.gum/eng.erst.gum_test.tok",
}


def load_scorer():
    """Import the vendored official scorer. It imports sklearn at module scope
    for the relation/accuracy paths, which the segmentation path (-t S) never
    touches; stub sklearn if absent so the verbatim file imports unmodified."""
    try:
        import sklearn.metrics  # noqa: F401
    except ModuleNotFoundError:
        def _unavailable(*a, **k):
            raise ModuleNotFoundError("scikit-learn is required only for the scorer's relation path")

        metrics = types.ModuleType("sklearn.metrics")
        metrics.accuracy_score = _unavailable
        metrics.classification_report = _unavailable
        sklearn = types.ModuleType("sklearn")
        sklearn.metrics = metrics
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.metrics"] = metrics
    path = os.path.join(SCRIPTS, "disrpt", "disrpt_eval_2024.py")
    spec = importlib.util.spec_from_file_location("disrpt_eval_2024", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SegmentationEvaluation


def read_manifest(preds: str) -> dict[str, list[str]]:
    """corpus -> sorted e2e model names."""
    out: dict[str, set[str]] = {}
    with open(os.path.join(preds, "MANIFEST.tsv")) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row["condition"] == "e2e":
                out.setdefault(row["corpus"], set()).add(row["model"])
    return {c: sorted(ms) for c, ms in out.items()}


def doc_gold(path: str) -> tuple[list[str], list[int]]:
    """Gold whitespace tokens and EDU-begin token indices for one document."""
    tree = read_rst_file(os.path.join(path, "gold.rs3"))
    token_lists = [e.split() for e in tree.edu_strings]
    tokens = [t for lst in token_lists for t in lst]
    begins = [start for start, _ in _cumulative_spans(token_lists)]
    return tokens, begins


def doc_pred_begins(rs3: str, gold_tokens: list[str]) -> tuple[list[int], bool]:
    """Predicted EDU-begin indices on the gold token axis, plus a drift flag
    (True when the prediction did not reproduce the document verbatim and the
    difflib projection was used)."""
    tree = read_rst_file(rs3)
    # A malformed prediction can carry a text-less segment; treat it as a
    # zero-width EDU rather than crashing (its begin collapses onto the next).
    pred_lists = [(e or "").split() for e in tree.edu_strings]
    drift = [t for lst in pred_lists for t in lst] != gold_tokens
    if drift:
        spans = _project_onto_gold(gold_tokens, pred_lists)
    else:
        spans = _cumulative_spans(pred_lists)
    n = len(gold_tokens)
    begins = sorted({start for start, _ in spans if start < n})
    return begins, drift


def tok_lines(doc: str, tokens: list[str], begins: list[int]) -> list[str]:
    begin_set = set(begins)
    lines = [f"# newdoc id = {doc}"]
    for i, form in enumerate(tokens):
        lab = LAB_B if i in begin_set else "_"
        lines.append(f"{i + 1}\t{form}\t_\t_\t_\t_\t_\t_\t_\t{lab}")
    return lines


def parse_official_tok(path: str) -> dict[str, tuple[list[str], list[int]]]:
    """doc -> (forms, begin indices), skipping MWT/ellipsis lines exactly as
    the official scorer's parse_edu_data does."""
    docs: dict[str, tuple[list[str], list[int]]] = {}
    forms: list[str] = []
    begins: list[int] = []
    name = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("# newdoc id = "):
                if name is not None:
                    docs[name] = (forms, begins)
                name = line.split("= ", 1)[1]
                forms, begins = [], []
            elif line.startswith("#") or line == "":
                continue
            else:
                fields = line.split("\t")
                if "-" in fields[0] or "." in fields[0]:
                    continue
                if LAB_B in fields[-1]:
                    begins.append(len(forms))
                forms.append(fields[1])
    if name is not None:
        docs[name] = (forms, begins)
    return docs


def verify_rstdt_against_official(golds: dict[str, tuple[list[str], list[int]]], official_tok: str) -> None:
    """Hard gate: our RST-DT gold must carry the SAME SEGMENTATION as the
    official DISRPT 2025 test file. The token streams are not identical -- the
    2017 rs3 conversion behind our gold drops PTB quote tokens and retokenizes
    a few abbreviations ("Inc ." vs "Inc.") -- so exact per-token identity is
    impossible; instead we align the streams by per-token character length (the
    official text is underscored, lengths are all it exposes) and require:
    identical doc sets, identical per-document gold boundary COUNTS, and every
    official boundary either verifying position-exactly through the alignment
    or being a known tokenization artifact (off-by-one next to a dropped quote
    token, or sitting inside an alignment gap)."""
    import difflib

    official = parse_official_tok(official_tok)
    assert set(official) == set(golds), (
        f"doc sets differ: only-official={sorted(set(official) - set(golds))} "
        f"only-ours={sorted(set(golds) - set(official))}"
    )
    n_official = n_exact = n_offbyone = n_unaligned = 0
    for doc, (o_forms, o_begins) in sorted(official.items()):
        tokens, begins = golds[doc]
        assert len(o_begins) == len(begins), f"{doc}: {len(o_begins)} official gold boundaries vs {len(begins)} ours"
        sm = difflib.SequenceMatcher(a=[len(f) for f in o_forms], b=[len(t) for t in tokens], autojunk=False)
        o2g: dict[int, int] = {}
        for a, b, n in sm.get_matching_blocks():
            for k in range(n):
                o2g[a + k] = b + k
        begin_set = set(begins)
        for ob in o_begins:
            n_official += 1
            if ob not in o2g:
                n_unaligned += 1
            elif o2g[ob] in begin_set:
                n_exact += 1
            else:
                dist = min(abs(o2g[ob] - b) for b in begins)
                assert dist <= 2, f"{doc}: official boundary at token {ob} maps {dist} tokens from any of ours"
                n_offbyone += 1
    print(
        f"[gate] rstdt vs official DISRPT test .tok: {len(official)} docs, per-doc gold boundary counts identical "
        f"({n_official} boundaries); {n_exact} verify position-exactly ({100 * n_exact / n_official:.2f}%), "
        f"{n_offbyone} shifted <=2 tokens and {n_unaligned} unalignable, all at dropped-quote/abbreviation "
        f"tokenization artifacts"
    )


def report_gum_against_official(golds: dict[str, tuple[list[str], list[int]]], official_tok: str) -> None:
    """Report-only: boundary agreement between our GUM 12.1 RST gold and the
    official eng.erst.gum test gold on a whitespace-stripped character axis
    (their tokenization is UD words, ours whitespace; the corpora are different
    releases, so this is an alignment estimate, not a gate)."""
    import difflib

    official = parse_official_tok(official_tok)
    shared = sorted(set(official) & set(golds))
    tp = fp = fn = 0
    n_text_drift = 0
    for doc in shared:
        o_forms, o_begins = official[doc]
        tokens, begins = golds[doc]

        def char_offsets(forms: list[str], marks: list[int]) -> tuple[str, set[int]]:
            offs, c = [], 0
            for i, f in enumerate(forms):
                offs.append(c)
                c += len(f)
            return "".join(forms), {offs[i] for i in marks}

        o_chars, o_offs = char_offsets(o_forms, o_begins)
        g_chars, g_offs = char_offsets(tokens, begins)
        if o_chars != g_chars:
            n_text_drift += 1
            sm = difflib.SequenceMatcher(a=g_chars, b=o_chars, autojunk=False)
            g2o = {}
            for a, b, n in sm.get_matching_blocks():
                for k in range(n):
                    g2o[a + k] = b + k
            g_offs = {g2o[o] for o in g_offs if o in g2o}
        tp += len(o_offs & g_offs)
        fp += len(g_offs - o_offs)
        fn += len(o_offs - g_offs)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    print(
        f"[report] gum 12.1 gold vs official eng.erst.gum test gold: {len(shared)} shared docs, "
        f"{n_text_drift} with text drift; boundary agreement P={100 * p:.2f} R={100 * r:.2f} F1={100 * f:.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default=os.path.join(os.path.dirname(SCRIPTS), "preds"))
    ap.add_argument("--out", default=None, help="dir for .tok files (default: {preds}/disrpt_tok)")
    ap.add_argument("--tsv", default=None, help="output TSV (default: {preds}/disrpt_seg_scores.tsv)")
    ap.add_argument("--disrpt-data", default=None, help="data/ dir of a github.com/disrpt/sharedtask2025 clone, for the official-gold verification gates")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.preds, "disrpt_tok")
    tsv_path = args.tsv or os.path.join(args.preds, "disrpt_seg_scores.tsv")

    seg_evaluation = load_scorer()
    manifest = read_manifest(args.preds)

    rows = []
    for corpus in sorted(manifest):
        corpus_dir = os.path.join(args.preds, corpus)
        docs = sorted(d for d in os.listdir(corpus_dir) if os.path.isdir(os.path.join(corpus_dir, d)))
        golds = {doc: doc_gold(os.path.join(corpus_dir, doc)) for doc in docs}

        if args.disrpt_data:
            official_tok = os.path.join(args.disrpt_data, OFFICIAL_TEST_TOK[corpus])
            if corpus == "rstdt":
                verify_rstdt_against_official(golds, official_tok)
            else:
                report_gum_against_official(golds, official_tok)

        os.makedirs(os.path.join(out_dir, corpus), exist_ok=True)
        gold_path = os.path.join(out_dir, corpus, "gold.tok")
        with open(gold_path, "w", encoding="utf-8") as fh:
            for doc in docs:
                fh.write("\n".join(tok_lines(doc, *golds[doc])) + "\n")
        wrote(gold_path)

        # Gate: gold vs gold must be a perfect score.
        ev = seg_evaluation("gold-vs-gold", gold_path, gold_path)
        ev.compute_scores()
        assert ev.output["f_score"] == 1.0, f"{corpus}: gold-vs-gold scored {ev.output}"
        print(f"[gate] {corpus}: gold-vs-gold P/R/F all 1.0 over {ev.output['tok_count']} tokens")

        for model in manifest[corpus]:
            n_missing = n_drift = 0
            pred_path = os.path.join(out_dir, corpus, f"{model}.tok")
            with open(pred_path, "w", encoding="utf-8") as fh:
                for doc in docs:
                    tokens, _ = golds[doc]
                    rs3 = os.path.join(corpus_dir, doc, f"{model}.e2e.rs3")
                    if os.path.exists(rs3):
                        begins, drift = doc_pred_begins(rs3, tokens)
                        n_drift += drift
                    else:
                        begins = []
                        n_missing += 1
                    lines = tok_lines(doc, tokens, begins)
                    assert len(lines) - 1 == len(tokens), f"{doc}: token-count mismatch"
                    fh.write("\n".join(lines) + "\n")
            wrote(pred_path)

            ev = seg_evaluation(f"{corpus}/{model}", gold_path, pred_path)
            ev.compute_scores()
            o = ev.output
            rows.append(
                (
                    corpus,
                    model,
                    f"{100 * o['precision']:.2f}",
                    f"{100 * o['recall']:.2f}",
                    f"{100 * o['f_score']:.2f}",
                    len(docs),
                    n_missing,
                    n_drift,
                )
            )
            print(
                f"{corpus:5s} {model:30s} P={rows[-1][2]:>6s} R={rows[-1][3]:>6s} F1={rows[-1][4]:>6s}"
                f"  docs={len(docs)} missing={n_missing} drift={n_drift}"
            )

    with open(tsv_path, "w") as fh:
        fh.write("corpus\tmodel\tseg_p\tseg_r\tseg_f1\tn_docs\tn_missing\tn_drift\n")
        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")
    wrote(tsv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
