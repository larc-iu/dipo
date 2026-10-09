"""`dipo icl predict` -- parse documents with the ICL parser (no scoring).

Bespoke CLI (the ICL parser has no torch checkpoint, so it does not use the
shared `run_predict`). Input is RS3/RS4 (uses gold EDUs when pipeline_mode is
gold_edu, else re-segments from the joined text), a raw .txt file/dir, or an
inline string.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dipo.common.log import console, dim, rule, success
from dipo.rst.data.reader import read_rst_file
from dipo.rst.parsers.icl._cli_common import build_parser, glob_docs, load_config, write_text
from dipo.rst.parsers.icl.serialize import doc_text


def main() -> None:
    argp = argparse.ArgumentParser(prog="dipo icl predict", description="Predict RST trees with the ICL parser")
    argp.add_argument("--config", required=True, help="jsonnet IclConfig")
    inp = argp.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", help="RS3/RS4 file or directory (gold EDUs used iff pipeline_mode=gold_edu)")
    inp.add_argument("--text-file", dest="text_file", help="raw .txt file or directory of .txt files")
    inp.add_argument("--text", help="inline raw text; tree written to stdout as RS4")
    argp.add_argument("--output-dir", help="required unless --text is used")
    argp.add_argument("--dry-run", action="store_true", help="print the assembled prompt for the first input, no API calls")
    args = argp.parse_args()

    cfg = load_config(args.config)
    parser = build_parser(cfg)
    mode = cfg.pipeline_mode

    # gold_edu needs gold EDUs, which raw text does not carry.
    if mode == "gold_edu" and (args.text is not None or args.text_file is not None):
        argp.error("pipeline_mode='gold_edu' needs gold EDUs; use --input (RS3/RS4), not --text/--text-file")

    if args.text is not None:
        if args.dry_run:
            console.print(parser.preview_prompt(args.text))
            return
        print(parser.predict_from_text(args.text).to_rs4_string())
        return

    if not args.output_dir:
        argp.error("--output-dir is required when --input or --text-file is used")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rule(f"ICL predict: {cfg.serialization}/{mode} via {parser.provider.describe()}")

    if args.text_file is not None:
        paths = glob_docs(args.text_file, ("*.txt",))
    else:
        paths = glob_docs(args.input, ("*.rs3", "*.rs4"))

    if args.dry_run:
        first = paths[0]
        if args.text_file is not None:
            probe = Path(first).read_text(encoding="utf-8")
        else:
            gold = read_rst_file(first, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
            probe = list(gold.edu_strings) if mode == "gold_edu" else doc_text(gold.edu_strings)
        console.print(parser.preview_prompt(probe))
        dim("[dry-run: no API calls made]")
        return

    for path in paths:
        did = Path(path).stem
        if args.text_file is not None:
            pred = parser.predict_from_text(Path(path).read_text(encoding="utf-8"))
        else:
            gold = read_rst_file(path, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
            if mode == "gold_edu":
                pred = parser.predict_with_gold_edus(gold)
            else:
                pred = parser.predict_from_text(doc_text(gold.edu_strings))
        write_text(out / f"{did}.rs4", pred.to_rs4_string())

    success(f"Done. Predictions in {out}")


if __name__ == "__main__":
    main()
