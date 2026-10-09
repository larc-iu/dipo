"""Build raw-text ("notok") variants of the RST corpora whose EDU text is word-tokenized.

Released parsers read real text, but several corpora store EDUs in a tokenized
form no user will ever type: RST-DT and PCC split off punctuation (`trade ,`,
`do n't`), GCDT spaces out Chinese words, and the Persian corpus spaces off
commas. A parser trained on that learns cues that are absent at inference
(" ," marks a boundary), so each variant here keeps the trees byte-identical and
replaces only the EDU strings with the corpus's untokenized text. Where two EDUs
are written with no space between them, the segment gets `prefix=""`, the same
convention as data/gum_12.1.0_notok (see RstNode.prefix).

    rstdt  ->  rstdt_notok   EDU lines of the LDC release (RSTtrees-WSJ-main-1.0/
                             {TRAINING,TEST}/*.edus). Aligns 1:1 with the DISRPT
                             EDUs on all 385 documents, which differ only in
                             tokenization and in having dropped every double quote.
    pcc    ->  pcc_notok     primary-data/*.txt of github.com/PeterBourgonje/pcc2.2,
                             aligned character by character (whitespace ignored).
    gcdt   ->  gcdt_notok    DISRPT 2025 zho.rst.gcdt CoNLL-U: the same tokens,
                             rejoined by their SpaceAfter=No flags.
    prstc  ->  prstc_notok   No untokenized source exists, so a narrow rule: drop
                             the space before closing punctuation and after
                             opening punctuation. Persian commas attach to the
                             preceding word, but the corpus spaces off 1,357 of
                             1,358 of them.

Basque (ert) and GUM (gum_12.1.0_notok) are already natural text.

Every builder asserts that the new EDU text equals the old once whitespace (and,
for RST-DT, double quotes) is removed, so no variant can change what an EDU
contains, only how it is spaced.

Usage:
    python3 scripts/build_raw_text_data.py --data-root data --only rstdt
    python3 scripts/build_raw_text_data.py --data-root data \\
        --pcc-src /path/to/pcc2.2 --disrpt-src /path/to/sharedtask2025
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess

from lxml import etree

SPLITS = ("train", "dev", "test")
PCC_REPO = "https://github.com/PeterBourgonje/pcc2.2"
DISRPT_REPO = "https://github.com/disrpt/sharedtask2025"


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def norm(s: str) -> str:
    """EDU content for the no-change assertion: tokenizers dropped double quotes
    (DISRPT RST-DT) and joined multiword tokens with "_" (PCC)."""
    return squash(s).replace('"', "").replace("_", "")


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def doc_id(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def write_variant(src: str, dst: str, texts: list[str], glued: list[bool]) -> None:
    """Copy `src` to `dst`, replacing segment texts; `glued[i]` sets prefix="" on segment i."""
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(src, parser)
    segs = list(tree.getroot().iter("segment"))
    if len(segs) != len(texts):
        raise ValueError(f"{src}: {len(segs)} segments but {len(texts)} texts")
    for seg, text, g in zip(segs, texts, glued):
        old = seg.text or ""
        if norm(old) != norm(text):
            raise ValueError(f"{src}: EDU content changed\n  old: {old!r}\n  new: {text!r}")
        if not text:
            raise ValueError(f"{src}: empty EDU")
        seg.text = text
        if g:
            seg.set("prefix", "")
        elif "prefix" in seg.attrib:
            del seg.attrib["prefix"]
    # Glue never applies before the first EDU; mirror gum_12.1.0_notok, which marks it.
    segs[0].set("prefix", "")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=False)


def segments(path: str) -> list[str]:
    return [s.text or "" for s in etree.parse(path).getroot().iter("segment")]


def corpus_files(data_root: str, corpus: str):
    for split in SPLITS:
        for f in sorted(glob.glob(os.path.join(data_root, corpus, split, "*.rs3"))):
            yield split, f


# --------------------------------------------------------------------------- RST-DT


def build_rstdt(data_root: str, ldc_root: str) -> int:
    ldc = {}
    for sub in ("TRAINING", "TEST"):
        for f in glob.glob(os.path.join(ldc_root, sub, "*.edus")):
            with open(f, encoding="latin-1") as fh:
                ldc[doc_id(f)] = [collapse(line) for line in fh if line.strip()]
    n = 0
    for split, f in corpus_files(data_root, "rstdt"):
        texts = ldc[doc_id(f)]
        write_variant(f, os.path.join(data_root, "rstdt_notok", split, os.path.basename(f)), texts, [False] * len(texts))
        n += 1
    return n


# --------------------------------------------------------------------------- char alignment


def align_to_raw(raw: str, edus: list[str], where: str, log: list[str]) -> tuple[list[str], list[bool]]:
    """Locate tokenized `edus` (in order) inside `raw`, ignoring whitespace on both sides.

    Each EDU is rebuilt from the raw characters it covers, with a single space
    wherever the raw text has whitespace. Three tokenizer/annotator artifacts are
    tolerated, and each repair is appended to `log`:
      - "_" joining a multiword token ("220_000" for raw "220 000") reads as a space;
      - a hyphenation leftover in the raw text ("Ab- stand", "zwi- schen", the
        hyphen followed by whitespace) that the tokenized text joined into one
        word is dropped together with the whitespace;
      - punctuation present only in the tokenized EDU (a period the annotators
        added after a headline-like line) is kept, glued to the preceding word.
    Anything else that diverges raises. Returns the EDU texts and whether each
    EDU touches the previous one with no whitespace in between."""
    idx = [i for i, c in enumerate(raw) if not c.isspace()]
    raw_sq = "".join(raw[i] for i in idx)
    first = squash(edus[0]).replace("_", "")[:30]
    p = raw_sq.find(first)
    if p < 0:
        raise ValueError(f"{where}: first EDU {first!r} not found in raw text")
    texts, glued = [], []
    last = None  # raw index of the last character consumed
    for i, edu in enumerate(edus):
        out: list[str] = []
        starts_glued = False
        chars = squash(edu)
        j = 0
        while j < len(chars):
            c = chars[j]
            if p < len(raw_sq) and raw_sq[p] == c:
                gap = last is not None and idx[p] > last + 1
                if out and gap:
                    out.append(" ")
                elif not out:
                    starts_glued = i > 0 and not gap
                out.append(c)
                last = idx[p]
                p += 1
                j += 1
            elif c == "_":
                j += 1
            elif raw_sq[p : p + 2] == "-" + c and raw[idx[p] + 1 : idx[p] + 2].isspace():
                log.append(f"{where}: EDU {i + 1}: dropped hyphenation leftover before {raw_sq[p + 1:p + 11]!r}")
                last = idx[p + 1] - 1
                p += 1
            elif not c.isalnum():
                log.append(f"{where}: EDU {i + 1}: kept {c!r} absent from raw text after {''.join(out)[-20:]!r}")
                out.append(c)
                j += 1
            else:
                raise ValueError(
                    f"{where}: EDU {i + 1} diverges from raw text at {c!r}: edu {edu!r} vs raw "
                    f"{raw_sq[max(0, p - 20):p]!r}|{raw_sq[p:p + 20]!r}"
                )
        texts.append("".join(out))
        glued.append(starts_glued)
    return texts, glued


# --------------------------------------------------------------------------- PCC


def build_pcc(data_root: str, pcc_src: str) -> int:
    n, errors, log = 0, [], []
    for split, f in corpus_files(data_root, "pcc"):
        with open(os.path.join(pcc_src, "primary-data", doc_id(f) + ".txt"), encoding="utf-8") as fh:
            raw = fh.read()
        try:
            texts, glued = align_to_raw(raw, segments(f), f, log)
        except ValueError as e:
            errors.append(str(e))
            continue
        write_variant(f, os.path.join(data_root, "pcc_notok", split, os.path.basename(f)), texts, glued)
        n += 1
    for line in log:
        print(f"[pcc] repaired {line}")
    if errors:
        raise ValueError(f"{len(errors)} PCC documents failed to align:\n" + "\n".join(errors))
    return n


# --------------------------------------------------------------------------- GCDT


def read_conllu_docs(paths: list[str]) -> dict[str, list[tuple[str, bool]]]:
    """doc id -> [(token form, space_after)] over the whole document."""
    docs: dict[str, list[tuple[str, bool]]] = {}
    cur = None
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("# newdoc id"):
                    cur = line.split("=", 1)[1].strip()
                    docs[cur] = []
                elif line and not line.startswith("#"):
                    cols = line.split("\t")
                    if "-" in cols[0] or "." in cols[0]:
                        continue  # multiword ranges / empty nodes
                    docs[cur].append((cols[1], "SpaceAfter=No" not in cols[9]))
    return docs


def build_gcdt(data_root: str, disrpt_src: str) -> int:
    docs = read_conllu_docs(sorted(glob.glob(os.path.join(disrpt_src, "data", "zho.rst.gcdt", "*.conllu"))))
    n = 0
    for split, f in corpus_files(data_root, "gcdt"):
        toks = docs[doc_id(f)]
        texts, glued = [], []
        k = 0
        for i, edu in enumerate(segments(f)):
            words = edu.split()
            forms = [t for t, _ in toks[k:k + len(words)]]
            if forms != words:
                raise ValueError(f"{f}: EDU {i + 1} tokens {words[:8]} != DISRPT {forms[:8]}")
            out = ""
            for j, (form, space_after) in enumerate(toks[k:k + len(words)]):
                out += form + (" " if space_after and j < len(words) - 1 else "")
            texts.append(out)
            glued.append(i > 0 and not toks[k - 1][1])
            k += len(words)
        if k != len(toks):
            raise ValueError(f"{f}: {len(toks) - k} DISRPT tokens left over")
        write_variant(f, os.path.join(data_root, "gcdt_notok", split, os.path.basename(f)), texts, glued)
        n += 1
    return n


# --------------------------------------------------------------------------- Persian

FA_CLOSE = "،؛؟!.:)»]"
FA_OPEN = "(«["


def detok_fa(text: str) -> str:
    text = re.sub(r"\s+([" + re.escape(FA_CLOSE) + r"])", r"\1", text)
    text = re.sub(r"([" + re.escape(FA_OPEN) + r"])\s+", r"\1", text)
    return collapse(text)


def build_prstc(data_root: str) -> int:
    n = 0
    for split, f in corpus_files(data_root, "prstc"):
        texts = [detok_fa(t) for t in segments(f)]
        # A segment opening on closing punctuation belongs against the previous one.
        glued = [i > 0 and t[:1] in FA_CLOSE for i, t in enumerate(texts)]
        write_variant(f, os.path.join(data_root, "prstc_notok", split, os.path.basename(f)), texts, glued)
        n += 1
    return n


# --------------------------------------------------------------------------- main


def ensure_clone(url: str, dest: str, sparse: list[str] | None = None) -> str:
    if os.path.isdir(dest):
        return dest
    if sparse:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", url, dest], check=True)
        subprocess.run(["git", "-C", dest, "sparse-checkout", "set", *sparse], check=True)
    else:
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, dest], check=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--only", choices=["rstdt", "pcc", "gcdt", "prstc"], action="append")
    ap.add_argument("--ldc-root", default=None, help="default: <data-root>/rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0")
    ap.add_argument("--pcc-src", default=None, help="pcc2.2 checkout (cloned into --work-dir if absent)")
    ap.add_argument("--disrpt-src", default=None, help="sharedtask2025 checkout (sparse-cloned if absent)")
    ap.add_argument("--work-dir", default="/tmp/dipo_raw_text_src")
    args = ap.parse_args()
    todo = args.only or ["rstdt", "pcc", "gcdt", "prstc"]
    os.makedirs(args.work_dir, exist_ok=True)
    for corpus in todo:
        if corpus == "rstdt":
            ldc = args.ldc_root or os.path.join(args.data_root, "rst_discourse_treebank", "data", "RSTtrees-WSJ-main-1.0")
            n = build_rstdt(args.data_root, ldc)
        elif corpus == "pcc":
            n = build_pcc(args.data_root, args.pcc_src or ensure_clone(PCC_REPO, os.path.join(args.work_dir, "pcc2.2")))
        elif corpus == "gcdt":
            src = args.disrpt_src or ensure_clone(
                DISRPT_REPO, os.path.join(args.work_dir, "sharedtask2025"), ["data/zho.rst.gcdt"]
            )
            n = build_gcdt(args.data_root, src)
        else:
            n = build_prstc(args.data_root)
        print(f"wrote {n} documents to {os.path.abspath(os.path.join(args.data_root, corpus + '_notok'))}")


if __name__ == "__main__":
    main()
