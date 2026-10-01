"""Build the non-English RST corpora into data/.

Four languages: Basque, German, Persian, Chinese.

DISRPT 2025 ships only .tok/.conllu/.rels for its RST corpora -- segmentation
and relation *pairs*, never constituent trees -- so the trees have to come from
each corpus's upstream distribution. Document counts match upstream exactly
(GCDT 50, PCC 176), so DISRPT's splits map onto the upstream .rs3 files 1:1 by
document ID: splits from DISRPT, trees from upstream.

    zho.rst.gcdt  github.com/logan-siyao-peng/GCDT   data/rs3/{train,dev,test}
                  Splits already applied upstream (40/5/5). Straight copy.
                  Same 32-relation inventory as GUM.

    deu.rst.pcc   github.com/PeterBourgonje/pcc2.2   rst/ (176 files, flat)
                  Split by configs/lib/pcc_disrpt_partition.json. Needs the
                  headline fix below.

    fas.rst.prstc github.com/hadiveisi/PersianRST    Corpus/ (150 files, flat)
                  Split by configs/lib/prstc_disrpt_partition.json. One document
                  needs its root repaired; see build_prstc.

eus.rst.ert (RST Basque TreeBank) is NOT auto-fetched: its host currently 403s
on every content path, so those 164 .rs3 files must come from the authors or a
repaired server. Once on disk, --ert <dir> copies them in using
configs/lib/ert_disrpt_partition.json.

THE PCC HEADLINE FIX. Every PCC document is a two-root forest: the first
<segment> is the newspaper headline and is left deliberately unattached, and
our reader requires exactly one root. DISRPT resolves this by dropping the
headline -- verified on maz-00001, whose .rels/.tok begin at "Dagmar Ziegler
sitzt..." with the headline "Auf Eis gelegt" absent and token numbering
starting after it. We follow DISRPT: drop the first segment. It was confirmed
childless and parentless in all 176 documents, so the drop is structurally
inert.

That leaves exactly one document disconnected, maz-12666, whose body itself
splits into two unattached groups (33 and 40). It is in DISRPT's TRAIN split,
so it is skipped rather than repaired: excluding it costs one training document
and touches no evaluation, whereas joining the two halves would mean inventing
a relation the annotators did not assign.

Usage:
    python3 scripts/build_multiling_data.py                  # clone + build all
    python3 scripts/build_multiling_data.py --only gcdt
    python3 scripts/build_multiling_data.py --ert ~/ert_rs3   # Basque, hand-obtained
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iudex.common.log import wrote  # noqa: E402
from iudex.rst.data.reader import read_rst_file  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ("train", "dev", "test")

SOURCES = {
    "gcdt": "https://github.com/logan-siyao-peng/GCDT.git",
    "pcc": "https://github.com/PeterBourgonje/pcc2.2.git",
    "prstc": "https://github.com/hadiveisi/PersianRST.git",
}


def clone(name: str, work_dir: str) -> str:
    dest = os.path.join(work_dir, name)
    if os.path.isdir(os.path.join(dest, ".git")):
        print(f"[{name}] using existing clone at {dest}")
        return dest
    os.makedirs(work_dir, exist_ok=True)
    print(f"[{name}] cloning {SOURCES[name]} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", "-q", SOURCES[name], dest], check=True)
    return dest


def validate(paths: list[str]) -> tuple[int, list[str]]:
    """Check every written file the way training will actually use it.

    `read_rst_file` alone is NOT enough and once shipped six broken documents to
    the cluster: reading only builds the tree, while training additionally
    converts it to parsing actions, and that conversion is where degenerate
    nodes blow up (a one-child multinuc indexes `edge_yields[1]`; a `span` group
    with no `span` child exhausts a `next()`). Both crash at step 0 of a job, not
    at parse time, so `parsing_actions()` has to be part of the gate.
    """
    ok, bad = 0, []
    for p in paths:
        try:
            tree = read_rst_file(p)
            actions = tree.parsing_actions()
            # Not just "does it raise": a repair that mis-parents a node can
            # yield a WELL-FORMED tree with too few actions, which surfaces far
            # downstream as a KeyError on a missing gold decision or as
            # "Mismatched span lengths" in the generative evaluation. A binarized
            # n-EDU tree has exactly n-1 attachment decisions; check it here.
            n_edus = len(tree.edus)
            if len(actions) != n_edus - 1:
                raise ValueError(f"{len(actions)} parsing actions for {n_edus} EDUs, expected {n_edus - 1}")
            ok += 1
        except Exception as e:  # noqa: BLE001 - want the message, not the class
            bad.append(f"{os.path.basename(p)}: {type(e).__name__}: {e}")
    return ok, bad


def repair_tree(body: ET.Element, doc_id: str) -> list[str]:
    """Fix the two degenerate rs3 shapes that survive parsing but break training.

    Both are annotation-tool artifacts, both are local, and every change is
    returned so none of it is silent.

    ONE-CHILD MULTINUC (5 Persian documents). A multinuc expresses a relation
    among two or more nuclei, so a multinuc group holding a single child says
    nothing; the group is collapsed and its child reattached to the grandparent
    under the group's own role. No EDU yield changes -- a vacuous node is
    removed. Iterated to a fixpoint, since collapsing can expose another.

    SPAN GROUP WITH NO NUCLEUS (Basque INF02). A `span` group should have exactly
    one `span` child, the nucleus, plus satellites. Group 21 there has two
    children both labelled `kontrastea` and no nucleus at all, which is
    incoherent as a span and exactly right as a multinuc contrast between two
    nuclei, so it is retyped. Only applied when the children agree on a single
    relation; a group whose children disagree is a real ambiguity and is left to
    fail loudly rather than guessed at.
    """
    repairs: list[str] = []

    def children_of(node_id: str) -> list[ET.Element]:
        return [n for n in body.findall("segment") + body.findall("group") if n.get("parent") == node_id]

    while True:
        for grp in body.findall("group"):
            if grp.get("type") != "multinuc":
                continue
            kids = children_of(grp.get("id"))
            if len(kids) != 1 or grp.get("parent") is None:
                continue
            child = kids[0]
            repairs.append(
                f"{doc_id}: collapsed one-child multinuc group {grp.get('id')} "
                f"(child {child.get('id')} '{child.get('relname')}' -> parent {grp.get('parent')} "
                f"as '{grp.get('relname')}')"
            )
            child.set("parent", grp.get("parent"))
            child.set("relname", grp.get("relname"))
            body.remove(grp)
            break
        else:
            break

    for grp in body.findall("group"):
        if grp.get("type") != "span":
            continue
        kids = children_of(grp.get("id"))
        rels = {k.get("relname") for k in kids}
        if len(kids) >= 2 and "span" not in rels and len(rels) == 1:
            grp.set("type", "multinuc")
            repairs.append(
                f"{doc_id}: retyped span group {grp.get('id')} to multinuc "
                f"({len(kids)} children, all '{rels.pop()}', no nucleus)"
            )
    return repairs


# GCDT is the only one of our corpora that keeps PTB bracket escapes in its
# surface text (604 of them across 26 of its 50 documents; RST-DT, GUM and PCC
# have none). DISRPT renders them as the real characters in its `# text` lines,
# and a diff of a full document showed these escapes to be the ONLY thing
# separating our reconstruction from DISRPT's canonical text, so undoing them
# moves us onto the standard representation rather than away from it.
PTB_BRACKETS = {
    "-LSB-": "[", "-RSB-": "]",
    "-LRB-": "(", "-RRB-": ")",
    "-LCB-": "{", "-RCB-": "}",
}


def build_gcdt(src: str, out_root: str) -> list[str]:
    written, unescaped, all_repairs = [], 0, []
    for split in SPLITS:
        src_dir = os.path.join(src, "data", "rs3", split)
        files = sorted(glob(os.path.join(src_dir, "*.rs3")))
        if not files:
            sys.exit(f"[gcdt] no .rs3 files under {src_dir}")
        out_dir = os.path.join(out_root, "gcdt", split)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            tree = ET.parse(f)
            body = tree.getroot().find("body")
            for seg in body.findall("segment"):
                if seg.text:
                    for esc, char in PTB_BRACKETS.items():
                        if esc in seg.text:
                            unescaped += seg.text.count(esc)
                            seg.text = seg.text.replace(esc, char)
            all_repairs += repair_tree(body, os.path.splitext(os.path.basename(f))[0])
            dst = os.path.join(out_dir, os.path.basename(f))
            tree.write(dst, encoding="utf-8", xml_declaration=True)
            written.append(dst)
        print(f"[gcdt] {split}: {len(files)} docs -> {out_dir}")
    print(f"[gcdt] unescaped {unescaped} PTB brackets to their literal characters")
    for r in all_repairs:
        print(f"[gcdt] REPAIRED {r}")
    return written


def build_pcc(src: str, out_root: str) -> list[str]:
    partition = json.load(open(os.path.join(REPO, "configs", "lib", "pcc_disrpt_partition.json")))
    written, skipped, all_repairs = [], [], []
    for split in SPLITS:
        out_dir = os.path.join(out_root, "pcc", split)
        os.makedirs(out_dir, exist_ok=True)
        for doc_id in partition[split]:
            src_file = os.path.join(src, "rst", f"{doc_id}.rs3")
            if not os.path.isfile(src_file):
                sys.exit(f"[pcc] missing upstream tree for {doc_id}: {src_file}")
            tree = ET.parse(src_file)
            body = tree.getroot().find("body")

            # Drop the unattached headline (see module docstring). Assert rather
            # than assume: a headline with children or a parent would mean the
            # upstream layout changed under us.
            headline = body.find("segment")
            nodes = body.findall("segment") + body.findall("group")
            assert headline.get("parent") is None, f"{doc_id}: first segment is attached"
            assert not [n for n in nodes if n.get("parent") == headline.get("id")], (
                f"{doc_id}: headline has children"
            )
            body.remove(headline)

            remaining = body.findall("segment") + body.findall("group")
            roots = [n for n in remaining if n.get("parent") is None]
            if len(roots) != 1:
                skipped.append((doc_id, split, len(roots)))
                continue
            all_repairs += repair_tree(body, doc_id)

            dst = os.path.join(out_dir, f"{doc_id}.rs3")
            tree.write(dst, encoding="utf-8", xml_declaration=True)
            written.append(dst)
        print(f"[pcc] {split}: {len(glob(os.path.join(out_dir, '*.rs3')))} docs -> {out_dir}")
    for r in all_repairs:
        print(f"[pcc] REPAIRED {r}")
    for doc_id, split, n in skipped:
        print(f"[pcc] SKIPPED {doc_id} ({split}): still a forest after the headline drop ({n} roots)")
    return written


def build_prstc(src: str, out_root: str) -> list[str]:
    """Persian RST Corpus. Split by DISRPT; one document needs its root repaired.

    149 of the 150 documents are single-rooted and need nothing. The exception is
    `shargh031`, whose two top-level spans were never joined: segments 1-28 hang
    off group 94 and 29-47 off group 99, contiguous, adjacent, jointly covering
    the document with no orphans and no overlap. Only the single top attachment
    is missing.

    It is in the TEST split, so it cannot be skipped the way PCC's maz-12666 is
    -- dropping it would take ~7% of the Persian test EDUs with it, and per our
    data-completeness rule an incomplete test set blocks the run outright. We
    therefore join the halves under a multinuclear `joint`. That relation is
    already in this corpus's inventory and is its most frequent non-`span` label
    (1,915 uses), and `joint` is RST's generic "these spans are coordinate",
    which is exactly the reading two adjacent top-level spans force. The
    intervention adds ONE constituent to ONE of 15 test documents (~0.15% of
    test constituents) and is disclosed in the paper.
    """
    partition = json.load(open(os.path.join(REPO, "configs", "lib", "prstc_disrpt_partition.json")))
    written, repaired, all_repairs = [], [], []
    for split in SPLITS:
        out_dir = os.path.join(out_root, "prstc", split)
        os.makedirs(out_dir, exist_ok=True)
        for doc_id in partition[split]:
            src_file = os.path.join(src, "Corpus", f"{doc_id}.rs3")
            if not os.path.isfile(src_file):
                sys.exit(f"[prstc] missing upstream tree for {doc_id}: {src_file}")
            tree = ET.parse(src_file)
            body = tree.getroot().find("body")
            nodes = body.findall("segment") + body.findall("group")
            roots = [n for n in nodes if n.get("parent") is None]

            if len(roots) > 1:
                new_id = str(max(int(n.get("id")) for n in nodes) + 1)
                root = ET.SubElement(body, "group")
                root.set("id", new_id)
                root.set("type", "multinuc")
                for r in roots:
                    r.set("parent", new_id)
                    r.set("relname", "joint")
                repaired.append((doc_id, split, len(roots), new_id))

            all_repairs += repair_tree(body, doc_id)
            dst = os.path.join(out_dir, f"{doc_id}.rs3")
            tree.write(dst, encoding="utf-8", xml_declaration=True)
            written.append(dst)
        print(f"[prstc] {split}: {len(partition[split])} docs -> {out_dir}")
    for doc_id, split, n, new_id in repaired:
        print(f"[prstc] REPAIRED {doc_id} ({split}): joined {n} roots under new multinuc group {new_id} (relname=joint)")
    for r in all_repairs:
        print(f"[prstc] REPAIRED {r}")
    return written


def build_ert(src: str, out_root: str) -> list[str]:
    """RST Basque TreeBank, from a directory of .rs3 files supplied by the authors.

    Not auto-fetched: the canonical host (ixa2.si.ehu.eus) returns 403 on every
    content path with "Server unable to read htaccess file", serves an incomplete
    TLS chain, and is unarchived in the Wayback Machine. Luke obtained the zip
    directly; its train/dev/test layout was verified identical to DISRPT's
    partition, document for document.

    Two documents are defective, and they differ in kind:

    INF06 (test) carries a **literally duplicated line** -- the element
    `<group id="17" parent="26" relname="span" type="span"/>` appears twice with
    byte-identical attributes. That is a serialization slip, not an ID collision
    with two meanings, so dropping the copy is lossless: the children of 17 and
    its parent are unambiguous either way. Duplicates whose attributes DIFFER are
    a genuine ambiguity and hard-fail instead of being guessed at.

    INF13 (dev) is effectively **unannotated**: only 4 of its 16 segments attach
    to anything, leaving 14 roots. This is the corpus's own state rather than a
    packaging problem -- DISRPT's own `.rels` lists a single relation for its 16
    EDUs, where every other dev document is within a few of the expected count.
    There is no tree to recover and inventing 15 attachments is out of the
    question, so it is dropped. It falls in DEV, which is used for model
    selection and reports no numbers, so the cost is far lower than the
    equivalent test-split exclusion would be.
    """
    part_file = os.path.join(REPO, "configs", "lib", "ert_disrpt_partition.json")
    if not os.path.isfile(part_file):
        sys.exit(f"[ert] no partition file yet at {part_file}")
    partition = json.load(open(part_file))
    written, deduped, dropped, all_repairs = [], [], [], []
    for split in SPLITS:
        out_dir = os.path.join(out_root, "ert", split)
        os.makedirs(out_dir, exist_ok=True)
        for doc_id in partition[split]:
            matches = glob(os.path.join(src, "**", f"{doc_id}.rs3"), recursive=True)
            if not matches:
                sys.exit(f"[ert] missing tree for {doc_id} under {src}")
            raw = open(matches[0], "rb").read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")

            tree = ET.ElementTree(ET.fromstring(text))
            body = tree.getroot().find("body")

            seen: dict[str, dict] = {}
            for node in list(body.findall("segment")) + list(body.findall("group")):
                nid = node.get("id")
                if nid not in seen:
                    seen[nid] = node.attrib
                    continue
                if seen[nid] != node.attrib:
                    sys.exit(
                        f"[ert] {doc_id}: id {nid} is reused with DIFFERENT attributes "
                        f"({seen[nid]} vs {node.attrib}); refusing to guess"
                    )
                body.remove(node)
                deduped.append((doc_id, split, nid))

            nodes = body.findall("segment") + body.findall("group")
            roots = [n for n in nodes if n.get("parent") is None]
            if len(roots) != 1:
                dropped.append((doc_id, split, len(roots), len(body.findall("segment"))))
                continue
            all_repairs += repair_tree(body, doc_id)

            dst = os.path.join(out_dir, f"{doc_id}.rs3")
            tree.write(dst, encoding="utf-8", xml_declaration=True)
            written.append(dst)
        print(f"[ert] {split}: {len(glob(os.path.join(out_dir, '*.rs3')))} docs -> {out_dir}")
    for r in all_repairs:
        print(f"[ert] REPAIRED {r}")
    for doc_id, split, nid in deduped:
        print(f"[ert] DEDUPED {doc_id} ({split}): removed a byte-identical duplicate of node {nid}")
    for doc_id, split, n, nseg in dropped:
        print(f"[ert] DROPPED {doc_id} ({split}): {n} roots over {nseg} segments -- effectively unannotated")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "data"), help="data root (default: <repo>/data)")
    ap.add_argument("--work-dir", default="/tmp/iudex_multiling_src", help="where upstream repos are cloned")
    ap.add_argument("--only", choices=["gcdt", "pcc", "prstc", "ert"], help="build a single corpus")
    ap.add_argument("--ert", help="directory of hand-obtained RST Basque TreeBank .rs3 files")
    args = ap.parse_args()

    builders = {"gcdt": build_gcdt, "pcc": build_pcc, "prstc": build_prstc}
    targets = [args.only] if args.only else ["gcdt", "pcc", "prstc"] + (["ert"] if args.ert else [])
    all_written: list[str] = []
    for name in targets:
        if name == "ert":
            if not args.ert:
                sys.exit("--only ert requires --ert <dir>")
            all_written += build_ert(args.ert, args.out)
        else:
            all_written += builders[name](clone(name, args.work_dir), args.out)

    print(f"\nValidating {len(all_written)} files with read_rst_file ...")
    ok, bad = validate(all_written)
    for b in bad[:20]:
        print(f"  FAIL {b}")
    print(f"  {ok}/{len(all_written)} parse cleanly" + (f", {len(bad)} FAILED" if bad else ""))
    for p in sorted({os.path.dirname(p) for p in all_written}):
        wrote(p)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
