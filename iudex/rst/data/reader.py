import os
import re
from collections import Counter
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lxml import etree

from iudex.rst.data.tree import RstEdge, RstNode, RstTree

logger = getLogger(__name__)


def _extract_one(filepath, elt, name, nullable=False):
    target = elt.findall(name)
    if not nullable and len(target) != 1:
        raise ValueError(f"rs4 file {filepath} does not have exactly one <{name}> element")
    elif nullable and len(target) == 0:
        return None
    else:
        return target[0]


def _norm_ws(text):
    """Collapse whitespace runs and trim; `None` passes through."""
    return None if text is None else re.sub(r"\s+", " ", text).strip()


def _drop_keys(d, ks):
    return {k: v for k, v in d.items() if k not in ks}


def _read_rs4_into_dict(filepath: str) -> Dict[str, Any]:
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    # lxml's own parse, not the stdlib ElementTree wrapper: driving an lxml
    # parser through xml.etree.ElementTree.parse leaves its error_log empty
    # even when recovery fired, which would defeat the check below.
    tree = etree.parse(filepath, parser)
    if len(parser.error_log) > 0:
        # recover=True silently deletes malformed spans (unescaped ampersands,
        # undefined entities) or drops everything after a mid-file truncation,
        # and the mangled result can still validate as a smaller tree. Reject
        # instead of training or evaluating on it.
        raise ValueError(f"{filepath} is not well-formed XML: {parser.error_log[0]}")
    document = dict()

    header = _extract_one(filepath, tree, "header")
    relations = _extract_one(filepath, header, "relations")
    document["relation_inventory"] = [r.attrib for r in relations.findall("rel")]

    body = _extract_one(filepath, tree, "body")
    terminals = body.findall("segment")
    # Normalise EDU whitespace. Some corpora store segment text with the
    # inter-EDU whitespace of the source document attached (PCC: 100% of EDUs,
    # Basque: 61%, including embedded newlines), which puts a whitespace run at
    # every EDU boundary. DMRST never saw it because tokenize_document strips
    # each EDU, but the generative family reads reconstruct_text verbatim and so
    # was handed its segmentation target -- German gen scored seg_f1 EXACTLY
    # 1.000 and Basque 0.980 on that artifact. Normalising here keeps both
    # families on identical text. RST-DT, GUM and GCDT have no such whitespace,
    # so this is a no-op for them.
    document["terminals"] = [
        {"text": _norm_ws(t.text), "type": "terminal", **t.attrib} for t in terminals
    ]
    nonterminals = body.findall("group")
    document["nonterminals"] = [n.attrib for n in nonterminals]
    secedges = _extract_one(filepath, body, "secedges", nullable=True)
    document["secondary_edges"] = [] if secedges is None else [e.attrib for e in secedges.findall("secedge")]
    return document


def _validate_dict(filepath: str, d: Dict[str, Any]) -> None:
    nodes = d["terminals"] + d["nonterminals"]
    ids = [n["id"] for n in nodes]
    if not len(set(ids)) == len(ids):
        raise ValueError(f"Document {filepath} does not have unique IDs for each node")
    for node in nodes:
        if "parent" in node and node["parent"] not in ids:
            raise ValueError(f"Document {filepath} has edge with non-existent parent {node['parent']}")
    roots = [n for n in nodes if "parent" not in n]
    if len(roots) != 1:
        raise ValueError(f"Document {filepath} does not have exactly one root")


def _process_dict(d: Dict[str, Any]) -> Tuple[List[RstNode], List[RstEdge]]:
    terminals = [RstNode(**_drop_keys(n, ["parent", "relname"])) for n in d["terminals"]]
    nonterminals = [RstNode(**_drop_keys(n, ["parent", "relname"])) for n in d["nonterminals"]]
    nodes = terminals + nonterminals

    terminal_edges = [
        RstEdge(source=n["parent"], target=n["id"], relation=n["relname"]) for n in d["terminals"] if "parent" in n
    ]
    nonterminal_edges = [
        RstEdge(source=n["parent"], target=n["id"], relation=n["relname"]) for n in d["nonterminals"] if "parent" in n
    ]
    secondary_edges = [
        RstEdge(source=e["source"], target=e["target"], relation=e["relname"], secondary=True)
        for e in d["secondary_edges"]
    ]
    edges = terminal_edges + nonterminal_edges + secondary_edges

    return nodes, edges


def _repair_multinuc_satellites(filepath: str, d: Dict[str, Any]) -> None:
    """Give a multinuc node's surplus satellites their own span nodes.

    A multinuc carrying ONE satellite is ordinary and binarizes correctly --
    RST-DT has 1,423 of them and GUM 3,087, all fine. TWO or more on the same
    multinuc is what breaks: binarization only ever consumes the multinuc
    members plus a single satellite, so the surplus never becomes a parsing
    action and the tree yields fewer than `n-1` of them. Downstream that shows
    up far from here, as `KeyError` on a missing gold decision in DMRST or as
    "Mismatched span lengths" in the generative evaluation.

    The RST-standard reading of a multinuc with several satellites is a nested
    one: the multinuc core takes its first satellite, that whole span takes the
    next, and so on. So each surplus satellite gets an interposed `span` node,
    which reduces the structure to the one-satellite shape that already works.
    Satellites attach nearest-first, keeping every span contiguous.

    Only ever fires on >= 2 satellites, so it CANNOT touch RST-DT or GUM (no
    multinuc in either carries more than one; most carry none at all) -- it is
    confined to 21 multinuc nodes across the Basque and German corpora, where it
    interposes 22 spans. Each repair is logged rather than done silently.
    """
    inventory = {r.get("name"): r.get("type") for r in d.get("relation_inventory", [])}
    nodes = d["terminals"] + d["nonterminals"]
    by_id = {n["id"]: n for n in nodes}
    children: Dict[str, list] = {}
    for n in nodes:
        if "parent" in n:
            children.setdefault(n["parent"], []).append(n)

    # EDU yield of every node, by walking each terminal up to the root.
    yields: Dict[str, list] = {}
    for i, term in enumerate(d["terminals"]):
        cur = term
        while cur is not None:
            yields.setdefault(cur["id"], []).append(i)
            cur = by_id.get(cur.get("parent")) if "parent" in cur else None

    next_id = max((int(n["id"]) for n in nodes if str(n["id"]).isdigit()), default=0) + 1

    for node in list(d["nonterminals"]):
        if node.get("type") != "multinuc":
            continue
        kids = children.get(node["id"], [])
        counts = Counter(k["relname"] for k in kids if "relname" in k)
        if not counts:
            continue
        # Which children are MEMBERS is declared by the rs3 header, which types
        # every relation `multinuc` or `rst`; a majority vote over children gets
        # it wrong whenever satellites outnumber members or tie with them. PCC
        # has two such nodes -- maz-11916 node 16 is {evidence: 2, joint: 2},
        # where `evidence` is an `rst` relation and wins the tie on document
        # order, and maz-14654 node 28 is {interpretation: 3, conjunction: 2}.
        # Reading either by majority dismembers the multinuc and invents
        # `(evidence, multinuc)`-style classes in the label space.
        # The inventory cannot be used alone: 721 of Persian's 2,007 multinuc
        # nodes have no child typed `multinuc` at all, so fall back to the vote.
        member_rel = next(
            (r for r, _ in counts.most_common() if inventory.get(r) == "multinuc"),
            counts.most_common(1)[0][0],
        )
        sats = [k for k in kids if k.get("relname") != member_rel]
        if len(sats) < 2:
            continue

        core = [i for k in kids if k.get("relname") == member_rel for i in yields.get(k["id"], [])]
        lo, hi = min(core), max(core)

        def gap(sat, lo=lo, hi=hi):
            y = yields.get(sat["id"], [])
            return 0 if not y else (lo - max(y) if max(y) < lo else min(y) - hi)

        sats.sort(key=gap)
        current = node
        for sat in sats[1:]:
            new = {"id": str(next_id), "type": "span"}
            next_id += 1
            if "parent" in current:
                new["parent"] = current["parent"]
                new["relname"] = current["relname"]
                current["parent"] = new["id"]
                current["relname"] = "span"
            else:
                current["parent"] = new["id"]
                current["relname"] = "span"
            sat["parent"] = new["id"]
            d["nonterminals"].append(new)
            logger.warning(
                f"{filepath}: multinuc node {node['id']} carries {len(sats)} satellites; "
                f"interposed span node {new['id']} for satellite {sat['id']} "
                f"('{sat.get('relname')}'). Two or more satellites on one multinuc "
                f"cannot binarize; nesting them is the standard RST reading."
            )
            current = new


def read_rst_file(
    filepath: str,
    binarize: bool = True,
    relation_types: Tuple[Tuple[str, str], ...] = None,
    relation_map: Optional[Dict[str, str]] = None,
) -> RstTree:
    """Read an RS3 or RS4 file and return an RstTree. If `relation_map` is
    set, the tree applies it at its output boundary (e.g. `parsing_actions`,
    `spans`, `relation_of`). Edges retain raw labels so that structure-
    inference logic that distinguishes multinuc-siblings from satellites by
    relation-name distinctness is not broken by fine→coarse collapses.
    """
    logger.debug(f"Reading {filepath}")
    d = _read_rs4_into_dict(filepath)
    _validate_dict(filepath, d)
    _repair_multinuc_satellites(filepath, d)
    nodes, edges = _process_dict(d)
    return RstTree(nodes, edges, binarize=binarize, relation_types=relation_types, relation_map=relation_map)


def read_rst_dir(
    directory: str,
    binarize: bool = True,
    relation_types: Tuple[Tuple[str, str], ...] = None,
    relation_map: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, RstTree]]:
    """Read all RS3/RS4 files from a directory, returning (filepath, tree) pairs.
    `relation_map` is forwarded to `RstTree` for output-boundary remapping.
    """
    paths = sorted(glob(str(Path(directory) / "*.rs3")))
    paths += sorted(glob(str(Path(directory) / "*.rs4")))
    results = []
    for p in paths:
        tree = read_rst_file(p, binarize=binarize, relation_types=relation_types, relation_map=relation_map)
        results.append((p, tree))
    return results


def infer_relation_types(
    directories: List[str],
    relation_map: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """Scan RS3/RS4 files in `directories` and return the union of
    (relation, kind) pairs observed in their parsing actions. A given relation
    may appear with both kinds. RST-DT, for example, has relations that are
    mononuclear in some contexts and multinuclear in others.

    When `relation_map` is set, observed relations are reported in the mapped
    space. Sorted for deterministic config hashing.
    """
    seen = set()
    for directory in directories:
        for _, tree in read_rst_dir(directory, relation_map=relation_map):
            for _, nuc, rel in tree.parsing_actions():
                kind = "multinuc" if nuc == "NN" else "rst"
                seen.add((rel, kind))
    return sorted(seen)


def determine_label_index(relation_types: Tuple[Tuple[str, str], ...]) -> List[str]:
    """Expand (relation, kind) pairs into joint nuclearity+relation labels.

    Parser label classifiers predict a single portmanteau label like
    "NS_elaboration" that encodes both the relation and which child is nucleus:
      - "multinuc" relations get one label "NN_<relation>"
      - "rst" (mononuclear) relations get two labels, "NS_<relation>" (nucleus
        left) and "SN_<relation>" (nucleus right).
    """
    out = []
    for relation, kind in relation_types:
        if kind == "multinuc":
            out.append(f"NN_{relation}")
        elif kind == "rst":
            out.append(f"NS_{relation}")
            out.append(f"SN_{relation}")
        else:
            raise ValueError(f"Unknown relation kind: {kind}")
    return out
