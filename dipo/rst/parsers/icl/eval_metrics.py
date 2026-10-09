"""Build the per-document `seg_data` that `evaluate_seg_and_e2e` consumes, for
the pred-EDU conditions (`e2e` / `two_call`) where the model's EDU set differs
from gold.

The trained parsers key end-to-end Parseval by subword-token range (see
`common/generative_eval.py`). ICL has no shared subword tokenizer, so we use a
**whitespace-token axis over the document** instead: the metric helpers are
agnostic to what a "token" is as long as gold and pred index the SAME sequence.
Gold EDU spans are exact cumulative counts; pred EDU spans are projected onto the
gold axis (identity when the model copied the document verbatim, difflib
alignment otherwise), so minor copy drift degrades the score honestly rather than
crashing.
"""

from __future__ import annotations

import difflib

from dipo.rst.data.tree import RstTree


def _cumulative_spans(token_lists: list[list[str]]) -> list[tuple[int, int]]:
    spans, c = [], 0
    for lst in token_lists:
        spans.append((c, c + len(lst)))
        c += len(lst)
    return spans


def _project_onto_gold(gold_tokens: list[str], pred_token_lists: list[list[str]]) -> list[tuple[int, int]]:
    """Map each predicted EDU's token span into the gold whitespace-token
    coordinate system. Exact when the concatenated prediction equals the gold
    token sequence; otherwise a monotonic difflib projection."""
    pred_flat = [t for lst in pred_token_lists for t in lst]
    if pred_flat == gold_tokens:
        return _cumulative_spans(pred_token_lists)

    sm = difflib.SequenceMatcher(a=pred_flat, b=gold_tokens, autojunk=False)
    p2g: dict[int, int] = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            p2g[i + k] = j + k

    spans: list[tuple[int, int]] = []
    p = 0
    prev_end = 0
    for lst in pred_token_lists:
        a, b = p, p + len(lst)
        p = b
        mapped = [p2g[x] for x in range(a, b) if x in p2g]
        if mapped:
            start, end = min(mapped), max(mapped) + 1
        else:
            start = end = prev_end  # unmatched EDU collapses to a zero-width boundary
        start = max(start, prev_end)
        end = max(end, start)
        spans.append((start, end))
        prev_end = end
    return spans


def failed_seg_data(gold_edus: list[str]) -> dict:
    """A `seg_data` record for a document whose prediction failed to parse: the
    gold mapping is kept (so the doc stays in the recall denominator) and the
    prediction is empty (`e2e_pred=None`, no pred boundaries), which scores it as
    zero rather than dropping it. Keeping failures in the denominator is what
    makes the ICL metric comparable to the trained parsers, which never fail."""
    gold_token_lists = [e.split() for e in gold_edus]
    gold_mapping = _cumulative_spans(gold_token_lists)
    return {
        "gold_edu_ends": [end - 1 for _, end in gold_mapping],
        "pred_edu_ends": [],
        "e2e_pred": None,
        "gold_edu_mapping": gold_mapping,
        "pred_edu_mapping": [],
    }


def seg_data_for_doc(gold_edus: list[str], pred_tree: RstTree) -> dict:
    """One `evaluate_seg_and_e2e` record: gold + pred EDU end indices and token
    spans over a shared whitespace-token axis, plus the pred tree."""
    gold_token_lists = [e.split() for e in gold_edus]
    gold_tokens = [t for lst in gold_token_lists for t in lst]
    gold_mapping = _cumulative_spans(gold_token_lists)

    pred_edus = list(pred_tree.edu_strings)
    pred_mapping = _project_onto_gold(gold_tokens, [e.split() for e in pred_edus])

    return {
        "gold_edu_ends": [end - 1 for _, end in gold_mapping],
        "pred_edu_ends": [end - 1 for _, end in pred_mapping],
        "e2e_pred": pred_tree,
        "gold_edu_mapping": gold_mapping,
        "pred_edu_mapping": pred_mapping,
    }
