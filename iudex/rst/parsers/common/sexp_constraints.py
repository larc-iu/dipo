"""Validity constraints for s-expression decoding (`gen`'s `sexp` serialization).

A pushdown automaton over decoder positions. Validity is enforced by
restricting which action ids are legal at each step. The state is immutable
under `step()` (returns a new state) so beam search can hold per-beam states
without aliasing.

Grammar (per `RstTree.to_sexp` / `from_sexp`, plan style):

  tree   ::= '(' LABEL tree tree ')'        -- pre-order internal
           | '(' tree tree LABEL ')'        -- post-order internal
           | '(' CONTENT* ')'               -- leaf with literal source content

  CONTENT ::= source token (verbatim from input)  -- when use_copy=False
            | <copy>                              -- when use_copy=True

Action vocabulary, as integer ids:

  open_id, close_id: '(' and ')'
  label_ids: set of valid internal-node labels (NS:rel, SN:rel, NN:rel)
  eos_id: end-of-sequence
  use_copy=True:  copy_id (the single `<copy>` token; advances the cursor)
                  no source_ids passed (the decoder's leaf-text is just <copy>s)
  use_copy=False: source_ids = list of input subword ids, one per cursor
                  position. The legal source token at any cursor i is exactly
                  source_ids[i]. Any other token in the source vocabulary is
                  illegal at that position.

Constraints enforced:
  * Root close legal iff cursor == source_len AND depth becomes 0 after close.
  * Cannot close an EDU leaf with zero content tokens.
  * Exactly one label per internal span. Pre-order: label legal only at the
    just-after-open slot. Post-order: label legal only at the just-before-close
    slot (after both subtrees have been emitted).
  * Source-token / <copy> emit legal iff inside an EDU leaf AND cursor < source_len.
  * EOS legal iff depth == 0 AND cursor == source_len AND a tree has been emitted.

The state intentionally does not depend on the model's hidden state. It is a
pure function of the action-id prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import FrozenSet, List, Optional, Tuple

# `GoldEduForcer.narrowed_legal`'s return: None (no narrowing) or a whitelist.
NarrowedLegal = Optional[FrozenSet[int]]


# Per-span state pushed onto the stack each time '(' opens a span.
# `kind` is None until we know whether the span is a leaf or an internal node;
# the first non-'(' action inside the span resolves it.
@dataclass(frozen=True)
class _Frame:
    kind: Optional[str] = None  # None | 'leaf' | 'internal'
    children_emitted: int = 0  # for internal nodes (0, 1, or 2)
    leaf_token_count: int = 0  # for leaf nodes
    leaf_has_text: bool = False  # leaf has copied a non-whitespace subword
    label_emitted: bool = False  # for internal nodes


@dataclass(frozen=True)
class SexpDecodingState:
    source_len: int
    traversal_order: str  # 'preorder' | 'postorder'
    use_copy: bool

    open_id: int
    close_id: int
    eos_id: int
    label_ids: FrozenSet[int]
    copy_id: Optional[int] = None  # required iff use_copy=True
    source_ids: Tuple[int, ...] = ()  # required iff use_copy=False; len == source_len

    # Minimum content-token count required before a leaf may close. Mirrors
    # the same-name knob on the encoder parsers. Inference-only (training uses
    # teacher-forced sequences). Exception: at end-of-source the leaf may
    # close even when below the threshold, since otherwise the final EDU
    # cannot commit.
    min_edu_length: int = 1

    # Multi-token relation-word labels (label_style='words'). Empty in token
    # mode, where a label is a single id in `label_ids`. Prefix-free by
    # construction (`build_word_label_vocab`), so the label trie needs no
    # terminators. When non-empty, `label_ids` is unused and labels are matched
    # as token sequences through `label_cursor`.
    word_label_ids: FrozenSet[Tuple[int, ...]] = frozenset()

    # Source positions whose subword decodes to whitespace only. A leaf holding
    # nothing else would be an empty EDU, so every leaf must eat at least one
    # TEXT position: blank positions are free to eat but do not count toward the
    # content budget (`remaining_content`), and a leaf owes one text position
    # until it has one. Empty (the default; gold forcing passes none) reduces
    # every gate below to its original form.
    blank_positions: FrozenSet[int] = frozenset()

    cursor: int = 0
    depth: int = 0
    stack: Tuple[_Frame, ...] = ()  # one frame per currently open span
    # Tokens of the in-progress multi-token label emitted so far in the current
    # label slot; () when not mid-label. Always a PROPER prefix of some label
    # between steps (completion is applied immediately in `step`). Words mode only.
    label_cursor: Tuple[int, ...] = ()
    root_emitted: bool = False  # set True after the root's matching ')' fires
    terminated: bool = False  # set True after EOS

    def __post_init__(self):
        if self.traversal_order not in ("preorder", "postorder"):
            raise ValueError(f"Unknown traversal_order {self.traversal_order!r}")
        if self.use_copy:
            if self.copy_id is None:
                raise ValueError("use_copy=True requires copy_id.")
        else:
            if len(self.source_ids) != self.source_len:
                raise ValueError(
                    f"use_copy=False requires source_ids of length {self.source_len}, got {len(self.source_ids)}."
                )

    @property
    def in_edu_leaf(self) -> bool:
        return bool(self.stack) and self.stack[-1].kind == "leaf"

    def is_terminal(self) -> bool:
        return self.terminated

    @property
    def _is_words_labels(self) -> bool:
        return bool(self.word_label_ids)

    def _label_continuations(self, cursor: Tuple[int, ...]) -> FrozenSet[int]:
        """Trie continuations of `cursor`: the token ids that legally extend the
        in-progress multi-token label. Empty once `cursor` is a complete label
        (prefix-free, so a complete label is never a proper prefix)."""
        n = len(cursor)
        return frozenset(seq[n] for seq in self.word_label_ids if len(seq) > n and seq[:n] == cursor)

    def label_next_ids(self) -> FrozenSet[int]:
        """Legal label token ids at the current position: in token mode the whole
        single-id `label_ids`; in words mode the trie continuations of the
        in-progress label (the labels' first tokens when not mid-label)."""
        if self._is_words_labels:
            return self._label_continuations(self.label_cursor)
        return self.label_ids

    def _apply_label_token(self, action_id: int) -> "SexpDecodingState":
        """Consume one token of a multi-token relation-word label (words mode).
        The first token commits a preorder frame to internal; the token that
        completes the label sets `label_emitted`. `label_cursor` tracks progress
        and resets to () on completion."""
        new_cursor = self.label_cursor + (action_id,)
        complete = new_cursor in self.word_label_ids
        top = self.stack[-1]
        if self.traversal_order == "preorder" and not self.label_cursor:
            new_top = replace(top, kind="internal", children_emitted=0, label_emitted=complete)
        else:
            new_top = replace(top, label_emitted=complete)
        return replace(
            self,
            stack=self.stack[:-1] + (new_top,),
            label_cursor=() if complete else new_cursor,
        )

    @property
    def remaining_content(self) -> int:
        """Text (non-blank) source positions not yet consumed by the cursor.
        Every leaf that still has to START must consume at least one of these,
        so this is the budget the obligation gates spend against."""
        return self.source_len - self.cursor - sum(1 for p in self.blank_positions if p >= self.cursor)

    def _pending_leaf_obligation(self) -> int:
        """Minimum number of leaves that must still START (each consuming at
        least one not-yet-consumed content position) to legally complete every
        currently-open frame.

        The stack is nested (`stack[0]` is the root, `stack[-1]` the innermost
        currently-open child), so this is NOT a naive per-frame sum: a child's
        leaves are PART of its parent's child-obligation, not additional to it.
        Walk innermost -> outermost. The innermost frame's own minimum:
          * leaf  -> 0 (already started; a leaf frame only exists once >=1
                       content token has been emitted), or 1 while every token
                       it holds is blank (it still owes a text position)
          * None  -> 1 (minimally becomes a 1-token leaf)
          * internal with c children emitted -> (2 - c) remaining child
                       subtrees, each >= 1 leaf
        Each ancestor already has one child currently open (the frame below it
        on the stack, which the inner term accounts for), so it contributes
        only its OTHER not-yet-opened children: `(2 - c) - 1 = (1 - c)`.
        Ancestors are always `internal` (a frame acquires a stacked child only
        via OPEN, which resolves a postorder None frame to internal and a
        preorder frame is internal once its label fired).

        Maintaining the invariant `remaining_content >= _pending_leaf_obligation()`
        at every in-tree state is what makes the OPEN gates below deadlock-free:
        it guarantees an internal node never reaches `children_emitted == 1` with
        the source exhausted (which would be unable to OPEN its 2nd child and
        unable to CLOSE).
        """
        if not self.stack:
            return 0
        inner = self.stack[-1]
        if inner.kind == "leaf":
            total = 0 if inner.leaf_has_text else 1
        elif inner.kind == "internal":
            total = max(0, 2 - inner.children_emitted)
        else:  # kind is None
            total = 1
        for anc in self.stack[:-1]:
            # anc.kind is "internal" by construction (see docstring). The open
            # child below it is one slot; it still owes (2 - c) - 1 children.
            total += max(0, (2 - anc.children_emitted) - 1)
        return total

    def _can_open_subtree(self) -> bool:
        """Whether opening a fresh child subtree (a new leaf, minimally) here
        is affordable, i.e. it won't obligate more leaves than there is
        remaining content to fill.

        Cases (derived in `_pending_leaf_obligation`'s docstring):
          * innermost is `internal` (preorder/postorder internal-node child
            slot): the new child is one of the (2 - c) leaves already counted,
            so the post-open obligation equals the current one. Affordable iff
            `remaining_content >= _pending_leaf_obligation()`. Under the
            maintained invariant this always holds, but it is also the backstop
            that forbids OPEN once content is exhausted.
          * innermost is `None` (postorder fresh frame deciding leaf-vs-
            internal): opening turns a 1-leaf obligation into a 2-leaf one, so
            the post-open obligation is current + 1. Affordable iff
            `remaining_content >= _pending_leaf_obligation() + 1`. (Content to
            start the SAME frame as a leaf is still offered separately, so
            forbidding OPEN here never empties the legal set.)
          * empty stack (pre-root OPEN): a tree needs >= 1 leaf, so iff
            `source_len >= 1`.
        """
        if not self.stack:
            return self.source_len >= 1
        inner = self.stack[-1]
        need = self._pending_leaf_obligation()
        if inner.kind is None:
            need += 1  # this frame would go from a 1-leaf to a 2-leaf node
        return self.remaining_content >= need

    def legal_actions(self) -> FrozenSet[int]:
        if self.terminated:
            return frozenset()

        # Mid multi-token label (words mode): only the trie continuations of the
        # in-progress label are legal until it completes. Labels never touch the
        # content/leaf budgets, so this is the sole gating while mid-label.
        if self.label_cursor:
            return self.label_next_ids()

        legal: List[int] = []

        # EOS: after root has been fully emitted and source exhausted.
        if self.depth == 0 and self.cursor == self.source_len and self.root_emitted:
            legal.append(self.eos_id)
            return frozenset(legal)

        # Pre-root: only '(' can start the tree. OPEN requires there be at
        # least one source position to fill the tree's (minimally one) leaf.
        if self.depth == 0 and not self.root_emitted:
            if self._can_open_subtree():
                legal.append(self.open_id)
            return frozenset(legal)

        # Inside an open span. Look at the innermost frame.
        top = self.stack[-1]

        if top.kind is None:
            # Just opened. Decide what's legal based on traversal_order.
            if self.traversal_order == "preorder":
                # Internal node: starts with a label.
                # Leaf: starts with a source/copy token.
                # The label commits this frame to a 2-leaf internal node, so it
                # is gated on affording the 2nd leaf exactly like the postorder
                # OPEN below (`_can_open_subtree` adds +1 for a None frame). The
                # content path stays offered, so the legal set is never empty.
                if self.cursor < self.source_len:
                    legal.extend(self._content_legal())
                if self._can_open_subtree():
                    legal.extend(sorted(self.label_next_ids()))
            else:
                # Postorder. Internal: child first, which is '('.
                # Leaf: starts with a source/copy token.
                # OPEN here would commit this frame to being a 2-leaf internal
                # node, so it is gated on being able to afford a 2nd leaf; the
                # content path below still lets the model make it a 1-leaf
                # node, so the legal set is never empty.
                if self._can_open_subtree():
                    legal.append(self.open_id)
                if self.cursor < self.source_len:
                    legal.extend(self._content_legal())
            return frozenset(legal)

        if top.kind == "leaf":
            # In a leaf the choices are eat-content or CLOSE. Both normally stay
            # open (the model picks EDU length / where to split); two narrow
            # gates keep the source fully consumable, both expressed via
            # `obl_rest` = the tree's outstanding leaf-STARTS excluding this leaf
            # (a leaf frame contributes 0, so `_pending_leaf_obligation` already
            # equals the ancestors' remaining-child demand). The maintained
            # invariant is `remaining_content >= _pending_leaf_obligation()`
            # PLUS `obligation >= 1 whenever content remains` (some future leaf
            # must be able to absorb leftover positions).
            #
            #   * CONTENT gate: eating drops remaining_content by 1 without
            #     changing obl_rest, so it is illegal once
            #     `remaining_content == obl_rest` (every remaining position is
            #     reserved for a distinct future leaf-start). Offered iff
            #     `remaining_content > obl_rest`. A future sibling leaf can
            #     absorb arbitrarily many positions, so this does NOT force the
            #     current leaf to swallow surplus, it just stops it from eating
            #     INTO another leaf's last reserved position.
            #   * CLOSE gate: closing when `obl_rest == 0` (this leaf is the last
            #     open leaf-slot) while content remains would leave the tree
            #     complete-but-unclosable (root-close needs cursor==source_len),
            #     so CLOSE is withheld then and the leaf must keep eating.
            #
            # Exactly one of the two is ever the sole option, and that option is
            # always available, so under min_edu_length=1 the legal set is never
            # empty (verified exhaustively over all reachable states).
            #
            # LIMITATION (min_edu_length > 1): `_can_close` keeps CLOSE off below
            # min length, so a leaf can be forced to keep eating past the
            # `remaining_content == obl_rest` line, over-eating into a later
            # sibling's min-length budget and stranding a 1-child internal node
            # (empty legal set). Same hazard `GoldEduForcer` documents for
            # min_edu>1; every shipped config pins min_edu_length=1.
            #
            # Blank positions (see `blank_positions`) cost no budget, so eating
            # one is always safe. A leaf holding only blanks owes one text
            # position (counted in `_pending_leaf_obligation`), which eating a
            # text token pays, so the CONTENT gate's comparison discounts it; and
            # `_can_close` keeps it from closing. Its eat stays legal throughout.
            obl = self._pending_leaf_obligation()
            owed_self = 0 if top.leaf_has_text else 1
            obl_rest = obl - owed_self
            has_content = self.cursor < self.source_len
            if has_content and (self.cursor in self.blank_positions or self.remaining_content > obl_rest):
                legal.extend(self._content_legal())
            must_keep_eating = has_content and obl_rest == 0
            if top.leaf_token_count > 0 and self._can_close() and not must_keep_eating:
                legal.append(self.close_id)
            return frozenset(legal)

        # top.kind == 'internal'
        # For an internal node's child slot, opening the next child is one of
        # the (2 - children_emitted) leaves already counted in the obligation,
        # so `_can_open_subtree` reduces to `remaining_content >= obligation`
        # (the maintained invariant) and stays True until content is exhausted.
        # The invariant guarantees content is NOT exhausted while a child is
        # still owed, so OPEN is always offered here and the set is never empty.
        if self.traversal_order == "preorder":
            # Label has already been emitted (it's how we discovered we're
            # internal). Need 2 children before close.
            if top.children_emitted < 2:
                if self._can_open_subtree():
                    legal.append(self.open_id)
            else:
                if self._can_close():
                    legal.append(self.close_id)
            return frozenset(legal)
        # Postorder internal node.
        if top.children_emitted < 2:
            if self._can_open_subtree():
                legal.append(self.open_id)
            return frozenset(legal)
        if not top.label_emitted:
            legal.extend(sorted(self.label_next_ids()))
            return frozenset(legal)
        if self._can_close():
            legal.append(self.close_id)
        return frozenset(legal)

    def _content_legal(self) -> List[int]:
        """Source-content tokens legal *right now*.

        use_copy=True: the single `<copy>` token.
        use_copy=False: the one source subword id at `source_ids[cursor]`
            (COPY-via-constraint).
        """
        if self.cursor >= self.source_len:
            return []
        if self.use_copy:
            return [self.copy_id]  # type: ignore[list-item]
        return [self.source_ids[self.cursor]]

    def _can_close(self) -> bool:
        """Whether closing the innermost span is legal right now (i.e. the
        span structurally permits it). Root-close additionally requires the
        cursor to have reached source_len. Leaf-close additionally requires
        the leaf to contain at least `min_edu_length` content tokens, except
        at end-of-source (where the final EDU must be allowed to commit
        regardless)."""
        if not self.stack:
            return False
        top = self.stack[-1]
        if top.kind is None:
            return False
        if top.kind == "leaf":
            if top.leaf_token_count == 0 or not top.leaf_has_text:
                return False
            min_len = max(1, int(self.min_edu_length))
            at_end = self.cursor == self.source_len
            if top.leaf_token_count < min_len and not at_end:
                return False
        if top.kind == "internal":
            if top.children_emitted != 2:
                return False
            if self.traversal_order == "postorder" and not top.label_emitted:
                return False
        # If this would close the root, require source exhausted.
        if self.depth == 1 and self.cursor != self.source_len:
            return False
        return True

    def step(self, action_id: int) -> "SexpDecodingState":
        if self.terminated:
            raise ValueError("step() called on a terminated state.")

        if action_id == self.eos_id:
            if not (self.depth == 0 and self.cursor == self.source_len and self.root_emitted):
                raise ValueError("EOS emitted in non-terminal position.")
            return replace(self, terminated=True)

        # Pre-root: opening the tree.
        if self.depth == 0 and not self.root_emitted:
            if action_id == self.open_id:
                return replace(
                    self,
                    depth=1,
                    stack=(_Frame(),),
                )
            raise ValueError(f"Action {action_id} illegal at the pre-root position.")

        # Post-root with an empty stack: the only legal action was EOS (handled
        # above). Any other action here is illegal. Raise ValueError (not the
        # IndexError that `self.stack[-1]` would throw on the empty stack) so
        # callers that drive the PDA inside a `try/except ValueError` (the beam
        # loops) treat it as an illegal continuation and prune the beam rather
        # than crashing. Reachable when a caller feeds a fallback/padding token
        # (e.g. beam-search topk backfill) into a post-root state.
        if not self.stack:
            raise ValueError(f"Action {action_id} illegal at the post-root position.")

        top = self.stack[-1]

        # Mid multi-token label (words mode): the only legal continuation is the
        # next trie token. Handled before the structural branches so a label token
        # that happens to equal open/close/copy can't be misread mid-label.
        if self.label_cursor:
            if action_id in self._label_continuations(self.label_cursor):
                return self._apply_label_token(action_id)
            raise ValueError("Illegal token inside a multi-token label.")

        # Action: '('
        if action_id == self.open_id:
            new_top = top
            if top.kind == "internal":
                pass  # entering a child slot; the parent's kind is fixed
            elif top.kind is None:
                if self.traversal_order != "postorder":
                    raise ValueError("Opening '(' inside a preorder unknown-kind span is illegal.")
                new_top = replace(top, kind="internal", children_emitted=0)
            else:
                raise ValueError(f"Cannot open '(' inside a {top.kind!r} span.")
            return replace(
                self,
                depth=self.depth + 1,
                stack=self.stack[:-1] + (new_top, _Frame()),
            )

        # Action: ')'
        if action_id == self.close_id:
            if not self._can_close():
                raise ValueError("')' illegal at this position.")
            popped_stack = self.stack[:-1]
            new_depth = self.depth - 1
            root_now = new_depth == 0
            if popped_stack:
                parent = popped_stack[-1]
                if parent.kind is None and self.traversal_order == "postorder":
                    parent = replace(parent, kind="internal", children_emitted=1)
                else:
                    parent = replace(parent, children_emitted=parent.children_emitted + 1)
                popped_stack = popped_stack[:-1] + (parent,)
            return replace(
                self,
                depth=new_depth,
                stack=popped_stack,
                root_emitted=self.root_emitted or root_now,
            )

        # Action: label. Token mode: a single id in `label_ids`. Words mode: the
        # FIRST token of a multi-token relation-word label (subsequent tokens are
        # handled by the mid-label branch above). `label_next_ids()` unifies both.
        if action_id in self.label_next_ids():
            if self.traversal_order == "preorder":
                if top.kind is not None:
                    raise ValueError("Label emitted at a non-open slot in preorder.")
            else:
                if top.kind != "internal" or top.children_emitted != 2 or top.label_emitted:
                    raise ValueError("Label emitted at an illegal slot in postorder.")
            if self._is_words_labels:
                return self._apply_label_token(action_id)
            if self.traversal_order == "preorder":
                new_top = replace(top, kind="internal", label_emitted=True, children_emitted=0)
            else:
                new_top = replace(top, label_emitted=True)
            return replace(self, stack=self.stack[:-1] + (new_top,))

        # Action: content token (<copy> or the source id at the cursor)
        is_content = False
        if self.use_copy:
            if action_id == self.copy_id:
                is_content = True
        else:
            if self.cursor < self.source_len:
                is_content = action_id == self.source_ids[self.cursor]
        if is_content:
            if top.kind == "internal":
                raise ValueError("Source content emitted inside an internal node slot.")
            has_text = top.leaf_has_text or self.cursor not in self.blank_positions
            if top.kind is None:
                new_top = replace(top, kind="leaf", leaf_token_count=1, leaf_has_text=has_text)
            else:
                new_top = replace(top, leaf_token_count=top.leaf_token_count + 1, leaf_has_text=has_text)
            return replace(
                self,
                stack=self.stack[:-1] + (new_top,),
                cursor=self.cursor + 1,
            )

        raise ValueError(f"Action {action_id} is not in the legal set.")


class GoldEduForcer:
    """Drive a `SexpDecodingState` to emit exactly `n_edus_target` leaves
    matching the gold ranges, regardless of how (un)trained the model is,
    while leaving the TREE SHAPE to the model.

    Strategy: budget-range planning. Each open frame carries an EDU-budget
    range [lo, hi], the number of gold leaves its subtree may still hold,
    kept on a stack parallel to `state.stack`:

      * the root gets the exact budget [n, n]
      * an internal node's first child gets [1, hi - 1] (its future sibling
        needs at least one leaf)
      * once the first child closes having consumed k leaves, the second
        child gets [max(1, lo - k), hi - k] (exact whenever the parent was
        exact, which is how the total resolves to exactly n at the root)

    At a fresh (kind=None) frame the range decides what to force:

      * lo >= 2: the subtree must be internal. Force the internal-node
        starter (OPEN in postorder, the label slot in preorder).
      * hi == 1: the subtree must be a single leaf. Force content to start it.
      * lo == 1 and hi >= 2: DEFER to the model. Both leaf and internal are
        consistent with gold segmentation, and this choice is exactly the
        structural signal a gold-EDU Parseval should measure. (A forcer that
        never defers here fixes the shape and reduces `gold_edu_*` to a
        labeling metric, the pre-2026-07 behavior.)

    Inside a leaf, content is forced until the cursor reaches the current
    gold range's end, then CLOSE. So segmentation is gold by construction,
    every bracketing decision is the model's, and label slots stay the
    model's choice (narrowed to `label_ids`).

    Usage:
        forcer = GoldEduForcer(n_edus_target, gold_ranges)
        for step in ...:
            narrowed = forcer.narrowed_legal(state)
            ... mask logits to (legal & narrowed) if not None, argmax ...
            new_state = state.step(chosen_id)
            forcer.observe(state, new_state, chosen_id)
            state = new_state

    Assumes the driven state has `min_edu_length == 1` (the consumer pins it
    for the forced state). With `min_edu_length > 1` an earlier leaf can
    overshoot and exhaust the source before a later leaf can start, deadlocking
    the forcer into an OPEN-spin to max length; honoring min_edu>1 here would
    need a force-toward-close fallback rather than deferring to the model.
    """

    def __init__(self, n_edus_target: int, gold_ranges: List[tuple]) -> None:
        if n_edus_target != len(gold_ranges):
            raise ValueError(f"n_edus_target={n_edus_target} != len(gold_ranges)={len(gold_ranges)}.")
        # M6 guard: zero-width `(s, s)` ranges (an EDU that aligned to no
        # subword and fell back to an `(anchor, anchor)` range) and backward
        # (non-monotonic) starts would otherwise make the forcer spin OPEN on
        # a frame whose leaf can never receive content and can never close.
        # Drop any range that has no room left past the running monotonic
        # floor, so every surviving range is a non-empty, non-decreasing
        # forward span keyed to a REAL gold end (never a fabricated one).
        # Clamping the start up to the floor and then keeping the true gold
        # end `e` is safe; fabricating `end = start + 1` past the floor would
        # invent a leaf with no content target and re-trigger the OPEN-runaway
        # this guard exists to prevent. The per-parser range producers already
        # emit tiling ranges (the C1 alignment helper), so this is the
        # belt-and-suspenders guard in the shared forcer.
        sanitized: List[tuple] = []
        floor = 0
        for s, e in gold_ranges:
            s, e = int(s), int(e)
            if e <= s:
                continue  # zero-width / inverted gold range
            start = max(s, floor)
            if e <= start:
                continue  # range falls entirely behind the floor: no room
            sanitized.append((start, e))
            floor = e
        self.n_edus_target = len(sanitized)
        self.gold_ranges = sanitized
        self.closed_leaves = 0
        # _budgets[i] = [lo, hi, child_consumed] for the i-th open frame:
        # the range of gold leaves its subtree may hold, plus how many leaves
        # its already-closed children consumed. Maintained parallel to
        # `state.stack` by `observe`.
        self._budgets: List[List[int]] = []

    def clone(self) -> "GoldEduForcer":
        """Deep-enough copy for beam expansion: each beam drives its own forcer
        and `observe` mutates `closed_leaves` + `_budgets`, so sibling beams
        expanded from one parent must not share a forcer. The driven
        `SexpDecodingState` is immutable and needs no clone (see
        `SexpDecodeState.clone` in `serializations/sexp.py`). `gold_ranges` is a
        tuple list never mutated after `__init__`, so it is shared, not copied.
        `__new__` skips re-running `__init__` (which would re-sanitize the
        ranges and reset the progress counters)."""
        new = GoldEduForcer.__new__(GoldEduForcer)
        new.n_edus_target = self.n_edus_target
        new.gold_ranges = self.gold_ranges
        new.closed_leaves = self.closed_leaves
        new._budgets = [list(b) for b in self._budgets]
        return new

    def _current_target_end(self) -> Optional[int]:
        if self.closed_leaves >= self.n_edus_target:
            return None
        return self.gold_ranges[self.closed_leaves][1]

    def narrowed_legal(self, state: SexpDecodingState) -> NarrowedLegal:
        """Narrowing of `state.legal_actions()` consistent with the gold-EDU
        plan. Two return shapes:

          * None -> no narrowing (use the model's argmax over the full legal
            set).
          * frozenset[int] of full-vocab ids -> whitelist. The caller masks
            logits to (legal & this set) and argmaxes. A singleton is a hard
            force. (In practice this is never the empty set.)
        """
        if state.is_terminal():
            return None

        # Mid multi-token label (words mode): the legal set is already the label
        # trie's continuations. Let the model pick the relation freely; label
        # internals don't touch segmentation or tree shape.
        if state.label_cursor:
            return None

        legal = state.legal_actions()

        # Inside an active leaf: force content or CLOSE.
        if state.in_edu_leaf and self.closed_leaves < self.n_edus_target:
            target_end = self._current_target_end()
            if target_end is None:
                return None
            if state.cursor < target_end:
                if state.use_copy:
                    return frozenset({state.copy_id}) if state.copy_id in legal else None
                if state.cursor >= state.source_len:
                    return None
                content_id = state.source_ids[state.cursor]
                return frozenset({content_id}) if content_id in legal else None
            return frozenset({state.close_id}) if state.close_id in legal else None

        # Pre-root: force OPEN.
        if state.depth == 0 and not state.root_emitted:
            if self.n_edus_target == 0:
                return frozenset({state.eos_id}) if state.eos_id in legal else None
            return frozenset({state.open_id}) if state.open_id in legal else None

        if not state.stack:
            # Post-root: tree closed. Force EOS.
            if state.cursor == state.source_len and state.root_emitted:
                return frozenset({state.eos_id}) if state.eos_id in legal else None
            return None

        top = state.stack[-1]
        lo, hi = (
            (self._budgets[-1][0], self._budgets[-1][1])
            if self._budgets
            else (self.n_edus_target, self.n_edus_target)
        )

        if top.kind == "internal":
            if top.children_emitted < 2:
                return frozenset({state.open_id}) if state.open_id in legal else None
            if state.traversal_order == "postorder" and not top.label_emitted:
                # Let the model pick the label, constrained to legal label tokens
                # (single ids in token mode, label first-tokens in words mode).
                return frozenset(state.label_next_ids()) & legal
            return frozenset({state.close_id}) if state.close_id in legal else None

        # top.kind is None: fresh frame. The budget range decides.
        if hi <= 1:
            # Must be a leaf. Force the first content token: at a fresh frame
            # `legal_actions()` also offers structural starters (labels /
            # OPEN), so deferring here lets the model turn the intended leaf
            # into an internal node that can never fit its budget.
            if state.use_copy:
                return frozenset({state.copy_id}) if state.copy_id in legal else None
            if state.cursor >= state.source_len:
                # No source left to start a leaf with. Defer (should not arise
                # for a valid, M6-sanitized gold range).
                return None
            content_id = state.source_ids[state.cursor]
            return frozenset({content_id}) if content_id in legal else None
        if lo >= 2:
            # Must be internal.
            if state.traversal_order == "preorder":
                # In preorder the first action inside an internal node is the LABEL
                # (its first token in words mode).
                return frozenset(state.label_next_ids()) & legal
            # Postorder: first action inside an internal is OPEN of its first child.
            return frozenset({state.open_id}) if state.open_id in legal else None
        # lo == 1 and hi >= 2: leaf vs internal is the model's structural
        # choice. Defer.
        return None

    def observe(self, before: SexpDecodingState, after: SexpDecodingState, action_id: int) -> None:
        """Update the parallel budget stack to mirror `after.stack`."""
        before_top = before.stack[-1] if before.stack else None
        # Leaf close detection.
        if action_id == before.close_id and before_top is not None and before_top.kind == "leaf":
            self.closed_leaves += 1

        before_depth = len(before.stack)
        after_depth = len(after.stack)
        if after_depth > before_depth:
            # OPEN pushed a frame.
            if not self._budgets:
                # Pre-root OPEN: the root must hold exactly n leaves.
                self._budgets.append([self.n_edus_target, self.n_edus_target, 0])
                return
            parent = self._budgets[-1]
            lo_p, hi_p, used_p = parent
            if before_top is not None and before_top.kind == "internal" and before_top.children_emitted == 1:
                # Second child: the first child's consumption resolves the range.
                lo_c = max(1, lo_p - used_p)
                hi_c = hi_p - used_p
            else:
                # First child (the parent was a fresh frame this OPEN committed
                # to internal, or an internal with no children yet). The parent
                # now holds >= 2 leaves; the child leaves >= 1 for its sibling.
                parent[0] = max(lo_p, 2)
                lo_c = 1
                hi_c = hi_p - 1
            self._budgets.append([max(1, lo_c), max(1, hi_c), 0])
        elif after_depth < before_depth:
            # CLOSE popped a frame. Propagate its consumed leaf count.
            closed = self._budgets.pop() if self._budgets else None
            if self._budgets and closed is not None:
                is_leaf = before_top is not None and before_top.kind == "leaf"
                self._budgets[-1][2] += 1 if is_leaf else closed[2]
        else:
            # Same depth: a preorder label commits a fresh frame to internal.
            # Detect the kind transition directly (None -> internal) so it works
            # for both single-id labels and the first token of a words-mode
            # multi-token label (`action_id in before.label_ids` would miss the
            # latter, since words mode leaves `label_ids` empty).
            after_top = after.stack[-1] if after.stack else None
            if (
                self._budgets
                and before_top is not None
                and before_top.kind is None
                and after_top is not None
                and after_top.kind == "internal"
            ):
                self._budgets[-1][0] = max(self._budgets[-1][0], 2)
