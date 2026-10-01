"""Exceptions raised by the generative parser at eval/predict time."""


class DecodeInvariantError(RuntimeError):
    """The validity mask and the serialization's state machine disagree: the mask
    admitted an action `apply` then rejected, or a live state offered no legal
    continuation at all (deadlock).

    Decoding is always constrained (there is no unconstrained mode), and
    `beam_topk_step` scores mask-illegal actions at -inf, so a FINITE-scored action
    the state machine rejects can only mean a bug in the mask or the automaton.
    Neither condition has a correct recovery: dropping the beam or falling back to
    a single-EDU tree would let a legality bug score as if the model produced that
    output, which is exactly the silent corruption `OverLengthError` exists to
    prevent. Distinct from `OverLengthError`: this is not a context-size problem
    and raising the length caps will not help.
    """


class OverLengthError(RuntimeError):
    """A document does not fit in the model's context: either the source exceeds
    `max_input_length`, or the decode ran the full `max_output_length` budget
    without completing a valid tree.

    This is always a hard crash, never a silent truncation or best-effort repair.
    A truncated/repaired tree scores as if the model genuinely produced it (e.g. a
    single-EDU fallback craters recall), which corrupts the metric invisibly. The
    training path already raises on over-length trees (train_gen.py); eval/predict
    mirrors that. Fix by raising `max_input_length` / `max_output_length` to fit the
    longest document (or drop the offending doc on purpose), then re-run.
    """
