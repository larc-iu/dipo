"""Unit tests for the `dipo gen eval` --inference-method parser."""

import pytest

from dipo.rst.parsers.gen.eval_gen import parse_inference_methods


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("greedy", [1]),
        ("beam-6", [6]),
        ("greedy,beam-6", [1, 6]),
        ("beam-6,greedy", [6, 1]),  # order preserved
        ("beam-4, beam-8", [4, 8]),  # whitespace around methods ignored
        (" greedy , beam-2 ", [1, 2]),
    ],
)
def test_valid_specs(spec, expected):
    assert parse_inference_methods(spec) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "",  # empty
        "   ",  # blank
        "beam",  # missing width
        "beam-",  # missing digits
        "beam-0",  # width < 2
        "beam-1",  # width 1 is greedy, must use 'greedy'
        "beam-x",  # non-numeric
        "beam--3",  # sign / non-digit
        "beam-3.5",  # not an integer
        "greedy,greedy",  # duplicate
        "beam-6,beam-6",  # duplicate
        "sample",  # unknown method
        "greedy,",  # trailing empty token
        ",greedy",  # leading empty token
    ],
)
def test_malformed_specs_raise(spec):
    with pytest.raises(ValueError):
        parse_inference_methods(spec)
