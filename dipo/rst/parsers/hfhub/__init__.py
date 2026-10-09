"""HuggingFace Hub distribution for dipo RST parsers.

Public API re-exported here so callers can write
`from dipo.rst.parsers.hfhub import load_parser_from_pretrained` instead of
reaching into `hub.py`. `push.py` is the shared CLI entry point dispatched
from `dipo <parser> push ...` (see `SHARED_COMMANDS` in `dipo/__main__.py`).
"""

from dipo.rst.parsers.hfhub.datasets import DATASETS, lookup
from dipo.rst.parsers.hfhub.hub import (
    load_parser_from_pretrained,
    push_parser_to_hub,
    render_model_card,
)

__all__ = [
    "DATASETS",
    "load_parser_from_pretrained",
    "lookup",
    "push_parser_to_hub",
    "render_model_card",
]
