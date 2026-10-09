"""Shared plumbing for the `icl` predict / eval commands (both in this one
package, so this is a within-parser helper, not cross-parser factoring).
"""

from __future__ import annotations

import json
from pathlib import Path

from tonga import Params

from dipo.common.log import wrote
from dipo.rst.data.reader import infer_relation_types
from dipo.rst.parsers.icl.configuration_icl import IclConfig
from dipo.rst.parsers.icl.modeling_icl import IclParser


def load_config(config_path: str) -> IclConfig:
    """Load the jsonnet config and infer the relation inventory from train+dev
    (the same source the trained parsers use, so the label set is comparable)."""
    cfg = IclConfig.from_dict(Params.from_file(config_path).as_dict(quiet=True))
    dirs = [cfg.train_dir, cfg.dev_dir]
    cfg.relation_types = infer_relation_types(dirs, relation_map=cfg.relation_map)
    return cfg


def build_parser(cfg: IclConfig) -> IclParser:
    return IclParser(cfg)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))
    wrote(str(path))


def write_text(path: Path, text: str) -> None:
    path.write_text(text)
    wrote(str(path))


def glob_docs(path: str, patterns: tuple[str, ...]) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    out: list[str] = []
    for pat in patterns:
        out += sorted(str(x) for x in p.glob(pat))
    return out
