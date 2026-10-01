"""Checkpoint-resolution and model-loading helpers for RST parser CLIs."""

import dataclasses
import os
import sys
from typing import TypeVar

import torch
from tonga import Params

from iudex.common.log import console
from iudex.common.training import derive_run_id, load_model_state
from iudex.rst import HASH_EXCLUDE

ConfigT = TypeVar("ConfigT")
ParserT = TypeVar("ParserT", bound=torch.nn.Module)


def resolve_checkpoint(
    config_path: str | None,
    checkpoint_path: str | None,
    config_cls: type,
    parser_name: str,
) -> str:
    """Return the .pt path to load. Exactly one of `config_path` /
    `checkpoint_path` must be set. With `config_path`, derive the run dir
    from the resolved config and look up `best_model.pt`. Exits non-zero
    with a helpful message if missing.
    """
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            console.print(f"[bold red]Checkpoint not found:[/bold red] [path]{checkpoint_path}[/path]")
            sys.exit(1)
        return checkpoint_path

    cfg = config_cls.from_dict(Params.from_file(config_path).as_dict(quiet=True))
    run_id, _ = derive_run_id(dataclasses.asdict(cfg), cfg.run_name, hash_exclude=HASH_EXCLUDE)
    run_dir = os.path.join(cfg.checkpoint_dir, run_id)
    derived_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(derived_path):
        console.print(
            f"[bold red]No trained model found for this config.[/bold red]\n"
            f"  Expected: [path]{derived_path}[/path]\n"
            f"  Train first with:\n"
            f"    python -m iudex {parser_name} train {config_path}"
        )
        sys.exit(1)
    return derived_path


def load_parser_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    config_cls: type[ConfigT],
    parser_cls: type[ParserT],
    *,
    compile_encoder: bool = False,
) -> ParserT:
    """Rehydrate a parser from a `.pt` checkpoint into eval mode on `device`.
    Trainable-only checkpoints (see `save_checkpoint`) work here because
    `parser_cls(cfg)` pulls fresh base weights from HF before the non-strict
    load overlays the trained parameters."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = config_cls.from_dict(checkpoint["config"])
    model = parser_cls(cfg, compile_encoder=compile_encoder)
    load_model_state(model, checkpoint)
    return model.to(device).eval()


def resolve_source(
    config_path: str | None,
    checkpoint_path: str | None,
    hub_id: str | None,
    config_cls: type,
    parser_name: str,
) -> tuple[str, str]:
    """Pick where to load the parser from. Returns (kind, value):

      - `"local"`: `value` is a local `.pt` path (→ `load_parser_from_checkpoint`).
      - `"hub"`:   `value` is a Hub repo id (→ `load_parser_from_pretrained`).

    Exactly one of `config_path` / `checkpoint_path` / `hub_id` must be set.
    """
    if hub_id is not None:
        return "hub", hub_id
    return "local", resolve_checkpoint(config_path, checkpoint_path, config_cls, parser_name)
