"""HuggingFace Hub distribution for dipo RST parsers.

On load, both the parser repo and the underlying encoder (e.g.
`xlm-roberta-base`) are fetched and cached. The encoder weights are then
immediately overwritten by `load_state_dict` (strict mode catches
architecture drift).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, TypeVar

import torch
import torch.nn as nn
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

from dipo.common.log import console, success
from dipo.rst.parsers.common.inference import load_parser_from_checkpoint
from dipo.rst.parsers.hfhub.datasets import lookup as lookup_dataset

ConfigT = TypeVar("ConfigT")
ParserT = TypeVar("ParserT", bound=nn.Module)

HUB_WEIGHTS_NAME = "best_model.pt"
HUB_CONFIG_NAME = "config.json"
HUB_CARD_NAME = "README.md"

_HUB_ID_PATTERN = re.compile(r"^[\w.\-]+/[\w.\-]+$")

# Per-parser model-card metadata. `paper_*` keys present iff this parser
# re-implements an external paper (toggles intro wording + bibtex block).
_PARSER_META: dict[str, dict[str, str]] = {
    "dmrst": {
        "human_name": "DMRST parser",
        "module_path": "dipo.rst.parsers.dmrst.modeling_dmrst",
        "class_name": "DMRSTParser",
        "description": "an end-to-end RST parser with joint EDU segmentation, a GRU decoder, and pointer attention",
        "paper_title": "DMRST: A Joint Framework for Document-Level Multilingual RST Discourse Segmentation and Parsing",
        "paper_authors": "Zhengyuan Liu, Ke Shi, Nancy F. Chen",
        "paper_venue": "CODI 2021",
        "paper_url": "https://aclanthology.org/2021.codi-main.15/",
    },
    "topdown_biaffine": {
        "human_name": "Top-down Biaffine RST parser",
        "module_path": "dipo.rst.parsers.topdown_biaffine.modeling_topdown_biaffine",
        "class_name": "TopdownBiaffineParser",
        "description": "a greedy top-down RST parser with biaffine split and label scoring (assumes gold EDU segmentation)",
        "paper_title": "A Simple and Strong Baseline for End-to-End Neural RST-style Discourse Parsing",
        "paper_authors": "Naoki Kobayashi, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura, Masaaki Nagata",
        "paper_venue": "Findings of EMNLP 2022",
        "paper_url": "https://aclanthology.org/2022.findings-emnlp.501/",
    },
    "sr_biaffine": {
        "human_name": "Shift-reduce Biaffine RST parser",
        "module_path": "dipo.rst.parsers.sr_biaffine.modeling_sr_biaffine",
        "class_name": "SRBiaffineParser",
        "description": "a transition-based shift-reduce RST parser with an FFN action head and a biaffine "
        "reduce-label head (assumes gold EDU segmentation)",
        "paper_title": "A Simple and Strong Baseline for End-to-End Neural RST-style Discourse Parsing",
        "paper_authors": "Naoki Kobayashi, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura, Masaaki Nagata",
        "paper_venue": "Findings of EMNLP 2022",
        "paper_url": "https://aclanthology.org/2022.findings-emnlp.501/",
    },
    # dipo original (no external paper): the `else` branch of render_model_card
    # supplies the intro/citation, so no `paper_*` keys.
    "gen": {
        "human_name": "generative RST parser",
        "module_path": "dipo.rst.parsers.gen.modeling_gen",
        "class_name": "GenParser",
        "description": "an end-to-end RST parser that fine-tunes a language model to generate a linearized tree "
        "(a bottom-up shift-reduce action sequence or a nested s-expression) and recovers segmentation + structure "
        "from the decoded string; the backbone (encoder-decoder or causal) and serialization are config-selected",
    },
}


def _is_hub_id(s: str) -> bool:
    """True if `s` looks like `org/name` and isn't an existing path or `.pt` file."""
    if os.path.exists(s) or s.endswith(".pt"):
        return False
    return bool(_HUB_ID_PATTERN.match(s))


def load_parser_from_pretrained(
    repo_or_path: str,
    *,
    parser_cls: type[ParserT],
    config_cls: type[ConfigT],
    device: torch.device,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    compile_encoder: bool = False,
) -> ParserT:
    """Load a parser from a Hub repo id, a local run directory, or a `.pt` file.

    Hub ids pull `best_model.pt` / `config.json` / `README.md` via
    `snapshot_download`. Directories look for `best_model.pt`. `.pt` paths
    load as-is.
    """
    if _is_hub_id(repo_or_path):
        snapshot_dir = snapshot_download(
            repo_id=repo_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=[HUB_WEIGHTS_NAME, HUB_CONFIG_NAME, HUB_CARD_NAME],
        )
        checkpoint_path = os.path.join(snapshot_dir, HUB_WEIGHTS_NAME)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Hub repo {repo_or_path!r} has no {HUB_WEIGHTS_NAME}")
    elif os.path.isdir(repo_or_path):
        checkpoint_path = os.path.join(repo_or_path, HUB_WEIGHTS_NAME)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No {HUB_WEIGHTS_NAME} in {repo_or_path}")
    else:
        checkpoint_path = repo_or_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
    return load_parser_from_checkpoint(checkpoint_path, device, config_cls, parser_cls, compile_encoder=compile_encoder)


def push_parser_to_hub(
    checkpoint_path: str,
    repo_id: str,
    *,
    parser_kind: str,
    private: bool = False,
    commit_message: str = "Upload parser",
    token: str | bool | None = None,
    extra_card_fields: dict[str, Any] | None = None,
    example_text: str | None = None,
    language: str | None = None,
) -> str:
    """Upload `best_model.pt`, `config.json`, and a generated `README.md`
    to `repo_id` in a single commit. Returns the repo URL.
    """
    if parser_kind not in _PARSER_META:
        raise ValueError(f"Unknown parser_kind: {parser_kind!r}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    checkpoint_meta = {
        k: checkpoint.get(k) for k in ("best_val", "global_step", "epoch", "config_hash") if k in checkpoint
    }

    # Pick up adjacent final_metrics.json if the train script wrote one. It
    # gives us dev + test corpus metrics keyed by split. Absence is fine.
    run_dir = os.path.dirname(checkpoint_path)
    final_metrics_path = os.path.join(run_dir, "final_metrics.json")
    final_metrics: dict | None = None
    if os.path.exists(final_metrics_path):
        try:
            with open(final_metrics_path, encoding="utf-8") as f:
                final_metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            final_metrics = None

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    # Prefer the adjacent run-dir config.json (byte-for-byte audit), fall back
    # to serializing the dict embedded in the checkpoint.
    adjacent_config = os.path.join(run_dir, HUB_CONFIG_NAME)
    if os.path.exists(adjacent_config):
        config_blob: str | bytes = adjacent_config
    else:
        config_blob = json.dumps(config, indent=2, default=str).encode()

    card = render_model_card(
        parser_kind=parser_kind,
        config=config,
        checkpoint_meta=checkpoint_meta,
        final_metrics=final_metrics,
        repo_id=repo_id,
        extra=extra_card_fields,
        example_text=example_text,
        language=language,
    )

    # Single commit so an interrupted push can't leave the repo with new
    # weights against a stale README / config.
    console.print(f"Uploading 3 files to [cyan]{repo_id}[/cyan] in a single commit...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=HUB_WEIGHTS_NAME, path_or_fileobj=checkpoint_path),
            CommitOperationAdd(path_in_repo=HUB_CONFIG_NAME, path_or_fileobj=config_blob),
            CommitOperationAdd(path_in_repo=HUB_CARD_NAME, path_or_fileobj=card.encode()),
        ],
        commit_message=commit_message,
        token=token,
    )

    url = f"https://huggingface.co/{repo_id}"
    success(f"Pushed to {url}")
    return url


def _format_relation_labels(config: dict[str, Any]) -> str:
    """Render the 'Relation labels' line. Sorted alphabetically for stable output."""
    relation_types = config.get("relation_types") or []
    relation_map = config.get("relation_map")

    distinct: list[str] = []
    seen: set[str] = set()
    for entry in relation_types:
        name = entry[0] if isinstance(entry, (list, tuple)) else str(entry)
        if name not in seen:
            seen.add(name)
            distinct.append(name)
    distinct.sort()

    if relation_map is None:
        descriptor = f"{len(distinct)} labels"
    else:
        descriptor = (
            f"Mapped from an original label inventory with {len(relation_map)} items "
            f"to {len(distinct)} labels. Mapped labels"
        )

    if not distinct:
        return f"**Relation labels:** {descriptor}.\n"
    label_list = ", ".join(f"`{n}`" for n in distinct)
    return f"**Relation labels:** {descriptor}:\n\n{label_list}\n"


# (column label, keys in final_metrics.json, first one present wins). The DMRST and
# gold-EDU parsers write their gold-EDU scores without the `gold_edu_` prefix.
_METRIC_COLUMNS = [
    ("Seg", ("seg_f1",)),
    ("E2E Span", ("e2e_span_f1",)),
    ("E2E Nuc", ("e2e_nuc_f1",)),
    ("E2E Rel", ("e2e_rel_f1",)),
    ("E2E Full", ("e2e_full_f1",)),
    ("Gold-EDU Span", ("gold_edu_span_f1", "span_f1")),
    ("Gold-EDU Nuc", ("gold_edu_nuc_f1", "nuc_f1")),
    ("Gold-EDU Rel", ("gold_edu_rel_f1", "rel_f1")),
    ("Gold-EDU Full", ("gold_edu_full_f1", "full_f1")),
]


def _metric(split_metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = split_metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _render_data_section(
    *,
    config: dict[str, Any],
    train_dir: str,
    checkpoint_meta: dict[str, Any],
    final_metrics: dict[str, dict[str, float]] | None,
) -> str:
    """Render the "Data" section: dataset info, relation labels, corpus metrics."""
    parts: list[str] = ["## Data\n"]

    dataset = lookup_dataset(train_dir)
    if dataset is not None:
        name = dataset.get("name", "?")
        url = dataset.get("url")
        lang = dataset.get("language")
        header = f"**[{name}]({url})**" if url else f"**{name}**"
        if lang:
            header += f" ({lang})"
        parts.append(header + ".\n")
        if dataset.get("description"):
            parts.append(dataset["description"] + "\n")
    elif train_dir:
        parts.append(f"Trained on `{train_dir}` (no entry in the dipo dataset registry).\n")

    parts.append("\n" + _format_relation_labels(config))

    metric_name = config.get("val_metric_name", "?")
    splits = [sp for sp in ("dev", "test") if isinstance((final_metrics or {}).get(sp), dict)]
    columns = [
        (label, keys)
        for label, keys in _METRIC_COLUMNS
        if any(_metric(final_metrics[sp], keys) is not None for sp in splits)
    ]
    if splits and columns:
        parts.append("\n### Metrics\n\n")
        parts.append("| Split | " + " | ".join(label for label, _ in columns) + " |\n")
        parts.append("| --- | " + " | ".join("---" for _ in columns) + " |\n")
        for sp in splits:
            cells = []
            for _, keys in columns:
                value = _metric(final_metrics[sp], keys)
                cells.append("-" if value is None else f"{100 * value:.1f}")
            parts.append(f"| {sp} | " + " | ".join(cells) + " |\n")
        parts.append(
            "\nF1 x 100 under original Parseval, with trivial leaf spans excluded. "
            "*E2E* scores parse trees predicted from raw text, segmentation included. "
            "*Gold-EDU* scores them given the gold EDU segmentation. "
        )
        if any(keys == ("seg_f1",) for _, keys in columns):
            parts.append(
                "*Seg* is the per-token BeginSeg F1 computed during training, "
                "which can read up to about a point below the official DISRPT scorer's value. "
            )
        parts.append(f"The checkpoint was selected on dev `{metric_name}`.\n")
    elif final_metrics:
        parts.append("\n### Metrics\n\n")
        parts.append("- _(metrics were recorded in an unrecognized format)_\n")
    else:
        best_val = checkpoint_meta.get("best_val")
        parts.append("\n### Metrics\n\n")
        if isinstance(best_val, (int, float)) and best_val >= 0:
            parts.append(f"- **Dev {metric_name}**: {best_val:.4f}\n")
            parts.append("- _(no `final_metrics.json` sidecar, test metric not recorded)_\n")
        else:
            parts.append("- _(no metrics recorded on this checkpoint)_\n")

    return "".join(parts)


def render_model_card(
    *,
    parser_kind: str,
    config: dict[str, Any],
    checkpoint_meta: dict[str, Any],
    final_metrics: dict[str, dict[str, float]] | None = None,
    repo_id: str,
    extra: dict[str, Any] | None = None,
    example_text: str | None = None,
    language: str | None = None,
) -> str:
    """Generate the README.md model card. Plain string templates, no jinja.

    `example_text` replaces the English sentence in the usage snippets (give a
    model's own language, unspaced for Chinese); `language` is its ISO 639-1 code
    for the card's front matter.

    `final_metrics`, when present, maps split name → metric-dict (e.g.
    `{"dev": {...}, "test": {...}}`). It supersedes `checkpoint_meta['best_val']`
    for the displayed metrics table.
    """
    from dipo.rst.parsers import PARSERS

    meta = _PARSER_META[parser_kind]
    encoder = config.get("model_name", "")
    train_dir = config.get("train_dir") or ""

    # A parser exposes `predict_from_text` iff it `supports_text` AND, for a
    # parser gated on a `segmentation` sub-config (dmrst), that
    # sub-config is non-null. The generative parsers have no `segmentation`
    # field and always support text, so absence of the key means text-capable.
    spec = PARSERS.get(parser_kind)
    supports_text = (spec.supports_text if spec is not None else True) and (
        "segmentation" not in config or config["segmentation"] is not None
    )
    if supports_text:
        text = example_text or (
            "Although the experiment was carefully designed, the results were inconclusive. "
            "We plan to repeat it tonight."
        )
        predict_snippet = f"tree = parser.predict_from_text(\n    {json.dumps(text, ensure_ascii=False)}\n)"
        cli_snippet = (
            f"dipo {parser_kind} predict \\\n"
            f"    --hub-id {repo_id} \\\n"
            f"    --text '{text}'"
        )
        cli_batch_note = (
            "To parse a raw `.txt` file or a directory of them instead, use `--text-file <path> --output-dir out/`.\n"
        )
    else:
        predict_snippet = (
            "# This parser requires gold EDU segmentation, so the input must be an RS3/RS4 file.\n"
            "from dipo.rst.data.reader import read_rst_file\n"
            "gold = read_rst_file(\n"
            '    "doc.rs3",\n'
            "    relation_types=parser.config.relation_types,\n"
            "    relation_map=parser.config.relation_map,\n"
            ")\n"
            "tree = parser.predict(gold)"
        )
        cli_snippet = (
            f"dipo {parser_kind} predict \\\n"
            f"    --hub-id {repo_id} \\\n"
            "    --input <doc.rs3> \\\n"
            "    --output-dir out/"
        )
        cli_batch_note = "`--input` also accepts a directory of `.rs3` / `.rs4` files.\n"

    extras_section = ""
    if extra:
        extras_section = "## Notes\n\n" + "".join(f"- **{k}**: {v}\n" for k, v in extra.items()) + "\n"

    config_block = json.dumps(config, indent=2, default=str)

    front_matter = (
        "---\n"
        "library_name: dipo\n"
        f"base_model: {encoder}\n"
        + (f"language:\n  - {language}\n" if language else "")
        + "tags:\n"
        "  - dipo\n"
        "  - iudex\n"
        "  - discourse-parsing\n"
        "  - rst\n"
        f"  - {parser_kind}\n"
        "---\n\n"
    )

    data_section = _render_data_section(
        config=config,
        train_dir=train_dir,
        checkpoint_meta=checkpoint_meta,
        final_metrics=final_metrics,
    )

    if meta.get("paper_url"):
        intro = (
            f"A pretrained [{meta['human_name']}]({meta['paper_url']}) trained with "
            "[Dipo](https://github.com/larc-iu/dipo)."
        )
        citation_block = f"""If you use this model, please cite both the underlying paper:

```bibtex
@inproceedings{{{parser_kind}_paper,
  title = {{{meta["paper_title"]}}},
  author = {{{meta["paper_authors"]}}},
  booktitle = {{{meta["paper_venue"]}}},
  url = {{{meta["paper_url"]}}},
}}
```

And the Dipo library:

```bibtex
@misc{{gessler-dipo-2026,
  author       = {{Gessler, Luke}},
  title        = {{{{Dipo: The Discourse Parsing Omnibus}}}},
  year         = {{2026}},
  howpublished = {{\\url{{https://github.com/larc-iu/dipo}}}},
}}
```"""
    else:
        intro = (
            f"A pretrained {meta['human_name']} ({meta['description']}) "
            "developed and trained in [Dipo](https://github.com/larc-iu/dipo)."
        )
        citation_block = """If you use this model, please cite the Dipo library:

```bibtex
@misc{gessler-dipo-2026,
  author       = {Gessler, Luke},
  title        = {{Dipo: The Discourse Parsing Omnibus}},
  year         = {2026},
  howpublished = {\\url{https://github.com/larc-iu/dipo}},
}
```"""

    body = f"""# {repo_id}

{intro}

This model uses [`{encoder}`](https://huggingface.co/{encoder}) as its underlying encoder.

{data_section}
## Usage

Install the library first with `pip install dipo`.

### CLI

```
{cli_snippet}
```

{cli_batch_note}
### Python

```python
from {meta["module_path"]} import {meta["class_name"]}

parser = {meta["class_name"]}.from_pretrained("{repo_id}")
{predict_snippet}
print(tree.to_rs4_string())
```

## Citation

{citation_block}
{extras_section}## Full training configuration

See below for the full training configuration this model was trained with.

```json
{config_block}
```
"""
    return front_matter + body
