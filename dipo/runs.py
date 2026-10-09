"""Inspect and manage dipo training runs.

Reads the sidecars written by `dipo.common.training`. Walks every framework
in `dipo.FRAMEWORKS` to tag rows by parser kind via `PARSERS[*].signature_field`.
"""

import argparse
import importlib
import json
import os
import re
import shutil
import sys
from datetime import datetime

from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

import dipo
from dipo.common.log import console, dim, success, wrote

# Last 12 hex chars of the run dir name are the config hash. Everything
# before is the optional run_name.
_HASH_SUFFIX_RE = re.compile(r"(?:^|-)([0-9a-f]{12})$")


def _all_parsers() -> dict:
    """Merge `PARSERS` across `dipo.FRAMEWORKS`. Aborts on a
    `signature_field` collision so `_infer_parser_kind` is well-defined."""
    merged: dict = {}
    by_sig: dict[str, str] = {}
    for fw_path in dipo.FRAMEWORKS:
        fw = importlib.import_module(fw_path)
        for name, spec in fw.PARSERS.items():
            merged[name] = spec
            owner = by_sig.get(spec.signature_field)
            if owner is not None and owner != name:
                sys.stderr.write(
                    f"dipo.runs: parsers {owner!r} and {name!r} both claim "
                    f"signature_field {spec.signature_field!r}. Pick a "
                    f"field unique to one of them.\n"
                )
                sys.exit(2)
            by_sig[spec.signature_field] = name
    return merged


def _infer_parser_kind(config: dict, parsers: dict) -> str:
    """Find the parser whose signature_field is present in `config`.

    Two parsers' signature_fields can both appear in one config (a superset config
    carries a sibling's field too). When that happens, first pick the parser whose
    signature_field appears on no OTHER parser's config dataclass (a truly
    distinguishing field). If still ambiguous, fall back to the parser whose default
    field set most closely matches the config's keys (minimum symmetric difference).
    Returns "?" on no match. (The generative parser `gen` has a unique `backbone`
    signature, so it resolves in the single-match fast path.)"""
    import dataclasses as _dc

    matches = [(name, spec) for name, spec in parsers.items() if spec.signature_field in config]
    if not matches:
        return "?"
    if len(matches) == 1:
        return matches[0][0]
    # Build per-parser default-field sets lazily. Only enter this branch
    # when there's actual ambiguity.
    fields_by_name: dict[str, set[str]] = {}
    for name, spec in parsers.items():
        try:
            cfg_cls = spec.load_config_cls()
            fields_by_name[name] = {f.name for f in _dc.fields(cfg_cls)}
        except Exception:
            fields_by_name[name] = set()

    distinguishing: list[str] = []
    for name, spec in matches:
        sig = spec.signature_field
        shared = any(sig in fields_by_name.get(other, set()) for other in parsers if other != name)
        if not shared:
            distinguishing.append(name)
    if len(distinguishing) == 1:
        return distinguishing[0]

    # Tie-break: parser whose default field set most closely matches the
    # config's keys (smallest symmetric difference). When two parsers differ
    # only by an added field, the config carrying that field matches the
    # superset parser's dataclass exactly while leaving the other one short.
    config_keys = set(config.keys())
    candidates = distinguishing if distinguishing else [name for name, _ in matches]
    scored = [(len(fields_by_name.get(name, set()) ^ config_keys), name) for name in candidates]
    scored.sort()
    return scored[0][1]


def _list_run_dirs(checkpoint_dir: str) -> list[str]:
    """Sorted basenames under `checkpoint_dir` that have a `config.json`."""
    if not os.path.isdir(checkpoint_dir):
        return []
    out = []
    for entry in sorted(os.listdir(checkpoint_dir)):
        if os.path.exists(os.path.join(checkpoint_dir, entry, "config.json")):
            out.append(entry)
    return out


def _resolve_run_id(checkpoint_dir: str, partial: str) -> str:
    """Unique run dir starting with `partial`. Exits with the candidate
    list on no/multi match."""
    matches = [e for e in _list_run_dirs(checkpoint_dir) if e.startswith(partial)]
    if not matches:
        console.print(f"[bold red]No run matching[/bold red] [path]{partial}[/path] in [path]{checkpoint_dir}[/path]")
        sys.exit(1)
    if len(matches) > 1:
        console.print(f"[bold red]Ambiguous run id[/bold red] [path]{partial}[/path]:")
        for m in matches:
            console.print(f"  [path]{m}[/path]")
        sys.exit(1)
    return matches[0]


def _read_parser_kind(run_dir: str) -> str | None:
    """Parser kind stamped into a JSON sidecar at train time (`save_checkpoint`
    forwards `parser_kind` into `best_model.json` / `last.json`). Preferred over
    `_infer_parser_kind`, whose config field-set heuristic is ambiguous for the
    generative-parser cluster. Returns None for older runs that predate the
    stamp, so callers fall back to inference."""
    for sidecar in ("best_model.json", "last.json"):
        path = os.path.join(run_dir, sidecar)
        try:
            with open(path, encoding="utf-8") as f:
                kind = json.load(f).get("parser_kind")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(kind, str) and kind:
            return kind
    return None


def _read_best_meta(run_dir: str) -> tuple[str, str]:
    """(best_val_str, step_str) from `best_model.json`. ("-", "-") if absent
    or unreadable. "(no best)" if no `best_model.pt` exists at all."""
    sidecar = os.path.join(run_dir, "best_model.json")
    if not os.path.exists(sidecar):
        return ("(no best)" if not os.path.exists(os.path.join(run_dir, "best_model.pt")) else "-"), "-"
    try:
        with open(sidecar, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "-", "-"
    val = meta.get("best_val")
    val_str = f"{val:.4f}" if isinstance(val, (int, float)) and val >= 0 else "-"
    step = meta.get("global_step")
    step_str = str(step) if isinstance(step, int) else "-"
    return val_str, step_str


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _format_size(n: int) -> str:
    unit = "B"
    val: float = float(n)
    for u in ("KB", "MB", "GB", "TB"):
        if val < 1024:
            break
        val /= 1024
        unit = u
    return f"{val:.1f} {unit}" if unit != "B" else f"{n} B"


def _latest_mtime(run_dir: str) -> float:
    """mtime of the freshest of best_model.pt / last.pt / config.json (so
    callers see real work, not dir-touch)."""
    for name in ("best_model.pt", "last.pt", "config.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return os.path.getmtime(p)
    return os.path.getmtime(run_dir)


# ---------------------------------------------------------------------------
# `runs list`


def list_runs(checkpoint_dir: str) -> None:
    if not os.path.isdir(checkpoint_dir):
        console.print(f"[bold red]No such directory:[/bold red] [path]{checkpoint_dir}[/path]")
        sys.exit(1)

    parsers = _all_parsers()
    # (mtime, row); sorted most- to least-recently-touched below.
    rows: list[tuple[float, tuple[str, ...]]] = []
    for entry in _list_run_dirs(checkpoint_dir):
        run_dir = os.path.join(checkpoint_dir, entry)
        try:
            with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        kind = _read_parser_kind(run_dir) or _infer_parser_kind(cfg, parsers)
        run_name = cfg.get("run_name") or "-"
        model_name = cfg.get("model_name", "?")
        train_dir = cfg.get("train_dir") or "?"
        best_val_str, step_str = _read_best_meta(run_dir)
        mtime = _latest_mtime(run_dir)
        modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        rows.append((mtime, (entry, run_name, kind, model_name, train_dir, best_val_str, step_str, modified)))

    if not rows:
        console.print(f"[dim]No runs found in[/dim] [path]{checkpoint_dir}[/path]")
        return

    rows.sort(key=lambda r: r[0], reverse=True)

    table = Table(title=f"Runs in {checkpoint_dir}", show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("run_id", style="bold")
    table.add_column("run_name", style="dim")
    table.add_column("parser")
    table.add_column("model_name")
    table.add_column("train_dir")
    table.add_column("best_val", justify="right", style="bold green")
    table.add_column("step", justify="right", style="dim")
    table.add_column("modified", style="dim")
    for _, row in rows:
        table.add_row(*row)
    console.print(table)


# ---------------------------------------------------------------------------
# `runs show`


def show_run(checkpoint_dir: str, partial: str) -> None:
    run_id = _resolve_run_id(checkpoint_dir, partial)
    run_dir = os.path.join(checkpoint_dir, run_id)
    parsers = _all_parsers()

    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    kind = _read_parser_kind(run_dir) or _infer_parser_kind(cfg, parsers)
    best_val_str, step_str = _read_best_meta(run_dir)
    size = _format_size(_dir_size(run_dir))
    modified = datetime.fromtimestamp(_latest_mtime(run_dir)).strftime("%Y-%m-%d %H:%M")

    header = Table(show_header=False, padding=(0, 2), box=None)
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("run_id", run_id)
    header.add_row("run_name", cfg.get("run_name") or "-")
    header.add_row("parser", kind)
    header.add_row("run_dir", run_dir)
    header.add_row("modified", modified)
    header.add_row("size", size)
    header.add_row("best_val", best_val_str)
    header.add_row("step", step_str)
    console.print(Panel(header, title=f"[bold magenta]Run[/bold magenta] {run_id}", border_style="magenta"))

    console.print(Panel(Pretty(cfg), title="[bold cyan]Config[/bold cyan]", border_style="cyan"))

    fm_path = os.path.join(run_dir, "final_metrics.json")
    if os.path.exists(fm_path):
        try:
            with open(fm_path, encoding="utf-8") as f:
                final = json.load(f)
            metrics = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
            metrics.add_column("split", style="bold")
            all_keys: list[str] = []
            for d in final.values():
                if isinstance(d, dict):
                    for k in d:
                        if k not in all_keys:
                            all_keys.append(k)
            for k in all_keys:
                metrics.add_column(k, justify="right")
            for split, d in final.items():
                if not isinstance(d, dict):
                    continue
                metrics.add_row(
                    split, *[f"{d[k]:.4f}" if k in d and isinstance(d[k], (int, float)) else "-" for k in all_keys]
                )
            console.print(Panel(metrics, title="[bold green]Final metrics[/bold green]", border_style="green"))
        except (OSError, json.JSONDecodeError):
            pass

    files = Table(show_header=False, padding=(0, 2), box=None)
    files.add_column(style="bold")
    files.add_column(justify="right", style="dim")
    for name in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, name)
        if os.path.isdir(full):
            count = sum(1 for _ in os.scandir(full))
            files.add_row(f"{name}/", f"{count} entries")
        else:
            files.add_row(name, _format_size(os.path.getsize(full)))
    console.print(Panel(files, title="[bold yellow]Files[/bold yellow]", border_style="yellow"))


# ---------------------------------------------------------------------------
# `runs diff`


def diff_runs(checkpoint_dir: str, partial_a: str, partial_b: str) -> None:
    a = _resolve_run_id(checkpoint_dir, partial_a)
    b = _resolve_run_id(checkpoint_dir, partial_b)
    if a == b:
        console.print(f"[dim]{a} and {b} resolve to the same run.[/dim]")
        return

    with open(os.path.join(checkpoint_dir, a, "config.json"), encoding="utf-8") as f:
        cfg_a = json.load(f)
    with open(os.path.join(checkpoint_dir, b, "config.json"), encoding="utf-8") as f:
        cfg_b = json.load(f)

    keys = sorted(set(cfg_a) | set(cfg_b))
    diffs = [k for k in keys if cfg_a.get(k) != cfg_b.get(k)]

    if not diffs:
        console.print(f"[green]No differences[/green] between [path]{a}[/path] and [path]{b}[/path].")
        return

    sentinel = object()
    table = Table(
        title=f"Diff: {a} vs {b}",
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("field", style="bold")
    table.add_column(a[:16])
    table.add_column(b[:16])
    for k in diffs:
        va = cfg_a.get(k, sentinel)
        vb = cfg_b.get(k, sentinel)
        sa = "[dim]<missing>[/dim]" if va is sentinel else _short_repr(va)
        sb = "[dim]<missing>[/dim]" if vb is sentinel else _short_repr(vb)
        table.add_row(k, sa, sb)
    console.print(table)


def _short_repr(v) -> str:
    if isinstance(v, (dict, list)) and len(str(v)) > 60:
        return f"{type(v).__name__}({len(v)} items)"
    return repr(v)


# ---------------------------------------------------------------------------
# `runs rename`


def rename_run(checkpoint_dir: str, partial: str, new_run_name: str) -> None:
    if not new_run_name or "/" in new_run_name:
        console.print(f"[bold red]Invalid run_name:[/bold red] {new_run_name!r}")
        sys.exit(1)
    run_id = _resolve_run_id(checkpoint_dir, partial)
    run_dir = os.path.join(checkpoint_dir, run_id)

    m = _HASH_SUFFIX_RE.search(run_id)
    if m is None:
        console.print(f"[bold red]Can't extract hash from run id[/bold red] {run_id!r}")
        sys.exit(1)
    cfg_hash = m.group(1)
    new_run_id = f"{new_run_name}-{cfg_hash}"
    new_run_dir = os.path.join(checkpoint_dir, new_run_id)
    if os.path.exists(new_run_dir):
        console.print(f"[bold red]Target already exists:[/bold red] [path]{new_run_dir}[/path]")
        sys.exit(1)

    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    old_run_name = cfg.get("run_name")
    cfg["run_name"] = new_run_name
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    os.rename(run_dir, new_run_dir)

    success(f"Renamed [path]{run_id}[/path] → [path]{new_run_id}[/path]")
    if old_run_name != new_run_name:
        dim(
            f"  Note: embedded run_name inside last.pt / best_model.pt is unchanged "
            f"({old_run_name!r}). It isn't used for run-dir resolution. If you use "
            f"`predict --config <jsonnet>`, update the jsonnet's run_name to {new_run_name!r}."
        )


# ---------------------------------------------------------------------------
# `runs delete` / `runs delete-all`


def _summarize_run(checkpoint_dir: str, run_id: str, parsers: dict) -> str:
    """One-line summary used in delete prompts: parser, best_val, size."""
    run_dir = os.path.join(checkpoint_dir, run_id)
    try:
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        cfg = {}
    kind = _read_parser_kind(run_dir) or _infer_parser_kind(cfg, parsers)
    run_name = cfg.get("run_name") or "-"
    best_val_str, _ = _read_best_meta(run_dir)
    size = _format_size(_dir_size(run_dir))
    return f"[path]{run_id}[/path]  name={run_name}  parser={kind}  best_val={best_val_str}  size={size}"


def delete_run(checkpoint_dir: str, partial: str, assume_yes: bool) -> None:
    run_id = _resolve_run_id(checkpoint_dir, partial)
    run_dir = os.path.join(checkpoint_dir, run_id)
    parsers = _all_parsers()

    console.print("[bold yellow]About to delete:[/bold yellow]")
    console.print(f"  {_summarize_run(checkpoint_dir, run_id, parsers)}")

    if not assume_yes:
        try:
            answer = console.input("[bold]Delete this run? (y/N):[/bold] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            console.print("[dim]Aborted.[/dim]")
            return

    shutil.rmtree(run_dir)
    success(f"Deleted [path]{run_dir}[/path]")


def delete_all_runs(checkpoint_dir: str) -> None:
    parsers = _all_parsers()
    run_ids = _list_run_dirs(checkpoint_dir)
    if not run_ids:
        console.print(f"[dim]No runs to delete in[/dim] [path]{checkpoint_dir}[/path]")
        return

    total_size = sum(_dir_size(os.path.join(checkpoint_dir, r)) for r in run_ids)
    console.print(
        f"[bold yellow]About to delete {len(run_ids)} run(s) from[/bold yellow] [path]{checkpoint_dir}[/path]:"
    )
    for r in run_ids:
        console.print(f"  {_summarize_run(checkpoint_dir, r, parsers)}")
    console.print(f"\n[bold]Total: {len(run_ids)} runs, {_format_size(total_size)}[/bold]")
    console.print("[bold red]This cannot be undone.[/bold red]")

    try:
        answer = console.input("Type [bold]delete all[/bold] to confirm: ").strip()
    except EOFError:
        answer = ""
    if answer != "delete all":
        console.print("[dim]Aborted (did not type 'delete all').[/dim]")
        return

    for r in run_ids:
        shutil.rmtree(os.path.join(checkpoint_dir, r))
    success(f"Deleted {len(run_ids)} run(s).")


# ---------------------------------------------------------------------------
# Archiving and disk reclaim

# Subdirs worth keeping when a run is archived: the prediction dumps (e2e,
# gold-EDU), any `archive_*` decode-pass snapshots (the beam-6 headline lives
# here once a greedy pass overwrites the top-level final_metrics.json), and
# side-script re-eval outputs. `tb/` and `*.pt` are never archived.
def _is_archivable_subdir(name: str) -> bool:
    n = name.lower()
    return "prediction" in n or n.startswith("archive") or n.endswith("_reval") or n.startswith("gold_edu")


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_run_for_archive(run_dir: str, dest: str) -> dict[str, str]:
    """Copy a run's durable subset (every root `*.json`, plus prediction /
    archive / reval subdirs, minus `*.pt` and `tb/`) into `dest`. Returns a
    {relpath: sha256} manifest of the copied files for verification."""
    os.makedirs(dest, exist_ok=True)
    manifest: dict[str, str] = {}
    for entry in sorted(os.listdir(run_dir)):
        src = os.path.join(run_dir, entry)
        if os.path.isfile(src) and entry.endswith(".json"):
            shutil.copy2(src, os.path.join(dest, entry))
            manifest[entry] = _sha256(src)
        elif os.path.isdir(src) and _is_archivable_subdir(entry):
            dst = os.path.join(dest, entry)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.pt"))
            for root, _dirs, files in os.walk(dst):
                for fn in files:
                    p = os.path.join(root, fn)
                    manifest[os.path.relpath(p, dest)] = _sha256(p)
    return manifest


def _verify_archived(dest: str) -> None:
    """Re-parse every archived JSON so a corrupt copy is caught before the
    source .pt is reclaimed. Raises on failure."""
    for root, _dirs, files in os.walk(dest):
        for fn in files:
            if fn.endswith(".json"):
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    json.load(f)


def _archive_index_row(archive_run_dir: str, run_id: str, kind: str) -> dict:
    def load(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    cfg = load(os.path.join(archive_run_dir, "config.json"))
    side = load(os.path.join(archive_run_dir, "best_model.json")) or load(os.path.join(archive_run_dir, "last.json"))

    # Headline = the highest-beam final_metrics among the top-level file and any
    # archive_*/ decode snapshot. The two-pass eval protocol (beam-6 headline,
    # then a greedy robustness pass) overwrites the top-level final_metrics.json
    # with greedy, stashing beam-6 in archive_beam6/, so the top-level file
    # alone would under-report the headline for those runs.
    candidates = []
    top = os.path.join(archive_run_dir, "final_metrics.json")
    if os.path.exists(top):
        candidates.append(load(top))
    for entry in os.listdir(archive_run_dir):
        snap = os.path.join(archive_run_dir, entry, "final_metrics.json")
        if entry.startswith("archive") and os.path.exists(snap):
            candidates.append(load(snap))
    fm = max(candidates, key=lambda c: (c.get("decode") or {}).get("num_beams", 0), default={})
    td = str(cfg.get("train_dir", "")).lower()
    corpus = "gum" if "gum" in td else "rstdt" if "rstdt" in td else "?"

    def m(split, key):
        try:
            return f"{fm[split][key]:.4f}"
        except Exception:
            return ""

    peft = cfg.get("peft") or {}
    curr = cfg.get("curriculum") or {}
    n_rs4 = 0
    for root, _dirs, files in os.walk(archive_run_dir):
        n_rs4 += sum(1 for x in files if x.endswith(".rs4"))
    return {
        "run_id": run_id, "parser_kind": kind, "corpus": corpus,
        "model_name": str(cfg.get("model_name", "")), "seed": str(cfg.get("seed", "")),
        "beams": str((fm.get("decode") or {}).get("num_beams", "")),
        "dev_e2e_full": m("dev", "e2e_full_f1"), "dev_gold_full": m("dev", "gold_edu_full_f1"),
        "test_e2e_full": m("test", "e2e_full_f1"), "test_gold_full": m("test", "gold_edu_full_f1"),
        "test_seg": m("test", "seg_f1"), "has_final": "Y" if fm else "-",
        "best_val": str(side.get("best_val", "")), "epoch": str(side.get("epoch", "")),
        "curriculum": str(curr.get("type", "simple") if curr else ""),
        "peft_r": str(peft.get("r", "")) if peft else "full",
        "train_dir": str(cfg.get("train_dir", "")), "n_rs4": str(n_rs4),
    }


_INDEX_COLS = ["run_id", "parser_kind", "corpus", "model_name", "seed", "beams",
               "dev_e2e_full", "dev_gold_full", "test_e2e_full", "test_gold_full",
               "test_seg", "has_final", "best_val", "epoch", "curriculum", "peft_r",
               "train_dir", "n_rs4"]


def _rebuild_archive_index(archive_dir: str) -> None:
    """Rewrite `<archive_dir>/INDEX.tsv` from scratch by walking the archive
    (one row per <parser_kind>/<run_id>). The per-run JSON is authoritative,
    the index a lossy denormalized view, so a full rebuild can never drift."""
    rows = []
    for kind in sorted(os.listdir(archive_dir)):
        kd = os.path.join(archive_dir, kind)
        if not os.path.isdir(kd):
            continue
        for run_id in sorted(os.listdir(kd)):
            rd = os.path.join(kd, run_id)
            if os.path.isdir(rd) and os.path.exists(os.path.join(rd, "config.json")):
                rows.append(_archive_index_row(rd, run_id, kind))
    path = os.path.join(archive_dir, "INDEX.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(_INDEX_COLS) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in _INDEX_COLS) + "\n")
    wrote(path)


def archive_runs(checkpoint_dir, archive_dir, run_ids, run_name, do_all, delete_pt, harvested_at):
    """Harvest finished runs' durable subset into a plain-dir archive at
    `<archive_dir>/<parser_kind>/<run_id>/`. Targeted by default: pass run ids,
    `--run-name` prefix, or `--all` (a bare invocation refuses to sweep).

    Only runs with a `final_metrics.json` are harvested (it is written last, so
    its presence is the completion sentinel: predictions are complete and the
    source is quiescent). Each run stages into `<run_id>.tmp` then atomically
    swaps in, is verified (every JSON re-parses), and gets a `_harvest.json`
    manifest. `--delete-pt` then removes the source `*.pt` (and `tb/`) once the
    copy verifies, reclaiming disk while the numbers/predictions live on."""
    if not (run_ids or run_name or do_all):
        console.print("[bold red]Refusing to archive without a target.[/bold red] Pass run ids, --run-name PREFIX, or --all.")
        sys.exit(2)

    all_dirs = _list_run_dirs(checkpoint_dir)
    if run_ids:
        selected = [_resolve_run_id(checkpoint_dir, r) for r in run_ids]
    elif run_name:
        selected = [d for d in all_dirs if d.startswith(run_name)]
    else:
        selected = all_dirs
    if not selected:
        console.print("[yellow]No matching runs.[/yellow]")
        return

    os.makedirs(archive_dir, exist_ok=True)
    parsers = _all_parsers()
    n_ok = n_skip = 0
    for run_id in selected:
        run_dir = os.path.join(checkpoint_dir, run_id)
        if not os.path.exists(os.path.join(run_dir, "final_metrics.json")):
            dim(f"  skip {run_id}: no final_metrics.json (not finished)")
            n_skip += 1
            continue
        try:
            with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            dim(f"  skip {run_id}: unreadable config.json")
            n_skip += 1
            continue
        kind = _read_parser_kind(run_dir) or _infer_parser_kind(cfg, parsers)

        dest_parent = os.path.join(archive_dir, kind)
        os.makedirs(dest_parent, exist_ok=True)
        final_dest = os.path.join(dest_parent, run_id)
        tmp_dest = final_dest + ".tmp"
        if os.path.exists(tmp_dest):
            shutil.rmtree(tmp_dest)

        manifest = _copy_run_for_archive(run_dir, tmp_dest)
        _verify_archived(tmp_dest)
        with open(os.path.join(tmp_dest, "_harvest.json"), "w", encoding="utf-8") as f:
            json.dump({"source_path": os.path.abspath(run_dir), "harvested_at": harvested_at,
                       "files": manifest}, f, indent=2)
        if os.path.exists(final_dest):
            shutil.rmtree(final_dest)
        os.replace(tmp_dest, final_dest)
        success(f"  archived {kind}/{run_id} ({len(manifest)} files)")
        n_ok += 1

        if delete_pt:
            reclaimed = 0
            for entry in os.listdir(run_dir):
                p = os.path.join(run_dir, entry)
                if os.path.isfile(p) and entry.endswith(".pt"):
                    reclaimed += os.path.getsize(p)
                    os.remove(p)
                elif os.path.isdir(p) and entry == "tb":
                    for r, _d, fs in os.walk(p):
                        reclaimed += sum(os.path.getsize(os.path.join(r, x)) for x in fs)
                    shutil.rmtree(p)
            if reclaimed:
                dim(f"    reclaimed {reclaimed / 1024**3:.1f}GB of .pt/tb from source")

    _rebuild_archive_index(archive_dir)
    success(f"Archived {n_ok} run(s) to {archive_dir}" + (f", skipped {n_skip}" if n_skip else "") + ".")


# ---------------------------------------------------------------------------
# CLI wiring


def main():
    parser = argparse.ArgumentParser(prog="dipo runs", description="Inspect and manage dipo training runs")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint-dir", default="checkpoints", help="Root checkpoint dir to walk")

    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("list", parents=[common], help="List runs under a checkpoint directory")

    p_show = sub.add_parser("show", parents=[common], help="Deep-inspect a single run")
    p_show.add_argument("run_id", help="Run id (or unique prefix)")

    p_diff = sub.add_parser("diff", parents=[common], help="Show config-field differences between two runs")
    p_diff.add_argument("run_id_a", help="First run id (or unique prefix)")
    p_diff.add_argument("run_id_b", help="Second run id (or unique prefix)")

    p_rename = sub.add_parser("rename", parents=[common], help="Set a run_name (renames the dir to <name>-<hash>)")
    p_rename.add_argument("run_id", help="Run id (or unique prefix)")
    p_rename.add_argument("new_run_name", help="New run_name")

    p_delete = sub.add_parser("delete", parents=[common], help="Delete a single run (prompts unless --yes)")
    p_delete.add_argument("run_id", help="Run id (or unique prefix)")
    p_delete.add_argument("--yes", action="store_true", help="Skip the y/N prompt")

    sub.add_parser(
        "delete-all", parents=[common], help="Delete every run in the checkpoint dir (requires typing 'delete all')"
    )

    p_arch = sub.add_parser(
        "archive", parents=[common],
        help="Harvest finished runs' config+metrics+predictions into a backed-up archive (targeted by default)",
    )
    p_arch.add_argument("run_ids", nargs="*", help="Run ids (or unique prefixes) to archive")
    p_arch.add_argument("--archive-dir", required=True, help="Destination archive root (plain dir, <parser_kind>/<run_id>/)")
    p_arch.add_argument("--run-name", help="Archive every run whose dir name starts with this prefix")
    p_arch.add_argument("--all", action="store_true", help="Archive every finished run (explicit opt-in to a full sweep)")
    p_arch.add_argument("--delete-pt", action="store_true", help="After a verified copy, delete the source .pt/tb to reclaim disk")

    args = parser.parse_args()

    if args.subcommand == "list":
        list_runs(args.checkpoint_dir)
    elif args.subcommand == "show":
        show_run(args.checkpoint_dir, args.run_id)
    elif args.subcommand == "diff":
        diff_runs(args.checkpoint_dir, args.run_id_a, args.run_id_b)
    elif args.subcommand == "rename":
        rename_run(args.checkpoint_dir, args.run_id, args.new_run_name)
    elif args.subcommand == "delete":
        delete_run(args.checkpoint_dir, args.run_id, args.yes)
    elif args.subcommand == "delete-all":
        delete_all_runs(args.checkpoint_dir)
    elif args.subcommand == "archive":
        archive_runs(
            args.checkpoint_dir, args.archive_dir, args.run_ids, args.run_name,
            args.all, args.delete_pt, datetime.now().isoformat(timespec="seconds"),
        )


if __name__ == "__main__":
    main()
