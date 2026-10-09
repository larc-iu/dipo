"""Re-evaluate a trained `gen` run at a chosen decode width, without retraining.

Loads a finished run's frozen `config.json` and `best_model.pt` and re-runs the
*exact* final evaluation `train()` performs -- `evaluate_on_dev` on the dev and
test sets, e2e and gold-EDU -- at one or more decoding methods. Runs now default
to a greedy final eval, so `final_metrics.json` already holds the paper number;
this command exists to add the optional beam ablation (or to re-decode an older
beam-trained run greedily).

`--inference-method` takes a comma-separated list, each `greedy` or `beam-N`
(N>=2); parsing is conservative and exits on any malformed method. The decode
width is HASH_EXCLUDEd from the run identity, so all decodes share one run dir:
`greedy` writes the canonical `final_metrics.json`, and `beam-N` writes
`final_metrics.beam{N}.json` (predictions under `*_predictions/final_beam{N}/`),
leaving the greedy file untouched. Running several methods at once evaluates each.

    dipo gen eval <run_dir> --inference-method greedy,beam-6

`run_dir` is the trained run directory (must contain `config.json` and
`best_model.pt`). Relative data paths in the frozen config resolve against the
current working directory, so run this from the checkout whose `data/` holds the
intended carve (NOT larc's non-carve `data/rstdt`).
"""

import argparse
import json
import os

import torch

from dipo.common.log import console, warn, wrote
from dipo.common.training import load_model_state
from dipo.rst.data.metrics import metrics_table
from dipo.rst.data.reader import read_rst_dir
from dipo.rst.parsers.common.generative_eval import evaluate_on_dev
from dipo.rst.parsers.gen.configuration_gen import GenConfig
from dipo.rst.parsers.gen.modeling_gen import GenParser


def evaluate(run_dir: str, num_beams: int) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    best_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.json in {run_dir}")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No best_model.pt in {run_dir}")

    # The frozen config carries the inferred relation_types the model trained with,
    # so the label vocab matches the checkpoint exactly (no re-inference).
    cfg = GenConfig.from_dict(json.load(open(cfg_path, encoding="utf-8")))
    if cfg.relation_types is None:
        raise ValueError(
            f"{cfg_path} has relation_types=None; expected the frozen (inferred) list. "
            "Re-eval needs the exact training vocab."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenParser(cfg).to(device)  # __init__ builds the serialization decode vocab
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    load_model_state(model, checkpoint)
    model.eval()

    dev_pairs = read_rst_dir(cfg.dev_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
    test_pairs = (
        read_rst_dir(cfg.test_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
        if cfg.test_dir is not None
        else None
    )

    # Each method keeps its own predictions/metrics so one method never clobbers
    # another when several run in one invocation.
    tag = "final" if num_beams == 1 else f"final_beam{num_beams}"
    final: dict[str, dict] = {}
    dev_m = evaluate_on_dev(
        model, dev_pairs, num_beams=num_beams, batch_size=cfg.dev_batch_size,
        output_dir=os.path.join(run_dir, "dev_predictions", tag), eval_gold_edu=True,
    )
    console.print(metrics_table(dev_m, title=f"Dev (num_beams={num_beams})"))
    final["dev"] = dev_m
    if test_pairs is not None:
        test_m = evaluate_on_dev(
            model, test_pairs, num_beams=num_beams, batch_size=cfg.dev_batch_size,
            output_dir=os.path.join(run_dir, "test_predictions", tag), eval_gold_edu=True,
        )
        console.print(metrics_table(test_m, title=f"Test (num_beams={num_beams})"))
        final["test"] = test_m
    final["decode"] = {"num_beams": num_beams}

    # Greedy is canonical -> final_metrics.json (matches the training-time final eval
    # and the main-table decode). A beam pass is an optional add-on and writes to a
    # beam-suffixed file so it never clobbers the greedy numbers.
    fname = "final_metrics.json" if num_beams == 1 else f"final_metrics.beam{num_beams}.json"
    metrics_path = os.path.join(run_dir, fname)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    wrote(metrics_path)
    return final


def parse_inference_methods(spec: str) -> list[int]:
    """Parse a ``--inference-method`` spec into an ordered list of beam widths.

    The spec is a comma-separated list of methods, each either ``greedy`` (beam
    width 1, the canonical main-table decode) or ``beam-N`` (an integer width
    ``N >= 2``). Whitespace around a method is ignored. Parsing is conservative:
    any empty, unknown, malformed, or duplicate method raises ``ValueError`` with
    a specific message, so the caller exits rather than silently guessing.

    Examples: ``greedy`` -> [1]; ``greedy,beam-6`` -> [1, 6]; ``beam-4, beam-8``
    -> [4, 8].
    """
    if spec is None or not spec.strip():
        raise ValueError("empty --inference-method (expected e.g. 'greedy' or 'greedy,beam-6')")
    widths: list[int] = []
    seen: set[int] = set()
    for raw in spec.split(","):
        tok = raw.strip()
        if tok == "greedy":
            width = 1
        elif tok.startswith("beam-"):
            n = tok[len("beam-"):]
            if not (n.isascii() and n.isdigit()):
                raise ValueError(f"malformed method {tok!r}: expected 'beam-<integer>' (e.g. 'beam-6')")
            width = int(n)
            if width < 2:
                raise ValueError(f"invalid method {tok!r}: beam width must be >= 2 (use 'greedy' for width 1)")
        else:
            raise ValueError(f"unknown method {tok!r}: expected 'greedy' or 'beam-<N>'")
        if width in seen:
            raise ValueError(f"duplicate method {tok!r} in --inference-method")
        seen.add(width)
        widths.append(width)
    return widths


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate a trained `gen` run at one or more decoding methods")
    parser.add_argument("run_dir", help="Trained run directory (contains config.json + best_model.pt)")
    parser.add_argument(
        "--inference-method",
        default=None,
        help="Comma-separated decoding methods, each 'greedy' or 'beam-N' (N>=2), e.g. "
        "'greedy', 'beam-6', or 'greedy,beam-6'. Default: greedy.",
    )
    # Deprecated single-width alias, hidden but kept so existing scripts keep working.
    parser.add_argument("--num-beams", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.path.isdir(args.run_dir):
        parser.error(f"not a directory: {args.run_dir}")

    if args.num_beams is not None:
        if args.inference_method is not None:
            parser.error("pass either --inference-method or --num-beams, not both")
        widths = [args.num_beams]
    else:
        try:
            widths = parse_inference_methods(args.inference_method or "greedy")
        except ValueError as e:
            parser.error(str(e))

    for width in widths:
        if width == 1:
            warn(
                f"Greedy eval will overwrite final_metrics.json in {args.run_dir} "
                "(the canonical greedy file). Copy it aside first if you need the existing one."
            )
        else:
            warn(
                f"Beam-{width} eval writes final_metrics.beam{width}.json in {args.run_dir}; "
                "the greedy final_metrics.json is left untouched."
            )
        evaluate(args.run_dir, width)


if __name__ == "__main__":
    main()
