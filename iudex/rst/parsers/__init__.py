"""Registry of RST parsers. Single source of truth for the dispatcher,
the shared `push` / `predict` CLIs, and `runs list`. To add a parser, add
one `ParserSpec` entry. Module paths follow `<package>/{configuration,
modeling,train,predict}_<name>.py` and class names follow `<Name>Config`
/ `<Name>Parser`.
"""

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ParserSpec:
    name: str
    package: str
    config_cls: str  # class name inside <package>/configuration_<name>.py
    parser_cls: str  # class name inside <package>/modeling_<name>.py
    # Exposes `--text` / `--text-file` in the predict CLI. Set on parsers
    # that implement `predict_from_text` (i.e. have a segmentation head).
    # Runtime still checks `model.segmenter is not None` since the segmenter
    # may be disabled per-config even on a `supports_text=True` parser.
    supports_text: bool
    # A config field used by `runs list` to tag a config.json with its parser
    # kind without importing parser modules. This is now the FALLBACK path:
    # training stamps `parser_kind` into the JSON sidecar, which `runs` reads
    # directly. The string should be distinct per parser, but the generative
    # cluster (seq2seq_* / decoder_only_*) shares fields, so inference relies on
    # a field-set heuristic when the sidecar stamp is absent (older runs).
    signature_field: str
    # False for API-client parsers (e.g. `icl`) that have no torch checkpoint.
    # The shared `push` / `predict` / `runs` machinery assumes a checkpoint, so it
    # does not apply to them (they own bespoke command modules instead).
    is_torch_model: bool = True

    def load_config_cls(self) -> type:
        mod = importlib.import_module(f"{self.package}.configuration_{self.name}")
        return getattr(mod, self.config_cls)

    def load_parser_cls(self) -> type:
        mod = importlib.import_module(f"{self.package}.modeling_{self.name}")
        return getattr(mod, self.parser_cls)


PARSERS: dict[str, ParserSpec] = {
    "topdown_biaffine": ParserSpec(
        name="topdown_biaffine",
        package="iudex.rst.parsers.topdown_biaffine",
        config_cls="TopdownBiaffineConfig",
        parser_cls="TopdownBiaffineParser",
        supports_text=False,
        signature_field="ffn_hidden_size",
    ),
    "sr_biaffine": ParserSpec(
        name="sr_biaffine",
        package="iudex.rst.parsers.sr_biaffine",
        config_cls="SRBiaffineConfig",
        parser_cls="SRBiaffineParser",
        supports_text=False,
        signature_field="action_ffn_hidden_size",
    ),
    "dmrst": ParserSpec(
        name="dmrst",
        package="iudex.rst.parsers.dmrst",
        config_cls="DMRSTConfig",
        parser_cls="DMRSTParser",
        supports_text=True,
        signature_field="attention_type",
    ),
    "gen": ParserSpec(
        name="gen",
        package="iudex.rst.parsers.gen",
        config_cls="GenConfig",
        parser_cls="GenParser",
        supports_text=True,
        # `backbone` (a discriminator) is unique to GenConfig; checkpoints also carry
        # an explicit parser_kind="gen" stamp, which `runs list` prefers.
        signature_field="backbone",
    ),
    # In-context-learning parser: an API client, not a torch model. It has no
    # training and no checkpoint, so it owns bespoke predict_icl / eval_icl
    # commands (the shared run_predict / push / runs-list machinery does not
    # apply and is never invoked for it). Registered here only so the dispatcher
    # routes `iudex icl <cmd>`. config_cls/parser_cls resolve for completeness;
    # supports_text is moot (the bespoke CLI does not consult it).
    "icl": ParserSpec(
        name="icl",
        package="iudex.rst.parsers.icl",
        config_cls="IclConfig",
        parser_cls="IclParser",
        supports_text=True,
        signature_field="pipeline_mode",
        is_torch_model=False,
    ),
}
