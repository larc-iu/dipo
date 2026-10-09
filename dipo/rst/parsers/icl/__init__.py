"""In-context-learning RST "parser": prompt a frozen LM (no fine-tuning) to
emit a linearized tree, then reconstruct it.

Unlike the trained parsers this one has no checkpoint and no training loop. It
is an API client, so it owns bespoke `predict` / `eval` commands (not the shared
torch `run_predict` / `push`). Two arms share ONE pipeline, selected by config:

- `provider.type="anthropic"`   -> frontier ICL (the `icl` paper arm).
- `provider.type="openai_compatible"` -> a self-hosted OpenAI-compatible endpoint
  (vLLM / llama.cpp), the substrate for the grammar-constrained `icl-c` arm.

See `configuration_icl.py` for the config surface and `modeling_icl.py` for the
segment-then-parse pipeline.
"""
