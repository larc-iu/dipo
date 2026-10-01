"""Config for the in-context-learning parser `icl`.

Composes three orthogonal axes, like `gen`'s backbone x serialization:

- `provider`: where the LM lives (`anthropic` frontier vs `openai_compatible`
  self-hosted). A single superset dataclass, branched by `type` in `provider.py`.
- `serialization`: how the tree is linearized (`sr` shift-reduce vs `sexp`).
- `pipeline_mode`: how many calls and what is given (`e2e` one call, `two_call`
  segment-then-parse, `gold_edu` parse-only over gold EDUs).

There is no training here, so the config carries no optimizer / checkpoint knobs.
`relation_types` is inferred from the data at load time (like the trained parsers)
and drives both the prompt's legal-label inventory and the decoder.
"""

from dataclasses import dataclass, field

from tonga import FromParams

from iudex.rst.parsers.common.config import parse_config_dict

_PROVIDERS = ("anthropic", "openai_compatible")
_SERIALIZATIONS = ("sr", "sexp")
_PIPELINE_MODES = ("e2e", "two_call", "gold_edu")


@dataclass
class ProviderConfig(FromParams):
    """LM endpoint knobs, a superset over both provider kinds (branched by
    `type` in `provider.build_provider`). Anthropic reads `thinking` / `effort`;
    openai_compatible reads `base_url`. `model` is a litellm model string
    (e.g. `anthropic/claude-opus-4-8`, or `openai/<name>` for a vLLM/llama.cpp
    endpoint)."""

    type: str = "anthropic"
    model: str = "anthropic/claude-opus-4-8"
    max_tokens: int = 120000

    # anthropic only
    thinking: bool = True
    effort: str = "high"  # low | medium | high | max

    # openai_compatible only (self-hosted vLLM / llama.cpp)
    base_url: str | None = None

    # Auth. `api_key_env` names the env var holding the key (preferred). `api_key`
    # is an inline literal for local/dummy endpoints only (e.g. llama.cpp's "foo");
    # never commit a real key here.
    api_key_env: str | None = None
    api_key: str | None = None

    # Per-request timeout (seconds) and retry budget for transient failures (5xx,
    # timeouts, rate limits). Real hosted/self-hosted endpoints hang or blip, so a
    # per-document eval must fail a stuck call cleanly (then retry) instead of
    # blocking. The default is generous enough for a slow thinking model.
    request_timeout: int = 600
    max_retries: int = 4

    # openai_compatible only: turn OFF a thinking model's reasoning via
    # extra_body chat_template_kwargs (the vLLM/reallms mechanism). Some hosted
    # thinking models over-reason on mechanical stages (segmentation) and are far
    # slower; this disables that. Ignored by the anthropic provider.
    disable_thinking: bool = False

    # Stream the completion. Turn this OFF for endpoints whose SSE deltas
    # mislabel their channels.
    #
    # Measured on the IU reallms gateway serving GLM-5.2: non-streaming returns
    # the answer in `content` and the reasoning in `reasoning_content`, correctly
    # separated -- but STREAMING puts the head of the reasoning in `content`,
    # switches to `reasoning_content` partway through, and leaves the actual
    # answer buried at the end of the reasoning channel. Reproduced against raw
    # SSE as well as litellm, so it is the gateway, not the client.
    #
    # Consequence if left on: every document parses a reasoning prefix as if it
    # were the answer and fails. Non-streaming costs the per-request progress
    # log and needs a request_timeout that covers the whole generation.
    stream: bool = True

    # Decoding temperature. 0 = greedy, which is the project-wide default for
    # every reported number and what the llama.cpp native path has always sent.
    #
    # This exists because the two paths had silently diverged: the native endpoint
    # hardcoded temperature 0, while the openai_compatible path sent NOTHING and
    # therefore inherited whatever the provider defaults to (typically 1.0). So the
    # constrained arm was decoding greedily and the unconstrained arm was sampling,
    # and the grammar was being credited for a difference that sampling partly
    # caused. Measured 2026-08-05 on GUM gold-EDU: two runs of the identical
    # unconstrained config produced 14/32 and 20/32 failures with only 9 documents
    # failing in both -- the failure set was substantially a draw, not a property
    # of the documents.
    #
    # Note that 0 buys greedy, not bit-reproducibility: hosted MoE serving batches
    # requests, and expert routing under different batch compositions can still
    # move a logit. It removes the dominant source of variance, not all of it.
    temperature: float = 0.0

    # openai_compatible only: extra JSON merged into the request body, for
    # vendor-specific fields litellm has no first-class argument for. Grammar
    # and thinking flags are built separately and take precedence on key
    # collision, so this cannot silently override them.
    #
    # The motivating case is OpenRouter's routing policy. OpenRouter is a broker:
    # one model id fans out to many upstream providers that differ in BOTH price
    # and capability. For deepseek-v4-pro (measured 2026-08-05) output price
    # ranged $0.87-$3.48/M across 18 providers -- and, worse, `max_completion_tokens`
    # ranged 16,384 (DeepInfra) to 1,048,576. A 69-EDU document already spends
    # ~28.7k output tokens on this protocol, so an unpinned route can truncate a
    # long document purely on which upstream happened to serve it, manufacturing
    # exactly the parse failures an unconstrained-decoding arm is meant to measure.
    # Pinning with {"provider": {"only": [...], "allow_fallbacks": false}} makes
    # the run reproducible and the cost predictable.
    extra_body: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in _PROVIDERS:
            raise ValueError(f"provider.type must be one of {_PROVIDERS} (got {self.type!r})")
        if self.effort not in ("low", "medium", "high", "max"):
            raise ValueError(f"provider.effort must be low|medium|high|max (got {self.effort!r})")
        if self.type == "openai_compatible" and not self.base_url:
            raise ValueError("provider.type='openai_compatible' requires base_url (the /v1 endpoint).")


@dataclass
class GrammarConfig(FromParams):
    """Grammar-constrained decoding (the `icl-c` arm). Only meaningful with a
    self-hosted `openai_compatible` provider (frontier Anthropic has no CFG
    support), the `sr` serialization, and a non-e2e pipeline (segmentation and
    the parse-stage SR sequence are each grammar-shaped; full e2e is not)."""

    enabled: bool = False
    # Grammar dialect / how it is passed to the endpoint:
    #   "gbnf"     -> llama.cpp, sent as extra_body={"grammar": <GBNF>} (implemented).
    #   "guidance" -> vLLM guided_grammar via llguidance (needs a Lark emitter, TODO).
    #   "xgrammar" -> vLLM guided_grammar via xgrammar (TODO).
    backend: str = "gbnf"

    # Lazily-triggered grammar (llama.cpp `grammar_lazy` + `grammar_triggers`).
    # An eager grammar constrains from token 1, so a thinking model cannot emit a
    # think block at all and icl-c is forced thinking-off -- which would make it
    # differ from the frontier arm in reasoning as well as in constraint. Lazy
    # mode lets the model think freely, then clamps the grammar on at
    # `think_end`, so the constraint is the only difference between the arms.
    lazy: bool = False
    think_end: str = "</think>"

    # BUDGET FORCING. Cap the think block at N tokens; llama.cpp then forces
    # `budget_message + think_end`, which fires the lazy trigger, so the answer is
    # grammar-constrained instead of the document being lost. Null = unbounded.
    #
    # This is the only lever that works on runaway reasoning: measured on RST-DT
    # test, generations that terminate naturally have a HARD CEILING at ~29.3k
    # tokens (n=78, median 12.2k, p95 22.6k) and nothing at all terminates between
    # there and a 245,760-token cap. The failure mode is bimodal -- finish by ~30k
    # or never -- which is why a 5x budget sweep (49k -> 98k -> 246k) left the
    # failure count flat at 20/22/18 while burning ~2.7h per lost document.
    #
    # Implemented CLIENT-SIDE, in two calls (see the provider). llama.cpp's own
    # `reasoning_budget_tokens` cannot be used here: when it FORCES the think-end
    # the grammar is left untriggered and the answer comes back unconstrained,
    # silently. Measured; see the provider for the A/B.
    think_budget: int | None = None
    # Injected verbatim ahead of the forced think_end. Doubles as the marker that
    # lets the eval count how many calls were truncated, so forced documents are
    # reported rather than silently blended in with naturally-completed ones.
    budget_message: str = "\n\nI have used my reasoning budget. Final answer now.\n"

    def __post_init__(self):
        if self.backend not in ("gbnf", "guidance", "xgrammar"):
            raise ValueError(f"grammar.backend must be gbnf|guidance|xgrammar (got {self.backend!r})")
        if self.lazy and self.backend != "gbnf":
            raise ValueError(f"grammar.lazy=True is implemented only for backend='gbnf' (got {self.backend!r})")
        if self.lazy and not self.think_end:
            raise ValueError("grammar.lazy=True requires a non-empty grammar.think_end trigger.")
        if self.think_budget is not None:
            if not self.lazy:
                raise ValueError(
                    "grammar.think_budget requires grammar.lazy=True: with an eager grammar there is "
                    "no think block to budget (the constraint applies from token 1)."
                )
            if self.think_budget <= 0:
                raise ValueError(
                    f"grammar.think_budget must be positive or null (got {self.think_budget}); "
                    "0 would force the end tag immediately, i.e. thinking off."
                )
            if not self.budget_message:
                raise ValueError(
                    "grammar.think_budget requires a non-empty budget_message: it is what gets written "
                    "before the forced think_end, and its presence is how forced calls are counted."
                )


@dataclass
class IclConfig(FromParams):
    train_dir: str
    dev_dir: str
    test_dir: str | None = None

    provider: ProviderConfig = field(default_factory=ProviderConfig)
    serialization: str = "sr"  # "sr" | "sexp"
    pipeline_mode: str = "two_call"  # "e2e" | "two_call" | "gold_edu"
    grammar: GrammarConfig = field(default_factory=GrammarConfig)

    # Re-emit each EDU's text in the parse stage instead of a bare `<shift>` (sr)
    # or a bare index (sexp), so the decode stream carries what is being attached
    # rather than only how many things have been attached.
    #
    # The motivation is that the parse stage otherwise emits hundreds of identical
    # `<shift>` tokens, giving the model no signal about WHICH EDU it is consuming;
    # position has to be tracked in hidden state, and the exact-count grammar hides
    # any drift by forcing a well-formed tree anyway. Under `sr` + grammar the text
    # is forced per state (the state determines which EDU is next), so this is pure
    # grounding: no freedom to paraphrase or skip is added. It also makes
    # `render_parse` identical to `render_e2e`, and so to gen's words-mode, which
    # is why it is a convergence rather than a third scheme.
    #
    # Costs roughly double the parse-stage output length.
    echo_edu_text: bool = False

    # Optional system-role message (a light RST priors primer), sent on every
    # call ahead of the cached prefix. Null = no system prompt. Corpus-general by
    # design (the relation inventory + task mechanics stay in the user prompt).
    system_prompt: str | None = None

    # In-context examples: k worked demonstrations sampled from `train_dir`.
    # `example_max_edus` restricts sampling to docs no larger than N EDUs so the
    # cached prefix stays manageable (null = any size).
    k: int = 5
    seed: int = 42
    example_max_edus: int | None = None

    # Inferred from the data at load time (union over train/dev). Drives the
    # prompt's legal-label inventory and the decoder's reduce-token map.
    relation_types: list[tuple[str, str]] | None = None
    relation_map: dict[str, str] | None = None

    def __post_init__(self):
        if self.serialization not in _SERIALIZATIONS:
            raise ValueError(f"serialization must be one of {_SERIALIZATIONS} (got {self.serialization!r})")
        if self.pipeline_mode not in _PIPELINE_MODES:
            raise ValueError(f"pipeline_mode must be one of {_PIPELINE_MODES} (got {self.pipeline_mode!r})")
        if self.k < 0:
            raise ValueError(f"k must be >= 0 (got {self.k})")
        if self.grammar.enabled:
            if self.provider.type != "openai_compatible":
                raise ValueError(
                    "grammar.enabled=True requires provider.type='openai_compatible' "
                    "(frontier Anthropic has no grammar-constrained decoding)."
                )
            if self.serialization != "sr":
                raise ValueError(
                    "grammar.enabled=True requires serialization='sr'. The constrained arm (icl-c) is "
                    "SR-only by design (postfix shift-reduce is compactly CFG-constrainable, sexp is not)."
                )
            if self.pipeline_mode == "e2e":
                raise ValueError(
                    "grammar.enabled=True requires pipeline_mode in {two_call, gold_edu} "
                    "(the e2e copy+structure constraint is not a practical CFG)."
                )
            if self.grammar.lazy and self.provider.disable_thinking:
                # This combination fails SILENTLY and catastrophically: with thinking
                # off the model never emits the trigger, so the grammar never clamps
                # on and every response is UNCONSTRAINED while the run still calls
                # itself icl-c. Refuse it rather than produce "constrained" numbers
                # that are nothing of the kind.
                raise ValueError(
                    "grammar.lazy=True requires provider.disable_thinking=False. The lazy grammar is "
                    f"triggered by {self.grammar.think_end!r}, which a non-thinking model never emits, "
                    "so the grammar would never engage and output would be silently unconstrained."
                )

    @classmethod
    def from_dict(cls, d: dict) -> "IclConfig":
        return parse_config_dict(cls, d)
