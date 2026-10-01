"""LM endpoint abstraction for the ICL parser.

`Provider.complete(cached_prefix, suffix)` is the one call the pipeline makes.
The prompt is split so the (byte-identical across documents) `cached_prefix`
carries task + relation inventory + k worked examples and the `suffix` carries
the one document. Two implementations:

- `AnthropicProvider`: frontier via litellm. Puts an Anthropic 1h prompt-cache
  breakpoint on the prefix (~1 write, N-1 reads over a run) and handles the
  Opus-4.8 request surface (adaptive thinking + `output_config.effort`, no
  temperature, streamed).
- `OpenAICompatibleProvider`: any OpenAI-compatible `/v1` endpoint (vLLM,
  llama.cpp). Server-side prefix KV caching handles prompt reuse. This is the
  substrate for the grammar-constrained `icl-c` arm: the `grammar` argument (extra
  request-body params, e.g. a GBNF grammar) is passed straight through to the
  endpoint (see serialize.sr_parse_grammar / segmentation_grammar).
"""

from __future__ import annotations

import os
import threading
import time

from iudex.common.log import dim, warn
from iudex.rst.parsers.icl.configuration_icl import ProviderConfig


class Provider:
    def complete(self, cached_prefix: str, suffix: str, *, grammar: dict | None = None, system: str | None = None) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    # Per-document generation lengths, so a run can report how long reasoning
    # actually ran rather than only whether it finished. Thread-local: the eval
    # predicts documents concurrently and one worker owns one document.
    def begin_doc(self) -> None:
        """Start recording a new document's calls."""

    def call_tokens(self) -> list[int]:
        """Generated-token counts for this document's calls, in order."""
        return []


def build_provider(cfg: ProviderConfig) -> Provider:
    if cfg.type == "anthropic":
        return AnthropicProvider(cfg)
    if cfg.type == "openai_compatible":
        return OpenAICompatibleProvider(cfg)
    raise ValueError(f"unknown provider.type {cfg.type!r}")


def _resolve_api_key(cfg: ProviderConfig) -> str | None:
    if cfg.api_key_env:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            warn(f"provider.api_key_env={cfg.api_key_env!r} is not set in the environment")
        return key
    return cfg.api_key


# Retry policy. `APIError` is included deliberately and is the reason the
# _PERMANENT list below exists: it is litellm's BASE class for everything, so it
# is also what gets raised bare for failures with no more specific type -- notably
# "Unable to get json response", a truncated or malformed HTTP body. Measured
# 2026-08-05: two GUM documents died that way against OpenRouter, and because
# `APIError` was absent here they were not retried even once. eval_icl then scored
# them as failed parses worth zero in the denominator, i.e. a transport blip was
# recorded as the model failing to produce a valid tree -- corrupting the one
# quantity the unconstrained arm is compared on.
_TRANSIENT = (
    "InternalServerError",
    "ServiceUnavailableError",
    "Timeout",
    "APIConnectionError",
    "RateLimitError",
    "APIError",
)

# Errors retrying cannot fix: the request is malformed, the credentials are wrong,
# or the input does not fit. Retrying a bad key six times with exponential backoff
# wastes minutes and buries the cause under a retry log.
#
# This list is DEFENSIVE, not currently load-bearing: as of litellm 1.x these
# classes descend from *openai*'s APIError, not from `litellm.APIError`, so the
# entry above does not actually catch them (verified by construction). The guard
# exists because that is an implementation detail of a third-party exception
# hierarchy, and if it ever converges the difference between "retry once" and
# "retry a permanent 400 six times" should not depend on noticing the change.
_PERMANENT = (
    "BadRequestError",
    "ContextWindowExceededError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
)


def _stream_completion(*, max_retries: int = 0, **kwargs) -> str:
    """Streamed litellm completion with hand-rolled retry on transient failures
    (5xx, timeouts, rate limits), so it needs no `tenacity` (litellm's own
    `num_retries` imports it and it is not always installed). Backs off
    exponentially; non-transient errors (e.g. 400) propagate immediately."""
    import litellm

    transient = tuple(getattr(litellm, n) for n in _TRANSIENT if hasattr(litellm, n))
    permanent = tuple(getattr(litellm, n) for n in _PERMANENT if hasattr(litellm, n))
    once = _stream_once if kwargs.get("stream") else _complete_once
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return once(litellm, kwargs)
        except transient as e:
            # `transient` catches APIError, which is also the base of the 4xx
            # family, so the permanent ones have to be filtered back out here
            # rather than simply omitted above.
            if isinstance(e, permanent) or attempt >= max_retries:
                raise
            warn(f"transient provider error ({type(e).__name__}), retry {attempt + 1}/{max_retries} in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")  # loop either returns or raises


def _complete_once(litellm, kwargs: dict) -> str:
    """One NON-streamed completion attempt, for endpoints whose SSE deltas
    mislabel `content` vs `reasoning_content` (see ProviderConfig.stream).
    Returns the answer channel only; reasoning is reported but discarded, which
    is what the callers want -- a thinking model's answer, without the thinking."""
    resp = litellm.completion(**kwargs)
    msg = resp.choices[0].message
    text = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    usage = getattr(resp, "usage", None)
    dim(
        f"finish_reason={resp.choices[0].finish_reason} "
        f"output_tokens={getattr(usage, 'completion_tokens', None)} answer_chars={len(text)} "
        f"thinking_chars={len(reasoning)}"
    )
    if not text:
        warn(
            f"empty answer (finish_reason={resp.choices[0].finish_reason}, "
            f"{len(reasoning)} chars of reasoning). If 'length', thinking consumed the whole "
            "budget before any answer was produced. Raise max_tokens."
        )
    return text


def _stream_once(litellm, kwargs: dict) -> str:
    """One streamed completion attempt: concatenate answer deltas and log a
    one-line diagnostic (finish_reason, token/char counts, cache reads) so an
    empty answer is debuggable."""
    parts: list[str] = []
    reasoning_chars = 0
    finish_reason = None
    final = None
    for chunk in litellm.completion(**kwargs):
        final = chunk
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        ch = choices[0]
        if getattr(ch, "finish_reason", None):
            finish_reason = ch.finish_reason
        delta = getattr(ch, "delta", None)
        if delta is None:
            continue
        if getattr(delta, "content", None):
            parts.append(delta.content)
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            reasoning_chars += len(rc)
    text = "".join(parts)

    out_toks = read = None
    try:
        usage = getattr(final, "usage", None)
        out_toks = getattr(usage, "completion_tokens", None)
        read = getattr(usage, "cache_read_input_tokens", None)
    except Exception:
        pass
    dim(
        f"finish_reason={finish_reason} output_tokens={out_toks} answer_chars={len(text)} "
        f"thinking_chars={reasoning_chars} cache_read={read}"
    )
    if not text:
        warn(
            f"empty answer (finish_reason={finish_reason}). If 'length'/'max_tokens', thinking "
            f"consumed the whole budget before any output. Raise max_tokens or lower effort."
        )
    return text


# ---------------------------------------------------------------------------
# Anthropic (frontier)
# ---------------------------------------------------------------------------

_litellm_effort_patched = False


def _patch_litellm_effort_guard() -> None:
    """litellm 1.83.x gates `output_config` effort='max' to Opus 4.6 only (its
    capability map predates 4.7/4.8). The API accepts 'max' on every Opus 4.6+
    model, so widen the guard for this process. No-op if the internal symbol
    moved (by then litellm likely knows 4.8)."""
    global _litellm_effort_patched
    if _litellm_effort_patched:
        return
    _litellm_effort_patched = True
    try:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        orig = AnthropicConfig._is_opus_4_6_model

        @staticmethod
        def _widened(model: str) -> bool:
            m = model.lower()
            return orig(model) or any(v in m for v in ("opus-4-7", "opus-4.7", "opus-4-8", "opus-4.8"))

        AnthropicConfig._is_opus_4_6_model = _widened
    except Exception as e:
        warn(f"could not patch litellm effort guard ({e!r}); effort='max' may be rejected")


class AnthropicProvider(Provider):
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    def describe(self) -> str:
        t = f"thinking={self.cfg.effort}" if self.cfg.thinking else "no-thinking"
        return f"anthropic {self.cfg.model} ({t})"

    def complete(self, cached_prefix: str, suffix: str, *, grammar: dict | None = None, system: str | None = None) -> str:
        if grammar is not None:
            raise ValueError("Anthropic has no grammar-constrained decoding (grammar must be None).")
        cfg = self.cfg
        if cfg.thinking and cfg.effort == "max":
            _patch_litellm_effort_guard()

        content = [
            {"type": "text", "text": cached_prefix, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": suffix},
        ]
        kwargs = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": cfg.max_tokens,
            "timeout": cfg.request_timeout,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if system:
            kwargs["system"] = system
        key = _resolve_api_key(cfg)
        if key:
            kwargs["api_key"] = key
        if cfg.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            # Forwarded verbatim into the request body by this litellm version
            # (extra_body is NOT merged; it lands as a literal rejected key).
            kwargs["output_config"] = {"effort": cfg.effort}
        return _stream_completion(max_retries=cfg.max_retries, **kwargs)


# ---------------------------------------------------------------------------
# OpenAI-compatible (self-hosted vLLM / llama.cpp) -- the icl-c substrate
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider(Provider):
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        self.api_key = _resolve_api_key(cfg) or "EMPTY"
        self._tls = threading.local()

    def begin_doc(self) -> None:
        self._tls.tokens = []

    def call_tokens(self) -> list[int]:
        return list(getattr(self._tls, "tokens", []))

    def describe(self) -> str:
        return f"openai_compatible {self.cfg.model} @ {self.cfg.base_url}"

    def complete(self, cached_prefix: str, suffix: str, *, grammar: dict | None = None, system: str | None = None) -> str:
        cfg = self.cfg
        # Optional system turn, then one user turn (prefix + suffix): the server
        # applies its own chat template and caches the shared prefix KV across docs.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": cached_prefix + "\n" + suffix})
        kwargs = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "api_base": cfg.base_url,
            "api_key": self.api_key,
            "timeout": cfg.request_timeout,
            # Sent EXPLICITLY. Omitting it inherits the provider's default (~1.0),
            # which silently made this path sample while the native path below
            # decoded greedily -- see ProviderConfig.temperature.
            "temperature": cfg.temperature,
        }
        if cfg.stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        # Config passthrough first, then the request-derived fields on top: a
        # stray key in cfg.extra_body can never displace the grammar or the
        # thinking flag.
        extra_body: dict = dict(cfg.extra_body)
        if grammar:
            extra_body.update(grammar)
        if cfg.disable_thinking:
            # vLLM chat-template flag (the reallms mechanism, verified) to turn a
            # thinking model's reasoning off. Grammar params (if any) merge alongside.
            extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        if extra_body.get("grammar_lazy"):
            # MUST bypass /v1/chat/completions: that endpoint overwrites
            # grammar_lazy / grammar_triggers / preserved_tokens unconditionally
            # from the chat template's own params (server-common.cpp, the OAI ->
            # llama params conversion), so a lazy grammar sent there is silently
            # downgraded to an EAGER one -- verified live, the model was clamped
            # from token 1 and never got to think. `grammar` itself survives,
            # which is exactly what makes the failure quiet. The native endpoints
            # parse all four fields properly.
            return self._native_complete(messages, extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return _stream_completion(max_retries=cfg.max_retries, **kwargs)

    # ----- llama.cpp native path (lazy grammars only) -----

    def _root_url(self) -> str:
        """`/apply-template` and `/completion` live at the server root, while
        `base_url` points at the OpenAI-compatible `/v1` subtree."""
        return (self.cfg.base_url or "").rstrip("/").removesuffix("/v1")

    def _post(self, path: str, payload: dict, key: str) -> str:
        """POST to a llama.cpp native endpoint with the shared retry policy."""
        import requests

        cfg = self.cfg
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                r = requests.post(
                    f"{self._root_url()}{path}", json=payload, headers=headers, timeout=cfg.request_timeout
                )
                r.raise_for_status()
                body = r.json()
                if "tokens_predicted" in body:
                    getattr(self._tls, "tokens", []).append(body["tokens_predicted"])
                return body.get(key, "")
            except Exception as e:  # noqa: BLE001 - retried below, re-raised on exhaustion
                last_exc = e
                if attempt == cfg.max_retries:
                    break
                delay = 2 ** (attempt + 1)
                warn(f"transient native-endpoint error ({type(e).__name__}), retry {attempt + 1}/{cfg.max_retries} in {delay}s")
                time.sleep(delay)
        raise RuntimeError(f"llama.cpp {path} failed after {cfg.max_retries} retries: {last_exc}")

    def _native_complete(self, messages: list[dict], extra_body: dict) -> str:
        cfg = self.cfg
        forcing = extra_body.get("_budget_forcing")
        body = {k: v for k, v in extra_body.items() if k not in ("chat_template_kwargs", "_budget_forcing")}
        template_body: dict = {"messages": messages}
        if "chat_template_kwargs" in extra_body:
            template_body["chat_template_kwargs"] = extra_body["chat_template_kwargs"]

        # Let the server apply its own chat template, so the prompt is
        # byte-identical to what the /v1 path would have produced and the
        # server-side prefix KV cache still hits across documents.
        prompt = self._post("/apply-template", template_body, "prompt")
        if not forcing:
            return self._post(
                "/completion",
                {"prompt": prompt, "n_predict": cfg.max_tokens, "temperature": cfg.temperature, "cache_prompt": True, **body},
                "content",
            )

        # ---- budget forcing, client side ----
        #
        # llama.cpp HAS a native reasoning budget (`reasoning_budget_tokens` &c.)
        # and it is NOT safe with our lazy grammar. Measured on one 4-EDU document
        # at temperature 0, same request either way:
        #   budget high enough to finish thinking naturally -> CONSTRAINED (3 reduces)
        #   budget low enough to force the tag              -> UNCONSTRAINED (4 reduces,
        #                                                      where exactly 3 are legal)
        # So the break is specific to the FORCED end-of-thinking, which is the only
        # case we would use it for. Mechanism not pinned down: `grammar_should_apply`
        # (common/sampling.cpp) does suppress the grammar sampler while the budget
        # sampler is COUNTING or FORCING, but that alone would break the natural end
        # too, and it does not -- so the likelier culprit is that the budget sampler
        # tokenizes `message + end_tag` itself and the resulting tag token does not
        # match the promoted single-token grammar trigger.
        #
        # Not chased further, because the failure is SILENT: a plausible-looking
        # answer that nothing constrained. That is not a risk worth carrying in the
        # one arm whose entire claim is validity.
        #
        # So do it here instead, out of two calls that each use only a path we
        # already trust. Call 1 is exactly today's lazy-grammar request, capped at
        # the budget. If it produced `</think>` the model finished thinking on its
        # own and the answer is already grammar-constrained -- nothing else to do.
        # Otherwise close the block ourselves and re-issue with an EAGER grammar
        # rooted at the answer, which is correct precisely because `</think>` is now
        # in the prompt rather than something the model still has to emit.
        # n_predict caps call 1's WHOLE generation, not just its think block --
        # there is no way to tell reasoning from answer without streaming, and
        # bounding the total is the entire point (a runaway must not be able to
        # burn max_tokens). A natural finisher therefore answers out of whatever
        # the budget has left, which is ample: the widest observed think block was
        # 29,281 tokens and RST-DT's longest answer is ~4.4k. The narrow edge --
        # thinking almost the whole budget, then a truncated answer -- fails loudly
        # in reconstruction rather than scoring as a valid parse.
        content = self._post(
            "/completion",
            {"prompt": prompt, "n_predict": forcing["tokens"], "temperature": cfg.temperature, "cache_prompt": True, **body},
            "content",
        )
        if forcing["think_end"] in content:
            return content
        forced_tail = forcing["message"] + forcing["think_end"]
        answer = self._post(
            "/completion",
            {
                # cache_prompt makes the re-send cheap: call 1's prompt and output
                # are already the slot's KV prefix, so this normally prefills nothing.
                "prompt": prompt + content + forced_tail,
                "n_predict": max(cfg.max_tokens - forcing["tokens"], 1),
                "temperature": cfg.temperature,
                "cache_prompt": True,
                "grammar": forcing["answer_grammar"],
            },
            "content",
        )
        # Rejoin so the caller sees one response in the usual shape: reasoning,
        # then the tag, then the constrained answer.
        return content + forced_tail + answer
