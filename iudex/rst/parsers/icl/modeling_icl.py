"""The ICL parser: prompt a frozen LM to linearize a tree, then reconstruct it.

Not an `nn.Module` and not checkpointed. It composes a `provider` (where the LM
lives) and a `serialization` (how the tree is linearized) into three pipelines,
selected by `cfg.pipeline_mode`:

- `e2e`: one call, raw text -> full tree (the model segments too).
- `two_call`: call 1 segments (verbatim + `||`), call 2 parses the resulting
  EDUs. The pilot's winner (segmentation is the bottleneck, so isolating it
  helps).
- `gold_edu`: skip segmentation, parse over gold EDUs (structure+labeling only).

`predict_from_text` / `predict_with_gold_edus` are the prediction surface the
bespoke ICL eval loop calls. ICL has no shared subword tokenizer, so it does not
reuse the generative parsers' `evaluate_on_dev` (the method names only coincide).
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable

from iudex.rst.data.reader import read_rst_dir
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.icl import serialize
from iudex.rst.parsers.icl.configuration_icl import IclConfig
from iudex.rst.parsers.icl.provider import build_provider


class IclParser:
    def __init__(self, cfg: IclConfig, *, icl_examples: list[RstTree] | None = None):
        if cfg.relation_types is None:
            raise ValueError("IclParser requires cfg.relation_types to be inferred before construction.")
        if cfg.grammar.enabled and cfg.grammar.backend != "gbnf":
            raise NotImplementedError(
                f"grammar backend {cfg.grammar.backend!r} is not implemented. Only 'gbnf' (llama.cpp) "
                "emits today; vLLM guided_grammar would need a Lark emitter."
            )
        self.config = cfg
        self.serialization = serialize.get_serialization(cfg.serialization)
        self.provider = build_provider(cfg.provider)
        self.relation_types = cfg.relation_types
        self.grammar_enabled = cfg.grammar.enabled
        self.icl_examples = icl_examples if icl_examples is not None else self._sample_examples()
        # Prefixes are byte-identical across documents (that is what makes the
        # provider's prompt cache pay off), so build each lazily and once.
        self._prefix_cache: dict[str, str] = {}
        # How many calls had their reasoning truncated by grammar.think_budget.
        # Counted per CALL, not per document, because two_call gives a document
        # two chances to be forced. Locked: eval predicts documents concurrently.
        self.forced_calls = 0
        self._forced_lock = threading.Lock()
        # Same thing per DOCUMENT. A bare count cannot answer the question that
        # actually matters -- WHICH documents were truncated -- and without that
        # a bounded run cannot be told apart from one where reasoning simply
        # drifted, which is exactly the ambiguity that stalled the first analysis.
        self._tls = threading.local()

    # ----- ICL example selection -----

    def _sample_examples(self) -> list[RstTree]:
        cfg = self.config
        pairs = read_rst_dir(cfg.train_dir, relation_types=cfg.relation_types, relation_map=cfg.relation_map)
        trees = [t for _, t in pairs]
        if cfg.example_max_edus is not None:
            trees = [t for t in trees if len(t.edu_strings) <= cfg.example_max_edus]
        if len(trees) < cfg.k:
            raise ValueError(f"need k={cfg.k} example docs but only {len(trees)} qualify in {cfg.train_dir}")
        # Permute once and take a prefix, so exemplar sets are NESTED across k
        # (k=1 subset of k=2 subset of k=4 ...). Drawing `sample(trees, k)` per k
        # instead confounds the k-sweep with which documents happened to be
        # drawn, and a lucky small-k draw can beat a larger k for reasons that
        # have nothing to do with k. `read_rst_dir` sorts, so this is reproducible.
        return random.Random(cfg.seed).sample(trees, len(trees))[: cfg.k]

    # ----- prompt assembly -----

    def _build_prefix(
        self,
        task: str,
        render_input: Callable[[RstTree], str],
        render_output: Callable[[RstTree], str],
    ) -> str:
        parts = [task, f"\nHere are {len(self.icl_examples)} worked examples.\n"]
        for i, tree in enumerate(self.icl_examples, 1):
            parts.append(f"--- Example {i} ---")
            parts.append(f"INPUT: {render_input(tree)}")
            parts.append(f"OUTPUT: {render_output(tree)}")
            parts.append("")
        return "\n".join(parts)

    def _e2e_prefix(self) -> str:
        if "e2e" not in self._prefix_cache:
            ser = self.serialization
            self._prefix_cache["e2e"] = self._build_prefix(
                ser.e2e_task_description(self.relation_types),
                lambda t: serialize.doc_text(t.edu_strings),
                ser.render_e2e,
            )
        return self._prefix_cache["e2e"]

    def _seg_prefix(self) -> str:
        if "seg" not in self._prefix_cache:
            self._prefix_cache["seg"] = self._build_prefix(
                serialize.SEGMENTATION_TASK,
                lambda t: serialize.doc_text(t.edu_strings),
                lambda t: serialize.render_segmentation(t.edu_strings),
            )
        return self._prefix_cache["seg"]

    def _parse_prefix(self) -> str:
        if "parse" not in self._prefix_cache:
            ser = self.serialization
            self._prefix_cache["parse"] = self._build_prefix(
                ser.parse_task_description(self.relation_types, echo_edu_text=self.config.echo_edu_text),
                lambda t: ser.parse_input(list(t.edu_strings), echo_edu_text=self.config.echo_edu_text),
                lambda t: ser.render_parse(t, echo_edu_text=self.config.echo_edu_text),
            )
        return self._prefix_cache["parse"]

    @staticmethod
    def _suffix(input_str: str) -> str:
        return f"\n--- Now do this ---\nINPUT: {input_str}\nOUTPUT: "

    def prewarm(self) -> None:
        """Build the (cached, byte-identical) prompt prefixes once up front, so
        concurrent workers share them without racing on lazy construction."""
        mode = self.config.pipeline_mode
        if mode == "e2e":
            self._e2e_prefix()
        elif mode == "two_call":
            self._seg_prefix()
            self._parse_prefix()
        else:
            self._parse_prefix()

    # ----- stages -----

    def _grammar_payload(self, gbnf: str) -> dict | None:
        """Wrap a GBNF string into the request fields the endpoint expects.

        Eager (the default) sends the grammar alone, constraining from token 1.
        Lazy re-roots the grammar behind the think-end trigger and asks llama.cpp
        to hold the constraint until that trigger fires, so the model may reason
        first. `preserved_tokens` is required by the server whenever the trigger
        word is a single token, else it rejects the request outright."""
        if not self.grammar_enabled:
            return None
        g = self.config.grammar
        if not g.lazy:
            return {"grammar": gbnf}
        payload = {
            "grammar": serialize.prefixed_root_grammar(gbnf, g.think_end),
            "grammar_lazy": True,
            # 1 == COMMON_GRAMMAR_TRIGGER_TYPE_WORD in llama.cpp's enum; the server
            # promotes it to a token trigger when the word is a single token.
            "grammar_triggers": [{"type": 1, "value": g.think_end}],
            "preserved_tokens": [g.think_end],
        }
        if g.think_budget is not None:
            # Private handoff to OpenAICompatibleProvider._native_complete, which
            # implements budget forcing as two calls. NOT llama.cpp's own
            # reasoning-budget fields: with those, a FORCED think-end leaves the
            # grammar untriggered and the answer comes back unconstrained (measured;
            # see the provider). `answer_grammar` is the BARE grammar, since by the
            # time it is used the tag is already in the prompt rather than generated.
            payload["_budget_forcing"] = {
                "tokens": g.think_budget,
                "message": g.budget_message,
                "think_end": g.think_end,
                "answer_grammar": gbnf,
            }
        return payload

    def _strip_think_end(self, response: str) -> str:
        """Drop the think block from a lazily-constrained response.

        The native endpoint returns raw text, so the content is the whole
        reasoning trace, then the think-end tag, then the grammar-constrained
        answer. Everything up to and including the first tag is reasoning; the
        grammar guarantees the tag appears exactly once (it is the root's first
        literal and cannot be produced inside the constrained tail).

        A MISSING tag is a hard failure, not something to pass through. The tag is
        what triggers the grammar, so if it never appeared the whole response was
        generated UNCONSTRAINED -- the runaway think block ran to max_tokens and
        was cut off mid-reasoning. Returning it would hand a reasoning trace to
        the SR parser, which then counts whatever `<shift>` strings the model
        happened to write while thinking: wsj_1146 came back as "5815 <shift>
        actions for 304 given EDUs", an error that hides its own cause. Worse, a
        reasoning trace that happens to contain a well-formed sequence would score
        as a valid parse while being unconstrained, silently breaking the one
        guarantee this arm exists to make."""
        if self.grammar_enabled and self.config.grammar.lazy:
            think, sep, answer = response.partition(self.config.grammar.think_end)
            if not sep:
                raise ValueError(
                    f"lazy grammar never triggered: no {self.config.grammar.think_end!r} in a "
                    f"{len(response)}-char response, so it was generated unconstrained (the think "
                    "block most likely ran to provider.max_tokens and was truncated mid-reasoning)."
                )
            g = self.config.grammar
            # The forced tag is preceded by our own injected message, so its
            # presence is an exact marker that this call's reasoning was cut off
            # rather than concluded. The answer is still fully grammar-constrained
            # (that is the point), but it was produced from truncated reasoning.
            if g.think_budget is not None and think.endswith(g.budget_message):
                with self._forced_lock:
                    self.forced_calls += 1
                self._tls.forced = getattr(self._tls, "forced", 0) + 1
            return answer.lstrip()
        return response

    # ----- per-document instrumentation -----

    def begin_doc(self) -> None:
        """Start a document. One worker thread owns one document, so the
        thread-local counters below describe exactly that document."""
        self._tls.forced = 0
        self.provider.begin_doc()

    def doc_stats(self) -> dict:
        """Forced-call count and per-call generated-token counts for the document
        this thread just predicted."""
        return {"forced": getattr(self._tls, "forced", 0), "call_tokens": self.provider.call_tokens()}

    def _segment(self, text: str) -> list[str]:
        grammar = self._grammar_payload(serialize.segmentation_grammar(text)) if self.grammar_enabled else None
        response = self.provider.complete(
            self._seg_prefix(), self._suffix(text), grammar=grammar, system=self.config.system_prompt
        )
        return serialize.parse_segmentation(self._strip_think_end(response))

    def _parse(self, edus: list[str]) -> RstTree:
        grammar = (
            self._grammar_payload(
                serialize.sr_parse_grammar(
                    len(edus), self.relation_types, edus=edus if self.config.echo_edu_text else None
                )
            )
            if self.grammar_enabled
            else None
        )
        response = self.provider.complete(
            self._parse_prefix(),
            self._suffix(self.serialization.parse_input(edus, echo_edu_text=self.config.echo_edu_text)),
            grammar=grammar,
            system=self.config.system_prompt,
        )
        return self.serialization.parse_over_edus(
            self._strip_think_end(response), edus, self.relation_types, echo_edu_text=self.config.echo_edu_text
        )

    def _predict_e2e(self, text: str) -> RstTree:
        response = self.provider.complete(
            self._e2e_prefix(), self._suffix(text), system=self.config.system_prompt
        )
        return self.serialization.parse_e2e(response, self.relation_types)

    # ----- public prediction API -----

    def predict_from_text(self, text: str) -> RstTree:
        mode = self.config.pipeline_mode
        if mode == "e2e":
            return self._predict_e2e(text)
        if mode == "two_call":
            return self._parse(self._segment(text))
        raise ValueError(
            f"predict_from_text is undefined for pipeline_mode={mode!r} (gold_edu needs gold EDUs; "
            "use predict_with_gold_edus)."
        )

    def predict_with_gold_edus(self, tree: RstTree) -> RstTree:
        """Parse-only over the given tree's gold EDUs (structure + labeling).
        Available in any mode, and the sole path for pipeline_mode='gold_edu'."""
        return self._parse(list(tree.edu_strings))

    # ----- introspection (for --dry-run) -----

    def preview_prompt(self, text_or_edus) -> str:
        """Assemble the prompt the FIRST stage would send for one input, for
        --dry-run inspection (no API call). `text_or_edus` is a raw string for
        e2e/two_call, or an EDU list for gold_edu."""
        mode = self.config.pipeline_mode
        if mode == "e2e":
            return self._e2e_prefix() + self._suffix(text_or_edus)
        if mode == "two_call":
            return self._seg_prefix() + self._suffix(text_or_edus)
        edus = text_or_edus if isinstance(text_or_edus, list) else list(text_or_edus.edu_strings)
        return self._parse_prefix() + self._suffix(
            self.serialization.parse_input(edus, echo_edu_text=self.config.echo_edu_text)
        )
