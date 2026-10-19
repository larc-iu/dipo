"""Maekawa et al. (2024): a causal LM with LoRA adapters, asked one parsing decision per
prompt. Top-down (span split) or bottom-up (shift-reduce) structure. Labels come from the
same decision ("unified", one adapter for everything) or from separate nuclearity and
relation decisions with their own adapters ("separate", theirs, or "joint")."""

import copy
import logging
from collections.abc import Generator

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dipo.rst.data.tree import RstTree
from dipo.rst.parsers.common.seqgen import drop_towers, reconstruct_text
from dipo.rst.parsers.maekawa.configuration_maekawa import MaekawaConfig
from dipo.rst.parsers.maekawa.decisions import (
    DEFAULT_NUC,
    DEFAULT_SPAN,
    NUC_ANSWERS,
    NUC_CODE,
    QUEUE_EMPTY,
    SPAN_ANSWERS,
    Decision,
    Labels,
    joint_prompt,
    nuc_prompt,
    rel_with_nuc_prompt,
    resolve_unified_bu,
    resolve_unified_td,
    span_prompt,
    top_down_answers,
    top_down_prompt,
    tree_from_nodes,
    unified_bu_prompt,
    unified_td_prompt,
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
PAD_TOKEN = "[PAD]"
# Their generate_answer budgets: 5 new tokens for an action or split index, 10 for a label
# (raised to fit the longest legal label, see `_max_new_tokens`).
MAX_NEW_TOKENS = {"span": 5, "top_down": 5, "nuc": 10, "rel_with_nuc": 10}
# PEFT adapter name per task. PEFT selects an adapter's weights by substring match on the
# name, so no name may contain another ("nuc" is inside "rel_with_nuc"; a test checks).
ADAPTER = {
    "span": "span",
    "top_down": "top_down",
    "nuc": "nuc",
    "rel_with_nuc": "rel",
    "joint": "joint",
    "unified_bu": "unified",
    "unified_td": "unified",
    "segment": "segment",
}


def adapter_name(task: str) -> str:
    return ADAPTER[task]


ParseSteps = Generator[Decision, str, list[tuple]]


class MaekawaParser(nn.Module):
    def __init__(self, config: MaekawaConfig, *, compile_encoder: bool = False, training: bool = False):
        super().__init__()
        del compile_encoder  # the shared loaders pass it; there is no separate encoder here
        if config.relation_labels is None:
            raise ValueError("MaekawaConfig.relation_labels is unset; training fills it from the data")
        self.config = config
        # Fixed at construction: the adapters installed below are exactly these.
        self.tasks = config.tasks
        if config.label_mode in ("joint", "unified") and not config.joint_labels:
            raise ValueError(
                f"label_mode={config.label_mode!r} needs MaekawaConfig.joint_labels; training fills it from the data"
            )
        self.labels = Labels(
            config.relation_labels, config.relation_surface, config.default_relation, config.joint_labels
        )
        self.last_decisions: list[dict] = []
        self._answer_len_cache: dict[str, int] = {}
        self.last_decisions_by_doc: list[list[dict]] = []

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        on_gpu = torch.cuda.is_available()
        bnb = config.peft.make_bnb_config() if on_gpu else None
        if config.peft.load_in_4bit and not on_gpu:
            logger.warning("peft.load_in_4bit needs a CUDA GPU; loading the base unquantized on CPU.")
        lm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16 if on_gpu else torch.float32,
            device_map={"": torch.cuda.current_device()} if on_gpu else None,
            **({"quantization_config": bnb} if bnb is not None else {}),
        )
        self._quantized = bnb is not None
        # Multimodal releases (Gemma 4, Qwen 3.5+) carry towers a text-only parser never feeds.
        drop_towers(lm, ("vision_tower", "visual", "audio_tower", "multi_modal_projector"))
        # Not a registered submodule: it wraps this parser's own model. Built before the
        # pad token is added so its new token ids match across runs of the same model.
        self._segmenter = None
        if "segment" in self.installed_tasks:
            from dipo.rst.parsers.maekawa.segmenter import SharedSegmenter, segmenter_config

            if config.relation_types is None:
                raise ValueError("MaekawaConfig.relation_types is unset; training fills it from the data")
            self._segmenter = SharedSegmenter(segmenter_config(config, config.relation_types), lm, self.tokenizer)
        if config.add_pad_token and self.tokenizer.pad_token is None:
            self._add_pad_token(lm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if training:
            if self._quantized:
                from peft import prepare_model_for_kbit_training

                lm = prepare_model_for_kbit_training(
                    lm,
                    use_gradient_checkpointing=config.gradient_checkpointing,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            elif config.gradient_checkpointing:
                lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                lm.enable_input_require_grads()
        self.lm = self._install_adapters(lm)
        if self._segmenter is not None:
            self._segmenter.attach(self.lm)
            pretrained = (config.pretrained_adapters or {}).get("segment")
            if pretrained is not None:
                self._segmenter.load_segment_state_dict(_load_segmenter_extras(pretrained))

    # ---- setup -------------------------------------------------------------------

    def _add_pad_token(self, lm) -> None:
        """Their smart_tokenizer_and_embedding_resize: add "[PAD]" and set its input and
        output embedding rows to the mean of the existing rows (exact mean, not
        transformers' sampled mean_resizing)."""
        self.tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        lm.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        with torch.no_grad():
            for emb in (lm.get_input_embeddings(), lm.get_output_embeddings()):
                w = emb.weight.data
                w[-1:] = w[:-1].mean(dim=0, keepdim=True)

    def _install_adapters(self, lm):
        from peft import LoraConfig, PeftModel, get_peft_model

        peft = self.config.peft
        fresh = LoraConfig(
            task_type="CAUSAL_LM",
            r=peft.r,
            lora_alpha=peft.alpha,
            lora_dropout=peft.dropout,
            target_modules=peft.target_modules,
            bias=peft.bias,
        )
        pretrained = self.config.pretrained_adapters or {}
        names = [adapter_name(t) for t in self.installed_tasks]
        model = None
        for task, name in zip(self.installed_tasks, names, strict=True):
            if task in pretrained:
                if model is None:
                    model = PeftModel.from_pretrained(lm, pretrained[task], adapter_name=name, is_trainable=False)
                else:
                    model.load_adapter(pretrained[task], adapter_name=name, is_trainable=False)
            elif model is None:
                model = get_peft_model(lm, fresh, adapter_name=name)
            else:
                model.add_adapter(name, fresh)
        return model

    @property
    def trained_tasks(self) -> tuple[str, ...]:
        """Tasks whose adapters this run trains: not pretrained, and in `only_tasks` if set."""
        pretrained = self.config.pretrained_adapters or {}
        only = self.config.only_tasks
        return tuple(t for t in self.config.all_tasks if t not in pretrained and (only is None or t in only))

    @property
    def installed_tasks(self) -> tuple[str, ...]:
        pretrained = self.config.pretrained_adapters or {}
        return tuple(t for t in self.config.all_tasks if t in pretrained or t in self.trained_tasks)

    @property
    def can_parse(self) -> bool:
        """Whether every adapter the parse type needs is present."""
        return set(self.tasks) <= set(self.installed_tasks)

    @property
    def segmenter(self):
        """The bundled segmenter, or None (gold or externally segmented EDUs only)."""
        return self._segmenter

    @property
    def device(self) -> torch.device:
        return next(self.lm.parameters()).device

    def to(self, *args, **kwargs):
        """A bitsandbytes 4-bit base is placed on its GPU at load time and cannot be
        moved (its packed weights would be corrupted), so moving is a no-op then."""
        if self._quantized:
            return self
        if self._segmenter is not None:  # unregistered, so moved by hand
            self._segmenter.head.to(*args, **kwargs)
            self._segmenter.serialization.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ---- checkpointing (LoRA weights only) ----------------------------------------

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        out = {}
        for task in self.trained_tasks:
            if task == "segment":
                state = self._segmenter.segment_state_dict()
            else:
                state = get_peft_model_state_dict(self.lm, adapter_name=adapter_name(task))
            for k, v in state.items():
                out[f"{task}::{k}"] = v.detach().cpu()
        return out

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        from peft import set_peft_model_state_dict

        by_task: dict[str, dict] = {}
        for k, v in state.items():
            task, name = k.split("::", 1)
            by_task.setdefault(task, {})[name] = v
        unknown = set(by_task) - set(self.trained_tasks)
        if unknown:
            raise RuntimeError(f"checkpoint holds adapters {sorted(unknown)} this config does not train")
        for task, sub in by_task.items():
            if task == "segment":
                self._segmenter.load_segment_state_dict(sub)
                continue
            result = set_peft_model_state_dict(self.lm, sub, adapter_name=adapter_name(task))
            unexpected = getattr(result, "unexpected_keys", [])
            if unexpected:
                raise RuntimeError(f"adapter {task!r}: unexpected keys, e.g. {unexpected[:3]}")

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str,
        *,
        device: str | torch.device | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        compile_encoder: bool = False,
    ) -> "MaekawaParser":
        from dipo.rst.parsers.hfhub import load_parser_from_pretrained

        dev = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return load_parser_from_pretrained(
            repo_or_path,
            parser_cls=cls,
            config_cls=MaekawaConfig,
            device=dev,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            compile_encoder=compile_encoder,
        )

    # ---- tokenization --------------------------------------------------------------

    def _prompt_ids(self, prompt: str, truncate: bool = False) -> list[int]:
        """BOS + prompt, as their collator ("<s>{input}") and generate_answer build it."""
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        bos = self.tokenizer.bos_token_id
        ids = ([bos] if bos is not None else []) + ids
        return ids[: self.config.max_source_tokens] if truncate else ids

    def _answer_ids(self, answer: str, truncate: bool = False) -> list[int]:
        """Answer + EOS, tokenized on its own as their collator does ("{output}</s>")."""
        ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        return ids[: self.config.max_target_tokens] if truncate else ids

    # ---- training --------------------------------------------------------------------

    def _decoder(self):
        """The transformer stack under the LM head (LoRA layers injected), so the head can
        run on chosen positions only."""
        return self.lm.get_base_model().get_decoder()

    def forward(self, decisions: list[Decision]) -> dict[str, torch.Tensor]:
        """Causal-LM loss on the answer tokens of a batch of same-task decisions, prompt
        tokens masked. Call `self.activate(task)` first.

        Returns `per_decision` (each decision's mean over its own answer tokens) and `loss`,
        their mean; a step's loss is `per_decision` summed over its micro-batches over its
        decision count, the same objective under any micro-batch layout. The LM head runs
        only at positions that predict answer tokens, so vocab-sized logits never exist for
        the prompt. Sequences are right-padded, which the causal mask hides."""
        seqs, answer_spans = [], []
        for d in decisions:
            p = self._prompt_ids(d.prompt, truncate=True)
            a = self._answer_ids(d.answer, truncate=True)
            seqs.append(p + a)
            answer_spans.append((len(p), len(p) + len(a)))
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        rows, cols, targets, owner = [], [], [], []
        for i, (s, (a0, a1)) in enumerate(zip(seqs, answer_spans, strict=True)):
            input_ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
            for j in range(a0, a1):  # hidden state at j - 1 predicts token j
                rows.append(i)
                cols.append(j - 1)
                targets.append(s[j])
                owner.append(i)
        dev = self.device
        hidden = self._decoder()(input_ids=input_ids.to(dev), attention_mask=mask.to(dev)).last_hidden_state
        picked = hidden[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)]
        logits = self._head(picked)
        token_loss = torch.nn.functional.cross_entropy(
            logits.float(), torch.tensor(targets, device=dev), reduction="none"
        )
        owner_t = torch.tensor(owner, device=dev)
        sums = torch.zeros(len(seqs), device=dev).index_add_(0, owner_t, token_loss)
        counts = torch.zeros(len(seqs), device=dev).index_add_(0, owner_t, torch.ones_like(token_loss))
        per_decision = sums / counts
        return {"loss": per_decision.mean(), "per_decision": per_decision}

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        base = self.lm.get_base_model()
        logits = base.get_output_embeddings()(hidden)
        # Composite models (Gemma 4) keep the cap on their text config.
        cap = getattr(base.config.get_text_config(), "final_logit_softcapping", None)
        if cap:  # Gemma-family models squash logits in their CausalLM forward, not in the head
            logits = torch.tanh(logits / cap) * cap
        return logits

    # ---- one decision -------------------------------------------------------------

    def candidates(self, d: Decision) -> list[str]:
        if d.task == "span":
            return list(SPAN_ANSWERS)
        if d.task == "nuc":
            return list(NUC_ANSWERS)
        if d.task == "rel_with_nuc":
            return list(self.labels.answers)
        if d.task == "joint":
            return list(self.labels.joint_answers)
        if d.task == "top_down":
            b, _, e = d.key
            return top_down_answers(b, e)
        if d.task in ("unified_bu", "unified_td"):
            raise ValueError("unified answers are validated part by part (resolve_unified_*), not enumerated")
        raise ValueError(f"unknown task {d.task!r}")

    def _longest_answer_tokens(self, prefix: str, answers: list[str] | None = None) -> int:
        """Tokens in the longest "<prefix> <answer>" (prefix "" = the bare answer), over the
        joint labels unless `answers` is given."""
        answers = self.labels.joint_answers if answers is None else answers
        key = (prefix, tuple(answers))
        cache = self._answer_len_cache
        if key not in cache:
            texts = [f"{prefix} {a}" if prefix else a for a in answers]
            cache[key] = max(len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts)
        return cache[key]

    def _max_new_tokens(self, task: str, decisions: list[Decision] = ()) -> int:
        """Their budgets, raised where needed to fit the longest legal answer plus EOS
        (top-down unified: with the batch's widest split index)."""
        floor = MAX_NEW_TOKENS["rel_with_nuc"]
        if task == "nuc":
            return max(floor, self._longest_answer_tokens("", list(NUC_ANSWERS)) + 1)
        if task == "rel_with_nuc":
            return max(floor, self._longest_answer_tokens("", self.labels.answers) + 1)
        if task == "joint":
            return max(floor, self._longest_answer_tokens("") + 1)
        if task == "unified_bu":
            return max(floor, self._longest_answer_tokens("reduce") + 1)
        if task == "unified_td":
            widest = max(d.key[2] - d.key[0] - 2 for d in decisions)
            return max(floor, self._longest_answer_tokens(str(widest)) + 1)
        return MAX_NEW_TOKENS[task]

    def resolve(self, d: Decision, text: str) -> tuple[str, list[str]]:
        """A generated answer made legal: the answer and the parts that fell back to a
        default ("answer" for the single-part tasks)."""
        if d.task == "unified_bu":
            return resolve_unified_bu(text, d.key[4] != QUEUE_EMPTY, self.labels)
        if d.task == "unified_td":
            return resolve_unified_td(text, d.key[0], d.key[2], self.labels)
        if text in self.candidates(d):
            return text, []
        return self._default(d.task), ["answer"]

    def _default(self, task: str) -> str:
        """Their fallbacks for an illegal answer: shift, split 0 (the unified top-down
        fallback is the middle split instead), the most frequent label."""
        if task == "span":
            return DEFAULT_SPAN
        if task == "top_down":
            return "0"
        if task == "nuc":
            return DEFAULT_NUC
        if task == "rel_with_nuc":
            return self.labels.default_answer
        return self.labels.joint_answers[0]  # joint

    def activate(self, task: str) -> None:
        """Route the LM through `task`'s adapter (and, in training, make it the trainable one)."""
        self.lm.set_adapter(adapter_name(task))

    @torch.no_grad()
    def decide_batch(self, decisions: list[Decision]) -> list[str]:
        """Answers to many prompts of one task. Under decode="generate" they run as
        left-padded greedy-generation batches of at most `inference_batch_size` prompts and
        `inference_max_tokens` padded prompt tokens, longest first; scoring runs per prompt.
        Per-decision records are appended to `last_decisions` in input order."""
        if not decisions:
            return []
        task = decisions[0].task
        if any(d.task != task for d in decisions):
            raise ValueError("decide_batch takes decisions of a single task")
        self.activate(task)
        prompts = [self._prompt_ids(d.prompt) for d in decisions]
        records = [
            {"task": task, "key": list(d.key), "prompt_tokens": len(p)} for d, p in zip(decisions, prompts, strict=True)
        ]
        answers: list[str | None] = [None] * len(decisions)
        if self.config.decode == "generate":
            for chunk in self._inference_chunks(prompts):
                budget = self._max_new_tokens(task, [decisions[i] for i in chunk])
                texts = self._generate([prompts[i] for i in chunk], budget)
                for i, text in zip(chunk, texts, strict=True):
                    records[i]["generated"] = text
                    answers[i], records[i]["fallback"] = self.resolve(decisions[i], text)
        if self.config.decode == "score" or self.config.record_distributions:
            for i, d in enumerate(decisions):
                cands = self.candidates(d)
                logprobs = self.score(torch.tensor([prompts[i]], device=self.device), cands)
                records[i]["candidates"], records[i]["logprobs"] = cands, logprobs
                if self.config.decode == "score":
                    answers[i] = cands[max(range(len(cands)), key=logprobs.__getitem__)]
        for r, a in zip(records, answers, strict=True):
            r["answer"] = a
        self.last_decisions.extend(records)
        return answers

    def _inference_chunks(self, prompts: list[list[int]]) -> list[list[int]]:
        """Indices of `prompts`, longest first, cut into batches under the inference caps."""
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        chunks, cur = [], []
        for i in order:
            longest = len(prompts[cur[0]]) if cur else len(prompts[i])
            if cur and (
                len(cur) + 1 > self.config.inference_batch_size
                or longest * (len(cur) + 1) > self.config.inference_max_tokens
            ):
                chunks.append(cur)
                cur = []
            cur.append(i)
        if cur:
            chunks.append(cur)
        return chunks

    def _generate(self, prompts: list[list[int]], max_new_tokens: int) -> list[str]:
        width = max(len(p) for p in prompts)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(prompts), width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(prompts):  # left padding: every prompt ends where generation starts
            ids[i, width - len(p) :] = torch.tensor(p, device=self.device)
            mask[i, width - len(p) :] = 1
        # An explicit config, so the checkpoint's own generation_config (Llama 2's asks for
        # sampling) cannot leak in. Their pinned transformers decoded greedily too.
        greedy = GenerationConfig(
            do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=pad, eos_token_id=self.tokenizer.eos_token_id
        )
        out = self.lm.generate(input_ids=ids, attention_mask=mask, generation_config=greedy)
        return [self.tokenizer.decode(row[width:], skip_special_tokens=True).strip() for row in out]

    @torch.no_grad()
    def score(self, prefix_ids: torch.Tensor, cands: list[str], chunk: int = 8) -> list[float]:
        """Log-likelihood of each candidate answer (its tokens + EOS) after the prompt.
        The prompt runs once, and its KV cache is shared by every candidate."""
        out = self.lm(input_ids=prefix_ids, use_cache=True)
        last = torch.log_softmax(out.logits[0, -1].float(), -1)
        prefix_len = prefix_ids.shape[1]
        scores: list[float] = []
        for start in range(0, len(cands), chunk):
            group = [self._answer_ids(c) for c in cands[start : start + chunk]]
            k, width = len(group), max(len(g) for g in group)
            inp = torch.full((k, width), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
            mask = torch.zeros((k, prefix_len + width), dtype=torch.long, device=self.device)
            mask[:, :prefix_len] = 1
            for i, g in enumerate(group):
                inp[i, : len(g)] = torch.tensor(g, device=self.device)
                mask[i, prefix_len : prefix_len + len(g)] = 1
            past = copy.deepcopy(out.past_key_values)
            past.batch_repeat_interleave(k)
            logits = self.lm(input_ids=inp, attention_mask=mask, past_key_values=past).logits
            lp = torch.log_softmax(logits.float(), -1)
            for i, g in enumerate(group):
                total = last[g[0]].item()
                for j in range(1, len(g)):
                    total += lp[i, j - 1, g[j]].item()
                scores.append(total)
        return scores

    # ---- whole trees ----------------------------------------------------------------
    # A document's parse is a generator that yields each Decision it needs and receives the
    # answer, so one implementation runs one document at a time or many in lockstep.

    def _label_steps(self, edus: list[str], left: tuple[int, int], right: tuple[int, int]) -> ParseSteps:
        node = (left[0], right[0], right[1])
        if "joint" in self.tasks:
            label = yield Decision("joint", node, joint_prompt(edus, left, right, self.labels.joint_answers), "")
            return self.labels.joint_of[label]
        nuc = yield Decision("nuc", node, nuc_prompt(edus, left, right), "")
        rel = yield Decision("rel_with_nuc", node, rel_with_nuc_prompt(edus, left, right, nuc, self.labels.answers), "")
        return NUC_CODE[nuc], self.labels.relation_of[rel]

    def _bottom_up_steps(self, edus: list[str]) -> ParseSteps:
        n = len(edus)
        stack: list[tuple[int, int]] = []
        queue = 0
        nodes = []
        unified = "unified_bu" in self.tasks
        while not (len(stack) == 1 and queue == n):
            labels = None
            if unified and len(stack) >= 2:
                # One decision wherever a reduce is legal, an empty queue included (the
                # action is forced there, the labels are not); the answer arrives legal.
                q = queue if queue < n else None
                key = (*stack[-2], *stack[-1], QUEUE_EMPTY if q is None else q)
                answer = yield Decision("unified_bu", key, unified_bu_prompt(edus, stack[-2], stack[-1], q), "")
                act, _, label = answer.partition(" ")
                labels = self.labels.joint_of[label] if act == "reduce" else None
            elif len(stack) >= 2 and queue < n:
                key = (*stack[-2], *stack[-1], queue)
                act = yield Decision("span", key, span_prompt(edus, stack[-2], stack[-1], queue), "")
            else:
                act = "shift" if queue < n else "reduce"
            if act == "shift":
                stack.append((queue, queue + 1))
                queue += 1
                continue
            right, left = stack.pop(), stack.pop()
            if labels is None:
                labels = yield from self._label_steps(edus, left, right)
            nodes.append((left[0], right[0], right[1], *labels))
            stack.append((left[0], right[1]))
        return nodes

    def _top_down_steps(self, edus: list[str]) -> ParseSteps:
        nodes = []
        todo = [(0, len(edus))] if len(edus) > 1 else []
        while todo:
            b, e = todo.pop(0)
            k = 0
            if "unified_td" in self.tasks:  # every span, 2-EDU ones too (for the labels)
                answer = yield Decision("unified_td", (b, -1, e), unified_td_prompt(edus, b, e), "")
                split, label = answer.split(" ", 1)
                s = b + 1 + int(split)
                nuc, rel = self.labels.joint_of[label]
                nodes.append((b, s, e, nuc, rel))
                todo += [span for span in ((b, s), (s, e)) if span[1] - span[0] > 1]
                continue
            if e - b > 2:
                k = int((yield Decision("top_down", (b, -1, e), top_down_prompt(edus, b, e), "")))
            s = b + 1 + k
            nuc, rel = yield from self._label_steps(edus, (b, s), (s, e))
            nodes.append((b, s, e, nuc, rel))
            todo += [span for span in ((b, s), (s, e)) if span[1] - span[0] > 1]
        return nodes

    @torch.no_grad()
    def predict_from_text(self, text: str) -> RstTree:
        """Segment raw text with the bundled segmenter, then parse it."""
        return self.predict_batch_from_texts([text])[0]

    @torch.no_grad()
    def predict_batch_from_texts(self, texts: list[str]) -> list[RstTree]:
        """Segment, then parse, many documents. Each tree keeps the segmenter's source
        token ranges, which end-to-end evaluation keys segmentation on."""
        if self._segmenter is None:
            raise RuntimeError("this parser has no segmenter (MaekawaConfig.segmentation is null); pass EDUs")
        self.eval()
        segmented = self._segmenter.segment(texts)
        preds = self.predict_batch(segmented)
        for pred, seg in zip(preds, segmented, strict=True):
            pred._pred_edu_source_ranges = seg._pred_edu_source_ranges  # type: ignore[attr-defined]
            pred._source_ids = seg._source_ids  # type: ignore[attr-defined]
        return preds

    @torch.no_grad()
    def predict(self, tree: RstTree) -> RstTree:
        """Parse over `tree`'s EDUs (gold, or another system's segmentation)."""
        pred = self.predict_batch([tree])[0]
        self.last_decisions = self.last_decisions_by_doc[0]
        return pred

    @torch.no_grad()
    def predict_batch(self, trees: list[RstTree]) -> list[RstTree]:
        """Parse many documents in lockstep: each round collects every unfinished document's
        next decision and answers them in task-grouped batches (`decide_batch`). Decisions,
        and so trees, are the ones `predict` makes one document at a time, up to numerics.
        Per-document decision records land in `last_decisions_by_doc`."""
        self.eval()
        steps_fn = self._bottom_up_steps if self.config.parse_type == "bottom_up" else self._top_down_steps
        gens = [steps_fn([e.strip() for e in t.edu_strings]) for t in trees]
        results: list[list[tuple] | None] = [None] * len(trees)
        by_doc: list[list[dict]] = [[] for _ in trees]
        pending: dict[int, Decision] = {}
        for i, g in enumerate(gens):
            try:
                pending[i] = next(g)
            except StopIteration as stop:
                results[i] = stop.value
        while pending:
            by_task: dict[str, list[int]] = {}
            for i, d in pending.items():
                by_task.setdefault(d.task, []).append(i)
            for docs in by_task.values():
                self.last_decisions = []
                answers = self.decide_batch([pending[i] for i in docs])
                for i, a, rec in zip(docs, answers, self.last_decisions, strict=True):
                    by_doc[i].append(rec)
                    try:
                        pending[i] = gens[i].send(a)
                    except StopIteration as stop:
                        results[i] = stop.value
                        del pending[i]
        self.last_decisions_by_doc = by_doc
        return [tree_from_nodes(nodes, t, self.config.relation_types) for nodes, t in zip(results, trees, strict=True)]


def _load_segmenter_extras(path: str) -> dict[str, torch.Tensor]:
    """The segmenter state an exported adapter dir carries next to its PEFT files."""
    import os

    from huggingface_hub import hf_hub_download

    local = os.path.join(path, SEGMENTER_FILE)
    file = local if os.path.exists(local) else hf_hub_download(path, SEGMENTER_FILE)
    return torch.load(file, map_location="cpu", weights_only=True)


SEGMENTER_FILE = "segmenter.pt"


class EndToEnd:
    """Segment-then-parse in the shape gen's evaluation expects (`evaluate_on_dev`):
    `predict_batch` over gold trees reads their text, not their EDUs."""

    def __init__(self, parser: MaekawaParser):
        self.parser = parser
        self.tokenizer = parser.tokenizer

    def eval(self):
        self.parser.eval()

    def decodes_in_batches(self, *, num_beams=None) -> bool:
        return True

    def predict_batch(self, trees: list[RstTree], *, num_beams=None) -> list[RstTree]:
        return self.parser.predict_batch_from_texts([reconstruct_text(t) for t in trees])
