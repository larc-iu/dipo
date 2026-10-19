from dataclasses import dataclass, field

from tonga import FromParams

from dipo.rst.parsers.common.config import PeftConfig, parse_config_dict

PARSE_TYPES = ("bottom_up", "top_down")
LABEL_MODES = ("separate", "joint", "unified")
DECODE_MODES = ("generate", "score")
# The adapters each parse type uses: the structural decision, then nuclearity, then the
# relation conditioned on the predicted nuclearity (Maekawa et al.'s best "rel_with_nuc").
TASKS_BY_PARSE_TYPE = {
    "bottom_up": ("span", "nuc", "rel_with_nuc"),
    "top_down": ("top_down", "nuc", "rel_with_nuc"),
}
# label_mode="unified": one decision (and adapter) per parser step answers structure and labels.
UNIFIED_TASK = {"bottom_up": "unified_bu", "top_down": "unified_td"}


@dataclass
class SegmentationConfig(FromParams):
    """The bundled EDU segmenter: `gen`'s `seg` serialization (copy every token, close
    each EDU with a shift) as one more LoRA adapter on the parser's base model, trained
    on the same documents. Selected by dev segmentation F1."""

    epochs: int = 5
    lr: float = 2e-4
    # Documents per optimizer step, split into micro-batches under the parser's
    # max_microbatch_tokens.
    docs_per_step: int = 4
    min_edu_length: int = 1
    # Longest document, in tokens, it reads (training skips longer ones, inference raises).
    max_input_tokens: int = 4096
    inference_batch_size: int = 8


def _default_peft() -> PeftConfig:
    """Maekawa et al.'s LoRA (r=64, alpha=16, dropout 0.1 on every linear layer but the LM
    head, which PEFT's "all-linear" already excludes) on an unquantized bf16 base. Their
    4-bit NF4 base is `load_in_4bit: true` (needs the `qlora` extra)."""
    return PeftConfig(r=64, alpha=16, dropout=0.1, target_modules="all-linear", load_in_4bit=False)


@dataclass
class MaekawaConfig(FromParams):
    """Maekawa et al. (2024, EACL), "Can we obtain significant success in RST discourse
    parsing by using Large Language Models?": a causal LM answers one parsing decision per
    prompt, with LoRA adapters on a shared base.

    The defaults are the strongest setup we found on GUM: top-down parsing with a single
    adapter that answers each node's split point, nuclearity and relation at once, on
    Qwen 3.8 27B. Their own setup (bottom-up, one adapter per decision type, Llama 2 on a
    4-bit base) is `parse_type: "bottom_up"`, `label_mode: "separate"`, a Llama 2 model,
    `peft.load_in_4bit: true`, `gradient_checkpointing: false`, `max_target_tokens: 10`.
    Training hyperparameters are theirs in either case."""

    train_dir: str
    dev_dir: str
    test_dir: str | None = None

    # Inferred at training time from train_dir + dev_dir, persisted for predict.
    relation_types: list[tuple[str, str]] | None = None
    # Optional fine->coarse relation remap applied by the reader (as in every parser).
    relation_map: dict[str, str] | None = None
    # Relations as offered in the relation prompt, in prompt order, named as in the data
    # (after `relation_map`). None = every relation in the data, most frequent in train
    # first, which is how Maekawa et al.'s RST-DT list is ordered. Set by training.
    relation_labels: list[str] | None = None
    # Surface form of each relation in prompts and answers, e.g. {"elaboration":
    # "Elaboration"} to reuse their released RST-DT adapters. None = the data's names.
    relation_surface: dict[str, str] | None = None
    # The answer used when a generated relation is not a legal label (their fallback is
    # the corpus's most frequent relation). None = relation_labels[0].
    default_relation: str | None = None
    # "unified": a single adapter makes every decision, structure and labels at once:
    # top-down "<split> <NS|SN|NN> <relation>", bottom-up "shift" or "reduce <NS|SN|NN>
    # <relation>", labels not listed in the prompt. Needs max_target_tokens room for the
    # longest answer (training checks). "separate" (theirs): the structural decision, then
    # nuclearity, then the relation given it, one adapter each. "joint": separate, but
    # nuclearity and relation as one "<NS|SN|NN> <relation>" decision.
    label_mode: str = "unified"
    # Joint labels as (nuclearity code, relation), most frequent in train first; their
    # order is the joint prompt's and the first is the fallback (joint and unified modes).
    # Set by training.
    joint_labels: list[tuple[str, str]] | None = None

    # Backbone: any causal LM on the Hub.
    model_name: str = "Qwen/Qwen3.8-27B"
    parse_type: str = "top_down"
    # "generate": greedy generation then their default-label fallback on illegal output
    # (faithful to their test.py). "score": argmax over every legal answer's
    # log-likelihood, so the output is always legal.
    decode: str = "generate"
    # Score every legal answer during prediction even under decode="generate", and keep
    # the per-decision distributions on `parser.last_decisions`.
    record_distributions: bool = False
    # Their tokenizer setup adds a "[PAD]" token and resizes the embeddings (new rows
    # initialized to the mean). Kept so their released adapters load unchanged.
    add_pad_token: bool = True
    # task -> Hub repo or local dir of a trained PEFT adapter to load instead of training
    # that task, e.g. {"span": "arumaekawa/rstdt-7b-span", ...}.
    pretrained_adapters: dict[str, str] | None = None

    peft: PeftConfig = field(default_factory=_default_peft)
    # The bundled segmenter, which lets the parser read raw text (predict_from_text) and
    # adds end-to-end scores to evaluation. null = gold or externally segmented EDUs only.
    segmentation: SegmentationConfig | None = field(default_factory=SegmentationConfig)
    # Needed for 27B+ bases. A 7B fits without it at micro-batch 16 under the token cap
    # below and trains about 1.5x faster.
    gradient_checkpointing: bool = True

    # Training, per adapter. Their hyperparameters (train.sh): lr, epochs, an effective
    # batch of 16 decisions, warmup, clipping, decay, betas, length caps.
    lr: float = 2e-4
    epochs: int = 5
    # An optimizer step covers batch_size x grad_accum decisions (16, theirs), split into
    # micro-batches of at most `batch_size` decisions and `max_microbatch_tokens` padded
    # tokens. The loss is a per-decision mean over the step, so any layout optimizes the
    # same objective as theirs. Their exact layout is batch_size 1, grad_accum 16,
    # max_microbatch_tokens null (a plain shuffle); this default (16 x 1, length-grouped
    # batches) trains 5-10x faster.
    batch_size: int = 16
    grad_accum: int = 1
    max_microbatch_tokens: int | None = 4096
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    weight_decay: float = 0.0
    adam_beta2: float = 0.999
    max_source_tokens: int = 4000
    # Room for the longest unified answer ("<split> <NS|SN|NN> <relation>" + EOS); theirs is 10.
    max_target_tokens: int = 24
    # Each epoch's adapter is scored by dev decision accuracy under gold parser states and
    # the best epoch is kept (their test_with_oracle.py). Caps the dev docs used for that.
    dev_max_docs: int | None = None
    # Train only these tasks in this run (e.g. split the adapters over two GPUs, one run
    # each). Every trained adapter is exported as a PEFT dir under <run>/adapters/<task>,
    # which a later config's `pretrained_adapters` can combine. Final evaluation runs only
    # when the run has every adapter its parse type needs.
    only_tasks: list[str] | None = None
    # Inference batching (gold-state dev scoring and lockstep parsing).
    inference_batch_size: int = 32
    inference_max_tokens: int = 32768
    log_every: int = 100

    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None
    seed: int = 42

    def __post_init__(self):
        if self.parse_type not in PARSE_TYPES:
            raise ValueError(f"parse_type must be one of {PARSE_TYPES} (got {self.parse_type!r})")
        if self.label_mode not in LABEL_MODES:
            raise ValueError(f"label_mode must be one of {LABEL_MODES} (got {self.label_mode!r})")
        if self.decode not in DECODE_MODES:
            raise ValueError(f"decode must be one of {DECODE_MODES} (got {self.decode!r})")
        if self.label_mode == "unified" and (self.decode == "score" or self.record_distributions):
            raise ValueError(
                "label_mode='unified' supports decode='generate' only, without record_distributions: "
                "its answers (top-down: every split x every label pair) are validated, not enumerated"
            )
        for name in ("epochs", "batch_size", "grad_accum", "inference_batch_size", "inference_max_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1 (got {getattr(self, name)})")
        if self.max_microbatch_tokens is not None and self.max_microbatch_tokens < 1:
            raise ValueError(f"max_microbatch_tokens must be at least 1 or null (got {self.max_microbatch_tokens})")
        if self.relation_labels is not None:
            if self.default_relation is not None and self.default_relation not in self.relation_labels:
                raise ValueError(f"default_relation {self.default_relation!r} is not in relation_labels")
            unknown = set(self.relation_surface or {}) - set(self.relation_labels)
            if unknown:
                raise ValueError(f"relation_surface names relations not in relation_labels: {sorted(unknown)}")
        if self.only_tasks is not None and not set(self.only_tasks) <= set(self.all_tasks):
            raise ValueError(f"only_tasks {self.only_tasks} must be among {self.all_tasks}")
        if self.pretrained_adapters is not None:
            unknown = set(self.pretrained_adapters) - set(self.all_tasks)
            if unknown:
                raise ValueError(
                    f"pretrained_adapters names tasks {sorted(unknown)} this config does not use ({self.all_tasks})"
                )
            both = set(self.only_tasks or ()) & set(self.pretrained_adapters)
            if both:
                raise ValueError(f"tasks {sorted(both)} are in both only_tasks and pretrained_adapters")

    @property
    def all_tasks(self) -> tuple[str, ...]:
        """The parsing tasks, then "segment" when the segmenter is bundled."""
        return self.tasks + (("segment",) if self.segmentation is not None else ())

    @property
    def tasks(self) -> tuple[str, ...]:
        if self.label_mode == "unified":
            return (UNIFIED_TASK[self.parse_type],)
        structure, *labels = TASKS_BY_PARSE_TYPE[self.parse_type]
        return (structure, "joint") if self.label_mode == "joint" else (structure, *labels)

    @classmethod
    def from_dict(cls, d: dict) -> "MaekawaConfig":
        return parse_config_dict(cls, d)
