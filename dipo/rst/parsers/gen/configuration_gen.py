"""Config for the unified generative parser `gen`.

`gen` collapses the four hand-written generative parsers (`{seq2seq,decoder_only}
_{sr,sexp}`) into one, dispatching on two orthogonal discriminators:

- `backbone`: "seq2seq" (encoder-decoder, two-stream + cross-attn) or
  "decoder_only" (causal, single source+output stream).
- `serialization`: "sr" (bottom-up shift-reduce action sequence) or "sexp"
  (nested s-expression).

Fields that are only meaningful for one axis value are documented as such and
validated in `__post_init__` (e.g. `chat_template` needs `backbone="decoder_only"`,
`width_band_loss` needs `serialization="sr"`, the sexp knobs need
`serialization="sexp"`).
"""

from dataclasses import dataclass, field

from tonga import FromParams

from dipo.common.log import warn
from dipo.rst.parsers.common.config import PeftConfig, parse_config_dict
from dipo.rst.parsers.common.curriculum import Curriculum, SimpleCurriculum

_BACKBONES = ("seq2seq", "decoder_only")
_SERIALIZATIONS = ("sr", "sexp")

# Knobs that existed when earlier runs were trained and no longer do; each maps to the
# value that is now the ONLY behavior. A finished run's frozen config.json still carries
# them, and tonga rejects unknown keys, so without this every pre-retirement run would
# be unloadable -- including by `dipo gen eval`, which re-evaluates a run from its
# frozen config. Dropped (loudly) when they match today's behavior; a value we can no
# longer reproduce raises rather than loading a config that misdescribes the run.
_RETIRED_KEYS = {
    "use_validity_constraints": True,  # decoding is always constrained
    "constrain_content": True,  # content is always pinned to the source cursor
}

# Selection-side knobs that were removed. They never touched the trained model
# (HASH_EXCLUDEd; they only shaped WHEN early-stopping fired), so an archived
# config.json that still carries one re-evaluates fine -- drop it on load, value
# and all, rather than raise like a behavior-bearing retired key.
#   patience_min_delta: a min-delta threshold on the smoothed dev stop signal that
#   ticked `stale` while the metric was still at its peak (stopping mid-anneal);
#   removed so ANY improvement resets patience.
_IGNORED_KEYS = {"patience_min_delta"}


@dataclass
class ChatTemplateConfig(FromParams):
    """Wrap the stream in the backbone's chat template (post-trained models).
    `backbone="decoder_only"` only. The document goes in the user turn after
    `instruction`, the output in the assistant turn, terminated by the model's
    turn-end token instead of raw EOS."""

    instruction: str = (
        "Segment the following document into elementary discourse units and "
        "parse it into an RST tree."
    )


@dataclass
class WidthBandLoss(FromParams):
    """Upweight the CE loss at reduce positions whose resulting constituent spans
    [min_width, max_width] EDUs (max_width null = unbounded). `serialization="sr"`
    only. Targets the mid-width (5-16 EDU) decision deficit (2026-06-10 cascade
    probe). Composes with `action_loss_weight` (that rebalances all structural vs
    copy positions, this rebalances within reduces by width)."""

    min_width: int = 5
    max_width: int | None = 16
    weight: float = 2.0

    def __post_init__(self):
        if self.min_width < 2:
            raise ValueError("width_band_loss.min_width must be >= 2 (a reduce spans at least 2 EDUs)")
        if self.max_width is not None and self.max_width < self.min_width:
            raise ValueError("width_band_loss.max_width must be >= min_width or null")
        if self.weight <= 0:
            raise ValueError("width_band_loss.weight must be > 0")

    def covers(self, width: int) -> bool:
        return width >= self.min_width and (self.max_width is None or width <= self.max_width)


@dataclass
class GenConfig(FromParams):
    train_dir: str
    dev_dir: str

    # --- Axis discriminators (what the parser dispatches on) ---
    backbone: str  # "seq2seq" | "decoder_only"
    serialization: str  # "sr" | "sexp"

    test_dir: str | None = None

    # Inferred at training time. Persisted so predict / from_pretrained know the
    # action vocabulary to register on the tokenizer.
    relation_types: list[tuple[str, str]] | None = None
    relation_map: dict[str, str] | None = None

    # Model. seq2seq wants an AutoModelForSeq2SeqLM (e.g. t5gemma-2-1b-1b),
    # decoder_only an AutoModelForCausalLM (e.g. gemma-3-1b-it). The defaults
    # below are the decoder_only 1B baseline; seq2seq configs set these
    # explicitly (t5gemma, 4096/6144, gradient_checkpointing=True).
    model_name: str = "google/gemma-3-1b-it"
    # seq2seq: encoder input length. decoder_only: source portion of the single
    # stream. Backbone-dependent default; set explicitly per config.
    max_input_length: int = 3072
    max_output_length: int = 5120
    gradient_checkpointing: bool = False

    # LoRA. Null = full fine-tuning. When set, the base stack is frozen and only the
    # LoRA adapters train, plus the newly-added token rows via the new-row shadow
    # (`peft.modules_to_save` is inert, see PeftConfig). The practical default at 1B+.
    peft: PeftConfig | None = None

    # `backbone="decoder_only"` only. Non-null: fine-tune through the tokenizer's
    # chat template. Null keeps the raw [BOS] source [SEP] stream. Hashed.
    # NOT yet wired in gen: a non-null value raises at construction.
    chat_template: ChatTemplateConfig | None = None

    # Curriculum strategy (Registrable). Default `SimpleCurriculum` reproduces
    # cold full-document training. The curriculum owns each phase's train trees,
    # dev set, and epoch budget (the run length).
    curriculum: Curriculum = field(default_factory=SimpleCurriculum)

    # Training
    lr: float = 3e-5
    # Separate learning rate for the newly-added token rows (via the new-row
    # shadow on the input embedding). Null = train them at `lr`. At a low
    # full-FT `lr` (e.g. 2e-5) the new rows barely leave their random init
    # (measured: norm stays at ~0.7x pretrained); a higher new_row_lr (~1e-3)
    # trains them properly. Weight decay is dropped on the new rows. Full-head
    # modes require a tied lm_head for this to reach the output side
    # (configure_new_row_training rejects untied). Hashed (changes the trained
    # model).
    new_row_lr: float | None = None
    weight_decay: float = 0.01
    batch_size: int = 1
    grad_accum: int = 16
    # "adamw" (two state tensors per param) or "adafactor" (factored 2nd moment,
    # far less memory). Use adafactor when the AdamW footprint OOMs.
    optimizer: str = "adafactor"
    num_warmup_steps: int | None = None
    # WSD-style LR schedule (make_wsd_scheduler): a short initial warmup, then HOLD
    # peak through the (unvalidated) subtree curriculum phases, a short re-warmup
    # into the full-document phase (eases the biggest distribution shift and the
    # decoder-only-sexp cold-start collapse), then linear decay to `min_lr_frac` of
    # peak over the rest of the full-doc phase. So the anneal lands entirely inside
    # the validated phase where accuracy is made, instead of being spent on subtrees.
    min_lr_frac: float = 0.1
    fulldoc_warmup_steps: int = 150
    max_grad_norm: float = 1.0
    amp: bool = True
    patience: int = 5
    # Early stopping runs on a TRAILING-MEAN of the last `patience_window` dev scores
    # (per-epoch full-dev is noisy, ~+-0.02; a raw running max makes patience fire on
    # a noise plateau mid-schedule). Checkpoint SELECTION stays raw argmax on
    # `val_metric_name`; only the stop signal is smoothed. `stale` increments once the
    # window is full and the smoothed score fails to beat the best smoothed score at
    # all -- NO min-delta threshold (a threshold ticks `stale` while the metric is
    # still at its peak, stopping mid-anneal). HASH_EXCLUDEd (selection-side, resume-safe).
    patience_window: int = 5
    log_every: int = 5
    # Skip dev validation until this epoch OF THE VALIDATING PHASE (0 = validate
    # from the phase start). Resume-safe (in HASH_EXCLUDE). Non-final curriculum
    # phases skip validation regardless.
    begin_validation_epoch: int = 0
    # Run dev validation every N epochs (global epoch count shared with the
    # curriculum phase loop). 1 = every epoch. The final epoch always validates.
    validate_every: int = 1
    checkpoint_dir: str = "checkpoints"
    # Save only requires_grad parameters (LoRA adapters, new embedding rows,
    # action head; frozen base comes from GenParser(cfg) on reload). Turns a
    # 62GB full-state .pt into ~1GB at 31B. Loads non-strict. HASH_EXCLUDEd.
    checkpoint_trainable_only: bool = False
    run_name: str | None = None
    seed: int = 42
    val_metric_name: str = "e2e_full_f1"

    # Decoding
    # Greedy by default: the final eval matches the greedy main-table decode, so
    # final_metrics.json is the paper number with no separate re-eval. Set >1 to
    # beam-search the final eval instead (an optional ablation).
    num_beams: int = 1
    eval_decode_greedy: bool = True
    # Min copies/content tokens before a boundary (shift / leaf-close) is legal
    # at decode time (inference only). 1 = off, bump to 2-3 to suppress
    # over-segmentation. At end-of-source the boundary is always legal.
    min_edu_length: int = 1
    # Cap per-epoch dev eval to the first N docs. None = full dev each epoch.
    # The final dev/test eval is always on the full split.
    dev_max_docs: int | None = None
    # Dev/test prediction chunk size. Under `batched_decode` this is the real
    # decode batch width (documents per shared forward); otherwise gen decodes
    # documents one at a time and this only groups the eval's progress logging.
    dev_batch_size: int = 1
    # Decode `dev_batch_size` documents per forward instead of one at a time
    # (greedy only; beam is per-document either way). Each row drives its own
    # automaton and mask, so the trees are the same ones the per-document path
    # produces -- pinned as exact equality in
    # tests/test_gen_batched_greedy_equivalence.py.
    #
    # Default OFF on purpose. That equality is exact for the arithmetic, but a
    # batched matmul is free to reduce in a different order than a batch-1 one,
    # which can perturb a logit in the last bits and flip a near-tied argmax. The
    # published table cells were all decoded per-document, and several configs
    # already carry dev_batch_size > 1 from when it was inert, so keying batching
    # off that alone would silently re-decode those runs in a second regime.
    # Opting in is therefore explicit, and per-run.
    batched_decode: bool = False

    # Loss
    # Gradient multiplier on structural positions (shift/reduce or parens/labels).
    # 1.0 = no rebalance. Bump only if action positions are demonstrably starved.
    action_loss_weight: float = 1.0
    # `serialization="sr"` only. Upweight reduces by the EDU width of the
    # constituent they create. Null = off. See WidthBandLoss.
    # NOT yet wired in gen: a non-null value raises in train_gen (the shared
    # loss_terms has no per-position weighting).
    width_band_loss: WidthBandLoss | None = None
    # Per-document loss weight proportional to (#EDUs ** exponent), normalized to
    # mean 1 over each phase's training set (Hu & Wan 2023 Eq. 2 uses exponent 1).
    # 0.0 disables it. Recomputed per curriculum phase.
    edu_loss_weight_exponent: float = 0.0
    # Label smoothing on the CE loss.
    label_smoothing: float = 0.1

    # --- Serialization knobs ---
    # `serialization="sexp"` only. Leaf/node emission order: "postorder" or
    # "preorder".
    traversal_order: str = "postorder"
    # True (default): source subwords become a `<copy>` sentinel scored by a small
    # fresh action head. False (sexp only): source subwords appear verbatim and
    # the full pretrained lm_head scores them (Hu & Wan 2023). SR always copies,
    # so use_copy must be True when serialization="sr".
    use_copy: bool = True
    # Label representation. "token": one opaque fused <reduce_{nuc}_{rel}> id per
    # merge over the small action head. "words": abbreviated nuclearity + relation
    # spelled as natural words over the full pretrained lm_head (reuses pretrained
    # knowledge); for sexp also switches brackets to literal { }. Requires
    # use_copy=True.
    label_style: str = "token"

    def __post_init__(self):
        if self.backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {_BACKBONES} (got {self.backbone!r})")
        if self.serialization not in _SERIALIZATIONS:
            raise ValueError(f"serialization must be one of {_SERIALIZATIONS} (got {self.serialization!r})")
        if self.label_style not in ("token", "words"):
            raise ValueError(f"label_style must be 'token' or 'words' (got {self.label_style!r})")
        if self.new_row_lr is not None and self.new_row_lr <= 0:
            raise ValueError(f"new_row_lr must be > 0 or null (got {self.new_row_lr!r})")
        if self.traversal_order not in ("preorder", "postorder"):
            raise ValueError(f"traversal_order must be 'preorder' or 'postorder' (got {self.traversal_order!r})")

        if self.chat_template is not None and self.backbone != "decoder_only":
            raise ValueError("chat_template requires backbone='decoder_only'.")
        if self.width_band_loss is not None and self.serialization != "sr":
            raise ValueError("width_band_loss requires serialization='sr'.")

        if self.serialization == "sr" and not self.use_copy:
            raise ValueError("serialization='sr' always copies content; use_copy must be True.")

        # words mode copies content and moves only labels to the full lm_head, so it
        # requires use_copy=True.
        if self.label_style == "words" and not self.use_copy:
            raise ValueError("label_style='words' requires use_copy=True (content stays copied; labels move to the full lm_head).")
        if self.use_copy is False and self.peft is not None:
            # use_copy=False (Hu & Wan full-vocab content) must emit source subwords over
            # the pretrained lm_head, which LoRA freezes. The hand-written sexp parsers
            # wired `modules_to_save` + a retie to train it; gen does not, so reject the
            # combo loudly rather than train it degraded. Use full fine-tuning for this arm.
            raise ValueError(
                "use_copy=False under LoRA is not supported in gen (the full lm_head must train to emit "
                "source subwords, but LoRA freezes it). Use full fine-tuning (peft=null) for the Hu & Wan arm."
            )

    @classmethod
    def from_dict(cls, d: dict) -> "GenConfig":
        d = dict(d)  # parse_config_dict pops keys; never mutate the caller's dict
        retired = []
        for key, only_behavior in _RETIRED_KEYS.items():
            if key not in d:
                continue
            value = d.pop(key)
            if value != only_behavior:
                raise ValueError(
                    f"This config sets the retired key {key}={value!r}, which gen can no longer "
                    f"reproduce ({key} is now always {only_behavior!r}). The run it describes was "
                    f"trained under behavior this code does not implement."
                )
            retired.append(key)
        if retired:
            warn(f"Ignoring retired config key(s) {sorted(retired)}: they now describe the only behavior.")
        for key in _IGNORED_KEYS & d.keys():
            d.pop(key)
        return parse_config_dict(cls, d)
