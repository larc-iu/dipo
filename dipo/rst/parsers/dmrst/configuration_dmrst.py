from dataclasses import dataclass, field

from tonga import FromParams

from dipo.rst.parsers.common.config import PeftConfig, parse_config_dict
from dipo.rst.parsers.common.curriculum import Curriculum, SimpleCurriculum
from dipo.rst.parsers.common.detokenization import Detokenizer


@dataclass
class _SegmentationConfig(FromParams):
    """Joint per-token EDU-boundary head. Set `segmentation: null` in
    jsonnet to disable (and lose `predict_from_text`)."""

    pos_weight: float = 10.0  # upweighted because EDU ends are rare
    start_loss: bool = False
    # When `scheme` is set (BIE/BO/EO), use the shared scheme-based segmenter
    # (dipo.rst.parsers.common.segmentation) with `loss` (crf/ce) and `dropout`,
    # instead of the paper's binary end-tagger; `start_loss` is then ignored.
    scheme: str | None = None
    loss: str = "crf"
    dropout: float = 0.5


@dataclass
class _DLWConfig(FromParams):
    """Dynamic loss weighting (paper §3.2). Set `dlw: null` for unweighted sum.

    The weight update compares the mean of the most recent `window // 2`
    optimizer steps' component losses against the mean of the preceding
    `window // 2` (or the rest, for odd `window`). With `window=2` (default)
    this collapses to `L_k(t-1) / L_k(t-2)`, reproducing the paper's
    formulation.
    """

    temperature: float = 2.0
    window: int = 2

    def __post_init__(self):
        if self.window < 2:
            raise ValueError(f"_DLWConfig.window must be >= 2 (got {self.window})")


@dataclass
class DMRSTConfig(FromParams):
    train_dir: str
    dev_dir: str
    test_dir: str | None = None

    # Inferred at training time from train_dir + dev_dir. Persisted so
    # predict / from_pretrained know the label space.
    relation_types: list[tuple[str, str]] | None = None

    # Optional fine→coarse relation remap applied by the reader. When set,
    # every non-"span" relname in the data must be a key (missing keys raise).
    # `relation_types` and the model's label space are in the mapped space.
    relation_map: dict[str, str] | None = None

    # Model
    model_name: str = "xlm-roberta-base"
    stride: int = 100
    # When set, encode with DMRST's original fixed sliding-window scheme
    # (reference module.py EncoderRNN): `encoder_window_size` content tokens per
    # window, `stride` context tokens discarded per interior side, no [CLS]/[SEP].
    # Requires encoder_window_size + 2*stride <= the encoder's positional budget.
    # None (default) uses the shared max-length striding (CLS/SEP per chunk).
    encoder_window_size: int | None = None
    attention_type: str = "dot_product"  # or "biaffine"
    classifier_use_bias: bool = True
    num_rnn_layers: int = 1
    encoder_dropout: float = 0.5
    decoder_dropout: float = 0.5
    labeler_dropout: float = 0.5
    doc_gru_dropout: float = 0.2
    # How to pool the EDUs of each child of a split into the single vector fed
    # to the label classifier:
    #   "mean":     average of all EDU representations in the child
    #   "last_edu": the last EDU representation in the child
    # The two collapse to the same thing for a 2-EDU span (split is forced, each
    # child has exactly one EDU).
    label_input_pooling: str = "mean"
    freeze_embeddings: bool = True
    freeze_encoder_layers: int = 3

    # LoRA encoder fine-tuning. See `PeftConfig`. Mutually exclusive with the
    # freeze fields above (set both to off when enabling peft).
    peft: PeftConfig | None = None

    # Curriculum strategy (Registrable). Default `SimpleCurriculum` reproduces
    # cold full-document training. `SubtreeSizeCurriculum` warms up on small
    # subtrees before full docs. The curriculum owns each phase's train trees,
    # dev set, and epoch budget (the run length).
    curriculum: Curriculum = field(default_factory=SimpleCurriculum)

    # Joint EDU segmentation (paper §3.1.1). See `_SegmentationConfig`.
    segmentation: _SegmentationConfig | None = None

    # Detokenizer for EDU text. Applied only when `segmentation` is non-null, so
    # end-to-end-from-text models train on natural text matching the raw input
    # `predict_from_text` receives. Registrable; see common.detokenization.
    detokenizer: Detokenizer | None = None

    # Tokenize multi-character CJK subwords one character per token, so every
    # EDU boundary in unspaced Chinese falls on a token break (train and raw-text
    # inference alike). See common.encoding.encode_with_offsets.
    split_cjk_tokens: bool = False

    # Dynamic loss weighting (paper §3.2). See `_DLWConfig`.
    dlw: _DLWConfig | None = None

    # Training
    lr: float = 1e-4
    encoder_lr: float | None = 2e-5
    grad_accum: int = 3
    # bf16 autocast on the training forward (CUDA only; bf16 needs no GradScaler).
    # Set false for full-fp32 training. Inference is always fp32.
    amp: bool = True
    # Return PyTorch's cached-but-free blocks to the driver between documents.
    # Purely a memory-management knob (numerically inert, HASH_EXCLUDEd): peak
    # memory here is one document's decoder graph, and documents vary widely in
    # EDU count, so blocks cached for a short document can be too fragmented to
    # satisfy a long one's larger allocation. Costs a synchronize per document,
    # so it is off by default and only worth enabling for backbones that sit near
    # the card's limit (XLM-R XXL on GUM).
    empty_cache_between_docs: bool = False
    # Recompute each decoder span decision during backward instead of keeping
    # every step's activations alive. The decoder dominates this parser's memory:
    # measured on a 250-EDU document with XLM-R XXL, it held 49.0GB against the
    # already-checkpointed encoder's 1.5GB, and enabling this cut it to 1.5GB --
    # peak 83.8GB -> 36.7GB. Off by default because it is a loss on small
    # backbones, where the bookkeeping exceeds the activations saved (measured
    # +17% peak on a 150M encoder). Numerically inert and HASH_EXCLUDEd; costs a
    # second decoder forward, negligible beside the encoder.
    checkpoint_decoder: bool = False
    patience: int = 10
    # Early stopping counts on a trailing mean of the last `patience_window` dev
    # scores rather than raw per-epoch dev (which is noisy, so a raw running-max
    # makes patience fire on a noise dip mid-schedule and truncates the run before
    # the LR decay tail). Checkpoint SELECTION stays raw argmax on `val_metric_name`.
    # HASH_EXCLUDEd (selection-side, resume-safe).
    patience_window: int = 5
    max_grad_norm: float = 5.0
    weight_decay: float = 0.01
    # Linear warmup before linear decay. `num_warmup_steps=None` (the default) warms
    # up over `num_warmup_epochs` epochs (num_warmup_epochs * steps_per_epoch); an
    # explicit int overrides with a literal step count (0 = no warmup).
    num_warmup_steps: int | None = None
    # The 5-epoch default is the discriminative standard (2026-07-27). The old
    # 1-epoch ramp let EuroBERT >=610m diverge on RST-DT the instant the LR hit peak
    # (2.1B loss 12->60 at epoch 2, exactly at warmup exit); a 5-epoch ramp reaches
    # 3e-4 gently enough to hold. Corpus-agnostic: it scales each corpus's own
    # steps_per_epoch, so RST-DT (103/epoch -> 515) and GUM differ automatically.
    num_warmup_epochs: int = 5
    log_every: int = 50
    # Skip dev validation until this epoch (0 = validate from the start). In
    # HASH_EXCLUDE, so changing it is resume-safe. Applies within a validating
    # phase. A curriculum's non-final phases skip validation regardless.
    begin_validation_epoch: int = 0
    # Run dev validation every N epochs (global epoch count shared with the
    # curriculum phase loop). 1 = every epoch. The final epoch always validates.
    validate_every: int = 1
    # Per-document loss weight proportional to (#EDUs ** edu_loss_weight_exponent),
    # normalized to mean 1 over each phase's training set (Hu & Wan 2023 Eq. 2 uses
    # exponent 1). 0.0 disables it (all documents weighted equally). Recomputed per
    # curriculum phase over that phase's trees.
    edu_loss_weight_exponent: float = 0.0
    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None
    seed: int = 42
    val_metric_name: str = "span_f1"

    @classmethod
    def from_dict(cls, d: dict) -> "DMRSTConfig":
        return parse_config_dict(cls, d)
