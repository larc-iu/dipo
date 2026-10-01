// Shared base for the gen <=4B RST-DT words-mode validation grid.
// Wave-1 recipe (subtree-size curriculum, beam-6, seed 42) ported to `gen`.
// Concrete configs set backbone / serialization / model_name / max_output_length /
// peft / lr / run_name (+ sexp knobs). label_style='words' is the go-forward
// serialization (relation words over the full lm_head, <label_end>-terminated).
// num_beams=1 => the final eval is GREEDY (both e2e and gold-EDU), matching the
// main-table decode, so final_metrics.json is the paper number with no re-eval.
// Beam-6 is an OPTIONAL add-on: `iudex gen eval <run> --num-beams 6` writes
// final_metrics.beam6.json (does not clobber the greedy final_metrics.json).
// Model selection: validate every OTHER epoch on the FULL dev set (validate_every=2,
// dev_max_docs=null). Dev decode is 50-88% of wall-clock, and train loss does NOT track
// dev at the limits (token: tiny loss moves vs pt-scale dev swings; words: pinned at the
// label-smoothing floor), so we can't drop dev -- but every-2 halves its cost at <=0.7pt
// selection cost. patience counts VALIDATIONS, so patience=10 x validate_every=2 = a
// 20-epoch wait, the same loose "ride the WSD anneal, select best-of-run" semantic as the
// original un-truncated protocol -- just at half the dev cost.
{
    train_dir: 'data/rstdt/train',
    dev_dir: 'data/rstdt/dev',
    test_dir: 'data/rstdt/test',
    relation_types: null,
    relation_map: import '../lib/rstdt_coarse_map.libsonnet',

    max_input_length: 16384,
    gradient_checkpointing: true,
    chat_template: null,
    label_style: 'words',

    curriculum: import '_curric_grid.libsonnet',

    weight_decay: 0.05,
    batch_size: 1,
    grad_accum: 8,
    optimizer: 'adafactor',
    num_warmup_steps: null,
    max_grad_norm: 1.0,
    amp: true,
    patience: 5,                                           // 2026-08-10: unified protocol (5 validations = 10 epochs at validate_every 2); reported runs used 10
    patience_window: 1,                                    // 2026-08-10: raw-value stopping (window 1); reported runs used the default 5-validation trailing mean
    log_every: 5,
    begin_validation_epoch: 0,
    validate_every: 2,
    checkpoint_dir: 'checkpoints',
    checkpoint_trainable_only: false,
    run_name: null,
    seed: 42,
    val_metric_name: 'e2e_full_f1',

    action_loss_weight: 1.0,
    edu_loss_weight_exponent: 0.0,
    label_smoothing: 0.1,

    dev_max_docs: null,
    dev_batch_size: 4,

    num_beams: 1,
    eval_decode_greedy: true,
    min_edu_length: 1,
    // Decode B docs in one shared forward instead of one-at-a-time; length-bucketed,
    // bit-identical to the serial path (equivalence-gated). What makes per-epoch
    // full-dev validation affordable. Greedy only (beam fills the batch dim itself).
    batched_decode: true,
}
