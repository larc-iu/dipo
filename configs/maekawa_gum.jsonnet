// Maekawa et al. (2024) LLM parser on GUM 12.1 (fine relations), from raw text: one LoRA
// adapter answers each node's split, nuclearity and relation top-down, and a second adapter
// on the same Qwen 3.8 27B base segments EDUs (see dipo/rst/parsers/maekawa/README.md).
// Every field below is the default; their own setup is noted in MaekawaConfig's docstring.
{
    train_dir: 'data/gum_12.1.0_notok/train',
    dev_dir: 'data/gum_12.1.0_notok/dev',
    test_dir: 'data/gum_12.1.0_notok/test',
    relation_map: null,                                    // GUM's native fine set

    model_name: 'Qwen/Qwen3.8-27B',
    parse_type: 'top_down',                                // or 'bottom_up'
    label_mode: 'unified',                                 // or 'separate' (theirs), 'joint'
    peft: { r: 64, alpha: 16, dropout: 0.1, target_modules: 'all-linear', load_in_4bit: false },
    gradient_checkpointing: true,
    segmentation: { epochs: 5, lr: 2e-4, docs_per_step: 4 },  // null = gold EDUs only

    lr: 2e-4,
    epochs: 5,
    batch_size: 16,
    grad_accum: 1,
    max_microbatch_tokens: 4096,
    max_target_tokens: 24,

    checkpoint_dir: 'checkpoints',
    run_name: null,
}
