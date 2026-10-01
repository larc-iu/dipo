// gen decoder_only x sexp, words mode, LoRA (google/gemma-3-1b-it). <=4B validation grid.
local base = import '_rstdt_words_base.libsonnet';
base + {
    backbone: 'decoder_only',
    serialization: 'sexp',
    model_name: 'google/gemma-3-1b-it',
    max_output_length: 16384,
    traversal_order: 'postorder',
    use_copy: true,
    peft: { r: 16, alpha: 32, dropout: 0.10, target_modules: 'all-linear', bias: 'none', dora: false },
    lr: 3e-4,
    run_name: 'gen-dec-sexp-words-lora-pev',
    batched_decode: false,  // sexp: unbatched validation (batched cascade biases down)
}
