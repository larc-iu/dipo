// gen seq2seq x sexp, words mode, LoRA (google/t5gemma-2-1b-1b). <=4B validation grid.
local base = import '_rstdt_words_base.libsonnet';
base + {
    backbone: 'seq2seq',
    serialization: 'sexp',
    model_name: 'google/t5gemma-2-1b-1b',
    max_output_length: 16384,
    traversal_order: 'postorder',
    use_copy: true,
    peft: { r: 16, alpha: 32, dropout: 0.10, target_modules: 'all-linear', bias: 'none', dora: false },
    lr: 3e-4,
    run_name: 'gen-s2s-sexp-words-lora-pev',
    batched_decode: false,  // sexp: unbatched validation (batched cascade biases down)
}
