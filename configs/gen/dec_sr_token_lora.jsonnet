// gen decoder_only x sr, TOKEN mode (relation labels as reserved tokens over a
// small action head), LoRA (google/gemma-3-1b-it). SR reverts to token mode
// (Luke 2026-07-18): simpler, sidesteps the words-mode untied-lm_head machinery.
// sexp stays words mode (Hu & Wan-style natural s-expressions).
local base = import '_rstdt_words_base.libsonnet';
base + {
    backbone: 'decoder_only',
    serialization: 'sr',
    label_style: 'token',
    model_name: 'google/gemma-3-1b-it',
    max_output_length: 16384,
    peft: { r: 16, alpha: 32, dropout: 0.10, target_modules: 'all-linear', bias: 'none', dora: false },
    lr: 3e-4,
    run_name: 'gen-dec-sr-token-lora',
}
