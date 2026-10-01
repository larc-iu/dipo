// gen seq2seq x sr, TOKEN mode, LoRA (google/t5gemma-2-1b-1b). SR token-mode
// redo (Luke 2026-07-18). sexp stays words mode (Hu & Wan-style).
local base = import '_rstdt_words_base.libsonnet';
base + {
    backbone: 'seq2seq',
    serialization: 'sr',
    label_style: 'token',
    model_name: 'google/t5gemma-2-1b-1b',
    max_output_length: 16384,
    peft: { r: 16, alpha: 32, dropout: 0.10, target_modules: 'all-linear', bias: 'none', dora: false },
    lr: 3e-4,
    run_name: 'gen-s2s-sr-token-lora',
}
