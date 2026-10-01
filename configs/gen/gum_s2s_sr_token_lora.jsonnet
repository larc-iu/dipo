// GUM 12.1.0: gen seq2seq (T5Gemma) x sr, TOKEN mode (paper's SR arm), LoRA. <=4B, quartz.
(import 's2s_sr_token_lora.jsonnet') + (import '_gum_override.libsonnet') + {
    run_name: 'gum-gen-s2s-sr-token-lora',
}
