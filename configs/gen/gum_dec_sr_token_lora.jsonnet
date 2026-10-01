// GUM 12.1.0: gen decoder_only x sr, TOKEN mode (reserved action tokens over a
// small head, the paper's SR arm), LoRA (gemma-3-1b-it). <=4B, quartz.
(import 'dec_sr_token_lora.jsonnet') + (import '_gum_override.libsonnet') + {
    run_name: 'gum-gen-dec-sr-token-lora',
}
