// GUM 12.1.0: gen decoder_only x sexp, words mode, LoRA (gemma-3-1b-it). <=4B.
(import 'dec_sexp_words_lora.jsonnet') + (import '_gum_override.libsonnet') + {
    run_name: 'gum-gen-dec-sexp-words-lora',
}
