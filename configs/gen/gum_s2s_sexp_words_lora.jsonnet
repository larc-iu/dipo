// GUM 12.1.0: gen seq2seq (T5Gemma) x sexp, words mode, LoRA. <=4B.
(import 's2s_sexp_words_lora.jsonnet') + (import '_gum_override.libsonnet') + {
    run_name: 'gum-gen-s2s-sexp-words-lora',
}
