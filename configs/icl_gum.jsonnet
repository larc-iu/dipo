// Frontier in-context-learning RST parser on GUM 12.1 (fine relations).
// Two-call pipeline (segment, then parse the EDUs), shift-reduce serialization,
// Claude Opus 4.8. No training: this config drives prompting + reconstruction.
{
    // Data. train_dir sources the k worked in-context examples; dev/test are the
    // eval splits (choose with `dipo icl eval --split ...`).
    train_dir: 'data/gum_12.1.0_notok/train',
    dev_dir: 'data/gum_12.1.0_notok/dev',
    test_dir: 'data/gum_12.1.0_notok/test',
    relation_types: null,                                 // inferred at load time (union over train+dev)
    relation_map: null,                                   // GUM uses its native fine relation set

    // Provider. Frontier Anthropic via litellm. Auth: leave api_key_env/api_key
    // null and litellm reads ANTHROPIC_API_KEY from the environment.
    provider: {
        type: 'anthropic',
        model: 'anthropic/claude-opus-4-8',
        max_tokens: 120000,
        thinking: true,
        effort: 'high',                                   // low | medium | high | max
        base_url: null,
        api_key_env: null,
        api_key: null,
    },

    serialization: 'sr',                                  // 'sr' | 'sexp' (SR >> sexp on the trained parsers)
    pipeline_mode: 'two_call',                            // 'e2e' | 'two_call' | 'gold_edu'

    // Grammar-constrained decoding is the self-hosted icl-c arm (unavailable on
    // Anthropic), so it stays disabled here. See configs/icl_local.jsonnet.
    grammar: { enabled: false, backend: 'gbnf' },

    // In-context examples.
    k: 5,
    seed: 42,
    example_max_edus: null,                               // cap example doc size (null = any)
}
