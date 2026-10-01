// Frontier in-context-learning RST parser on RST-DT (coarse relations).
// Two-call pipeline, shift-reduce serialization, Claude Opus 4.8.
{
    train_dir: 'data/rstdt/train',
    dev_dir: 'data/rstdt/dev',
    test_dir: 'data/rstdt/test',
    relation_types: null,                                 // inferred at load time (union over train+dev)
    relation_map: import 'lib/rstdt_coarse_map.libsonnet', // fine -> 18 coarse classes

    provider: {
        type: 'anthropic',
        model: 'anthropic/claude-opus-4-8',
        max_tokens: 120000,
        thinking: true,
        effort: 'high',
        base_url: null,
        api_key_env: null,
        api_key: null,
    },

    serialization: 'sr',
    pipeline_mode: 'two_call',

    grammar: { enabled: false, backend: 'gbnf' },

    // RST-DT docs are long; keep in-context examples small so the cached prefix
    // is manageable.
    k: 5,
    seed: 42,
    example_max_edus: 60,
}
