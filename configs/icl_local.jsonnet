// Local smoke-test config: self-hosted OpenAI-compatible endpoint (llama.cpp /
// vLLM) instead of the frontier API. Used to exercise the full pipeline against
// a small model without spending frontier tokens. NOT a paper config.
{
    train_dir: 'data/gum_12.1.0_notok/train',
    dev_dir: 'data/gum_12.1.0_notok/dev',
    test_dir: 'data/gum_12.1.0_notok/test',
    relation_types: null,
    relation_map: null,

    provider: {
        type: 'openai_compatible',
        model: 'openai/local',                            // llama.cpp ignores the name; litellm needs the openai/ route
        max_tokens: 8192,
        thinking: false,
        effort: 'high',                                   // unused for openai_compatible
        base_url: 'http://localhost:8081/v1',
        api_key_env: null,
        api_key: 'foo',                                   // local dummy key
    },

    serialization: 'sr',
    pipeline_mode: 'two_call',

    grammar: { enabled: false, backend: 'gbnf' },

    // Small k + shortish examples keep the prompt within a small model's context.
    k: 2,
    seed: 42,
    example_max_edus: 50,
}
