// icl-c: self-hosted, grammar-constrained in-context RST parsing on RST-DT.
// Qwen3.5-397B-A17B (UD-Q4_K_XL GGUF) served by llama.cpp on port 8083, with
// every attention/Gated-DeltaNet layer GPU-resident and only routed-expert FFN
// weights spilled to host RAM (see joblogs/serve_qwen397b.sh for why that split
// is mandatory rather than a tuning choice).
//
// Same pipeline as the 122B config; the pair is the intended size ladder that
// tests whether icl-c is capability-bound or segmentation-bound.
(import 'icl_rstdt.jsonnet') + {
    provider: {
        type: 'openai_compatible',
        model: 'openai/local',        // llama.cpp ignores the name; litellm needs the openai/ route
        // Sized for THINKING, not for the answer. The answer is document-length
        // (stage 1 re-emits the document with EDU delimiters; RST-DT's longest
        // test doc is 2888 subwords), but the think block dwarfs it: measured
        // 9.3k-11.7k generated tokens to segment 45- and 58-word documents.
        // A think block that exhausts this cap is a HARD failure for the
        // document, because the grammar only clamps on at </think> and the
        // reasoning-budget escape hatch is unusable with a lazy grammar.
        // Headroom check: per-slot context is 65536 (-c 262144 over -np 4) and
        // the worst-case prompt is ~8.1k, so this leaves ~8k spare.
        max_tokens: 49152,
        // Thinking-on generations run for TENS OF MINUTES, so the default 600s
        // timeout fires mid-generation: measured 32425 tokens on wsj_1146's parse
        // stage, and the client gave up at 10 min and retried work the server had
        // already finished (worse, llama.cpp does NOT cancel the orphaned task, so
        // it keeps burning a slot). Sizing: with 8 busy slots the per-slot rate is
        // ~8.1 tok/s, so a 30k-token generation takes ~62 min. Two hours covers it.
        request_timeout: 7200,
        thinking: false,
        effort: 'high',               // unused for openai_compatible
        base_url: 'http://localhost:8083/v1',
        api_key_env: null,
        api_key: 'foo',               // local dummy key
        // Thinking stays ON: with grammar.lazy the model reasons freely and the
        // grammar only clamps on at </think>, so icl-c differs from the frontier
        // arm in CONSTRAINT alone rather than in constraint plus reasoning.
        disable_thinking: false,
    },

    // The constrained arm: exact GBNF grammar decoding, which is the whole point
    // of icl-c versus the unconstrained frontier arm. `lazy` routes the request
    // to llama.cpp's native endpoint, since /v1/chat/completions silently
    // downgrades a lazy grammar to an eager one.
    grammar: { enabled: true, backend: 'gbnf', lazy: true, think_end: '</think>' },
}
