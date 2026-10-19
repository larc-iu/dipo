// Maekawa et al. (2024) LLM parser on RST-DT with their 18 coarse relations, from the
// original untokenized LDC text: the configs/maekawa_gum.jsonnet setup on RST-DT. The
// relation names and their order in prompts are theirs.
(import 'maekawa_gum.jsonnet') + {
    train_dir: 'data/rstdt_notok/train',
    dev_dir: 'data/rstdt_notok/dev',
    test_dir: 'data/rstdt_notok/test',
    relation_map: import 'lib/rstdt_coarse_map.libsonnet',
    relation_labels: [
        'elaboration', 'attribution', 'joint', 'same-unit', 'contrast', 'explanation',
        'background', 'cause', 'enablement', 'evaluation', 'temporal', 'condition',
        'comparison', 'topic-change', 'summary', 'manner-means', 'textual-organization',
        'topic-comment',
    ],
}
