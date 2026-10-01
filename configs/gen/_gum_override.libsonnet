// Shared GUM 12.1.0 override for the gen family: point train/dev/test at the
// GUM notok tree and drop the RST-DT coarse relation map so GUM's native FINE
// inventory is used (relation_types stays null => inferred at train time, ~32
// relations). Layer this onto any RST-DT gen config, then set a gum-* run_name:
//   (import 'dec_sr_words_lora.jsonnet') + (import '_gum_override.libsonnet')
//     + { run_name: 'gum-gen-dec-sr-words-lora' }
{
    train_dir: 'data/gum_12.1.0_notok/train',
    dev_dir: 'data/gum_12.1.0_notok/dev',
    test_dir: 'data/gum_12.1.0_notok/test',
    relation_map: null,
}
