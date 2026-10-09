"""Model-card rendering for Hub pushes: the metrics table, the tag, and the usage example."""

import os

from dipo.rst.parsers.hfhub.hub import CARD_LOGO, render_model_card

CONFIG = {
    "model_name": "xlm-roberta-base",
    "train_dir": "data/gcdt_notok/train",
    "val_metric_name": "e2e_full_f1",
    "relation_types": [["elaboration", "rst"], ["joint", "multinuc"]],
    "relation_map": None,
    "segmentation": {"scheme": "BIE"},
}

# DMRST writes its gold-EDU scores without a prefix, the generative parsers with one.
DMRST_METRICS = {
    "dev": {"seg_f1": 0.85, "e2e_span_f1": 0.6, "e2e_full_f1": 0.338, "span_f1": 0.73, "full_f1": 0.47},
    "test": {"seg_f1": 0.864, "e2e_span_f1": 0.588, "e2e_full_f1": 0.2944, "span_f1": 0.71, "full_f1": 0.419},
}
GEN_METRICS = {
    "dev": {"seg_f1": 0.978, "e2e_full_f1": 0.529, "gold_edu_full_f1": 0.551},
    "test": {"seg_f1": 0.981, "e2e_full_f1": 0.547, "gold_edu_full_f1": 0.565},
    "decode": {"num_beams": 1},
}


def card(kind="dmrst", metrics=DMRST_METRICS, **kw):
    return render_model_card(
        parser_kind=kind,
        config=CONFIG,
        checkpoint_meta={},
        final_metrics=metrics,
        repo_id="larc-iu/example",
        **kw,
    )


def table_rows(text):
    return [line for line in text.split("\n") if line.startswith("| ")]


def test_metrics_table_is_curated_and_in_percent():
    rows = table_rows(card())
    assert rows[0] == "| Split | Seg | E2E Span | E2E Full | Gold-EDU Span | Gold-EDU Full |"
    assert rows[2] == "| dev | 85.0 | 60.0 | 33.8 | 73.0 | 47.0 |"
    assert rows[3] == "| test | 86.4 | 58.8 | 29.4 | 71.0 | 41.9 |"


def test_gold_edu_scores_are_found_under_either_key_spelling():
    rows = table_rows(card("gen", GEN_METRICS))
    assert rows[0] == "| Split | Seg | E2E Full | Gold-EDU Full |"
    assert rows[3] == "| test | 98.1 | 54.7 | 56.5 |"


def test_a_non_split_key_does_not_become_a_row():
    text = card("gen", GEN_METRICS)
    assert [r.split(" |")[0] for r in table_rows(text)[2:]] == ["| dev", "| test"]
    assert "decode" not in text.split("### Metrics")[1].split("## Usage")[0]


def test_unrecognized_metrics_are_said_so_rather_than_dumped():
    assert "unrecognized format" in card(metrics={"dev": {"weird": 1}, "test": {"weird": 2}})


def test_dipo_and_legacy_iudex_tags_and_language_are_in_the_front_matter():
    front = card(language="zh").split("---")[1]
    assert "  - dipo\n" in front
    assert "  - iudex\n" in front  # the old name stays searchable for a while
    assert "library_name: dipo\n" in front
    assert "language:\n  - zh\n" in front
    assert "language:" not in card().split("---")[1]


def test_usage_says_how_to_install():
    assert "pip install dipo" in card().split("## Usage")[1]


def test_usage_examples_use_the_given_text():
    text = card(example_text="我们在这里工作。")
    assert "我们在这里工作。" in text.split("### CLI")[1].split("###")[0]
    assert 'parser.predict_from_text(\n    "我们在这里工作。"\n)' in text
    assert "carefully designed" in card()


def test_logo_sits_between_the_front_matter_and_the_title():
    after_front = card().split("---\n\n", 1)[1]
    assert after_front.startswith(CARD_LOGO)
    assert after_front[len(CARD_LOGO) :].startswith("# larc-iu/example\n")


def test_logo_url_points_at_a_file_in_this_repo():
    path = CARD_LOGO.split("/master/", 1)[1].split('"', 1)[0]
    assert os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path))
