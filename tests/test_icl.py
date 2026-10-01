"""Tests for the in-context-learning parser (`iudex.rst.parsers.icl`).

Covers the pieces that are only checked by hand otherwise: the string codecs
(words-mode SR + sexp, both shapes, N=1), the segmentation codec, the GBNF
grammar builders (exact-count / deterministic / word-labels), the whitespace-axis
seg_data + failure-as-zero metric accounting, and config validation.

Reads GUM if present (structural equality via `RstTree.__eq__`), skips otherwise.
"""

import math
from pathlib import Path

import pytest

from iudex.rst.data.reader import infer_relation_types, read_rst_dir
from iudex.rst.data.seg_metrics import evaluate_seg_and_e2e
from iudex.rst.data.tree import RstTree, Shift
from iudex.rst.parsers.common.seqgen import relation_to_words
from iudex.rst.parsers.icl import serialize as S
from iudex.rst.parsers.icl.configuration_icl import GrammarConfig, IclConfig, ProviderConfig
from iudex.rst.parsers.icl.eval_icl import _gold_edu_metrics
from iudex.rst.parsers.icl.eval_metrics import failed_seg_data, seg_data_for_doc
from iudex.rst.parsers.icl.modeling_icl import IclParser

REPO_ROOT = Path(__file__).resolve().parents[1]
GUM_DEV = REPO_ROOT / "data" / "gum_12.1.0_notok" / "dev"


@pytest.fixture(scope="module")
def gum():
    if not GUM_DEV.is_dir():
        pytest.skip("No GUM data under data/gum_12.1.0_notok/dev.")
    rels = infer_relation_types([str(GUM_DEV)])
    pairs = sorted(read_rst_dir(str(GUM_DEV), relation_types=rels), key=lambda p: len(p[1].edu_strings))
    return rels, [t for _, t in pairs[:5]]


# ---- codec round-trips ----


@pytest.mark.parametrize("name", ["sr", "sexp"])
def test_e2e_roundtrip(gum, name):
    rels, trees = gum
    ser = S.get_serialization(name)
    for t in trees:
        assert ser.parse_e2e(ser.render_e2e(t), rels) == t


@pytest.mark.parametrize("name", ["sr", "sexp"])
def test_parse_over_edus_roundtrip(gum, name):
    rels, trees = gum
    ser = S.get_serialization(name)
    for t in trees:
        assert ser.parse_over_edus(ser.render_parse(t), list(t.edu_strings), rels) == t


@pytest.mark.parametrize("name", ["sr", "sexp"])
def test_single_edu_roundtrip(gum, name):
    rels, _ = gum
    one = RstTree.from_shift_reduce([Shift()], edus=["a single edu"], relation_types=rels)
    ser = S.get_serialization(name)
    assert ser.parse_over_edus(ser.render_parse(one), ["a single edu"], rels).edu_strings == ["a single edu"]


def test_segmentation_roundtrip(gum):
    _, trees = gum
    for t in trees:
        edus = list(t.edu_strings)
        assert S.parse_segmentation(S.render_segmentation(edus)) == edus


# ---- words-mode label matches gen ----


def test_word_label_matches_gen_spelling():
    assert S.word_label("NS", "elaboration-additional") == "NS elaboration additional <label_end>"
    assert S.word_label("NN", "joint-list") == f"NN {relation_to_words('joint-list')} <label_end>"


# ---- GBNF grammars ----


def _enumerate(grammar: str):
    rules = {}
    for line in grammar.splitlines():
        if "::=" not in line:
            continue
        head, body = line.split("::=", 1)
        head = head.strip()
        if head in ("shift", "reduce"):
            continue
        rules[head] = [a.strip() for a in body.split("|")]
    out = []

    def walk(state, acc):
        for alt in rules[state]:
            toks = alt.split()
            if alt == '""':
                out.append(tuple(acc))
            else:
                walk(toks[1], acc + [toks[0]])

    walk(rules["root"][0].strip(), [])
    return out


def test_sr_grammar_is_exactly_catalan(gum):
    rels, _ = gum
    for n in range(1, 7):
        strings = _enumerate(S.sr_parse_grammar(n, rels))
        catalan = 1 if n == 1 else math.comb(2 * (n - 1), n - 1) // n
        assert len(strings) == catalan
        for s in strings:  # every string is a valid n-shift / (n-1)-reduce sequence
            depth = 0
            for a in s:
                depth += 1 if a == "shift" else -1
                assert depth >= 1  # stack never underflows
            assert depth == 1 and s.count("shift") == n


def test_sr_grammar_uses_word_labels(gum):
    rels, _ = gum
    reduce_line = next(line for line in S.sr_parse_grammar(3, rels).splitlines() if line.startswith("reduce"))
    assert "<label_end>" in reduce_line and "<reduce_" not in reduce_line


def test_sr_grammar_fallback_above_cap(gum):
    rels, _ = gum
    g = S.sr_parse_grammar(10, rels, max_exact=3)
    assert "tree ::= shift rest" in g


def test_segmentation_grammar_shape():
    g = S.segmentation_grammar("the cat sat")
    assert g.startswith('root ::= "the" gap "cat" gap "sat"')
    assert 'gap ::= " " | " || "' in g


# ---- seg_data / metric accounting ----


def test_seg_data_perfect_prediction(gum):
    _, trees = gum
    golds = trees
    sd = [seg_data_for_doc(list(t.edu_strings), t) for t in golds]
    m = evaluate_seg_and_e2e(golds, sd)
    assert m["seg_f1"] == pytest.approx(1.0) and m["e2e_full_f1"] == pytest.approx(1.0)


def test_failure_counts_as_zero_not_dropped(gum):
    _, trees = gum
    g = trees[0]
    perfect = evaluate_seg_and_e2e([g], [seg_data_for_doc(list(g.edu_strings), g)])
    with_failure = evaluate_seg_and_e2e([g, g], [seg_data_for_doc(list(g.edu_strings), g), failed_seg_data(list(g.edu_strings))])
    # the failed doc stays in the recall denominator, so recall drops
    assert with_failure["e2e_full_r"] < perfect["e2e_full_r"]


def test_gold_edu_metrics_penalize_failures(gum):
    _, trees = gum
    g = trees[0]
    assert _gold_edu_metrics([(g, g)], [])["full_f1"] == pytest.approx(1.0)
    assert _gold_edu_metrics([(g, g)], [g])["full_f1"] < 1.0


# ---- config validation ----


def _cfg(**kw):
    base = dict(train_dir="a", dev_dir="b")
    base.update(kw)
    return IclConfig(**base)


def test_grammar_requires_openai_sr_nonee2e():
    oc = dict(provider=ProviderConfig(type="openai_compatible", base_url="x"), grammar=GrammarConfig(enabled=True))
    _cfg(**oc)  # sr + two_call default: ok
    with pytest.raises(ValueError):
        _cfg(provider=ProviderConfig(type="anthropic"), grammar=GrammarConfig(enabled=True))
    with pytest.raises(ValueError):
        _cfg(serialization="sexp", **oc)
    with pytest.raises(ValueError):
        _cfg(pipeline_mode="e2e", **oc)


def test_grammar_backend_validated():
    with pytest.raises(ValueError):
        GrammarConfig(backend="bogus")


# ---- lazy (think-then-constrain) grammar ----


def test_prefixed_root_preserves_language(gum):
    """Re-rooting must add the trigger literal and change nothing else: llama.cpp
    feeds the trigger text into the grammar, so the prefix has to be part of the
    language, but the tail must still be exactly the old one."""
    rels, _ = gum
    for n in range(1, 6):
        wrapped = S.prefixed_root_grammar(S.sr_parse_grammar(n, rels), "</think>")
        assert wrapped.splitlines()[0] == 'root ::= "</think>" answer-root'
        # _enumerate walks from a bare state name, so re-point the tail at `root`
        # to compare the language behind the trigger against the unwrapped one.
        tail = "\n".join(wrapped.splitlines()[1:]).replace("answer-root", "root", 1)
        assert _enumerate(tail) == _enumerate(S.sr_parse_grammar(n, rels))


def test_prefixed_root_rejects_recursive_root():
    with pytest.raises(ValueError):
        S.prefixed_root_grammar('root ::= "a" x\nx ::= root', "</think>")
    with pytest.raises(ValueError):
        S.prefixed_root_grammar('start ::= "a"', "</think>")


def test_lazy_grammar_refuses_thinking_disabled():
    """The silent-failure guard: thinking-off never emits the trigger, so a lazy
    grammar would never engage and the run would be unconstrained while still
    calling itself icl-c."""
    oc = dict(provider=ProviderConfig(type="openai_compatible", base_url="x", disable_thinking=False))
    _cfg(grammar=GrammarConfig(enabled=True, lazy=True), **oc)  # thinking on: ok
    with pytest.raises(ValueError):
        _cfg(
            provider=ProviderConfig(type="openai_compatible", base_url="x", disable_thinking=True),
            grammar=GrammarConfig(enabled=True, lazy=True),
        )


def test_lazy_grammar_needs_gbnf_and_trigger():
    with pytest.raises(ValueError):
        GrammarConfig(enabled=True, lazy=True, backend="guidance")
    with pytest.raises(ValueError):
        GrammarConfig(enabled=True, lazy=True, think_end="")


def test_lazy_grammar_missing_trigger_is_a_hard_failure(gum):
    """A response with no think-end tag was generated UNCONSTRAINED (the tag is
    what triggers the grammar), so it must fail loudly rather than be handed to
    the SR parser as if it were an answer."""
    rels, trees = gum
    cfg = _cfg(
        provider=ProviderConfig(type="openai_compatible", base_url="x", disable_thinking=False),
        grammar=GrammarConfig(enabled=True, lazy=True),
        relation_types=rels,
    )
    parser = IclParser.__new__(IclParser)
    parser.config = cfg
    parser.grammar_enabled = True
    with pytest.raises(ValueError, match="never triggered"):
        parser._strip_think_end("reasoning that rambles <shift> <shift> and never closes")
    # the normal case still splits on the tag
    assert parser._strip_think_end("thinking...</think>  <shift> ") == "<shift> "


# ---- echo_edu_text: re-emit EDU content in the parse stage ----


@pytest.mark.parametrize("name", ["sr", "sexp"])
def test_echo_parse_roundtrip(gum, name):
    """The echoed form must invert exactly, and must equal the e2e rendering --
    that convergence is the point: the parse stage stops being a content-free
    variant, so one serialization covers both stages (and gen's words-mode)."""
    rels, trees = gum
    ser = S.get_serialization(name)
    for t in trees:
        rendered = ser.render_parse(t, echo_edu_text=True)
        assert rendered == ser.render_e2e(t)
        assert ser.parse_over_edus(rendered, list(t.edu_strings), rels, echo_edu_text=True) == t


@pytest.mark.parametrize("name", ["sr", "sexp"])
def test_echo_mismatched_text_is_rejected(gum, name):
    """Echoed text that disagrees with the given EDUs means the response was not
    actually constrained; scoring a tree over the wrong units would be silent."""
    rels, trees = gum
    ser = S.get_serialization(name)
    t = trees[-1]
    rendered = ser.render_parse(t, echo_edu_text=True)
    wrong = list(t.edu_strings)
    wrong[0] = "totally different words here"
    with pytest.raises(ValueError):
        ser.parse_over_edus(rendered, wrong, rels, echo_edu_text=True)


def test_echo_grammar_forces_edu_text(gum):
    """The echo grammar must force the EDU text without adding any freedom: same
    language size as the bare-shift grammar (Catalan), but each shift carries the
    EDU it consumes."""
    rels, _ = gum
    edus = ["alpha one", "beta two", "gamma three", "delta four"]
    g = S.sr_parse_grammar(4, rels, edus=edus)
    # one rule per EDU, not one per (s, d) state
    assert sum(1 for line in g.splitlines() if line.startswith("shift-")) == 4
    assert '"alpha one <shift> "' in g and '"delta four <shift> "' in g
    strings = _enumerate(g)
    assert len(strings) == math.comb(6, 3) // 4  # Catalan(3) = 5
    for s in strings:
        assert sum(1 for a in s if a.startswith("shift-")) == 4


def test_echo_grammar_size_stays_close_to_bare(gum):
    """The per-EDU indirection is what keeps this affordable; inlining the text at
    every (s, d) state would roughly double the grammar."""
    rels, _ = gum
    edus = [f"edu number {i} with some words in it" for i in range(60)]
    bare = S.sr_parse_grammar(60, rels)
    echo = S.sr_parse_grammar(60, rels, edus=edus)
    assert len(echo) < 1.35 * len(bare)


def test_prefixed_root_ignores_root_inside_literals(gum):
    """With echo_edu_text the right-hand sides carry arbitrary document text, so
    the re-root guard must look at rule REFERENCES only. RST-DT's wsj_1387 has an
    EDU reading "since it took root as cheap entertainment ...", which previously
    tripped the guard and failed the whole document."""
    rels, _ = gum
    edus = ["since it took root as cheap entertainment", "and then it grew"]
    g = S.sr_parse_grammar(2, rels, edus=edus)
    wrapped = S.prefixed_root_grammar(g, "</think>")  # must not raise
    assert wrapped.splitlines()[0] == 'root ::= "</think>" answer-root'
    # a genuine reference is still caught
    with pytest.raises(ValueError):
        S.prefixed_root_grammar('root ::= "a" x\nx ::= root "b"', "</think>")


def test_parse_input_format_per_serialization(gum):
    """The parse call gets ` || `-delimited EDUs, matching what stage 1 emits --
    except for sexp WITHOUT echo, whose leaves are bare integers and therefore
    genuinely need the numbering to refer to."""
    _, trees = gum
    edus = list(trees[0].edu_strings)
    sr, sx = S.get_serialization("sr"), S.get_serialization("sexp")
    for echo in (False, True):
        assert sr.parse_input(edus, echo_edu_text=echo) == S.delimited_edus(edus)
    assert sx.parse_input(edus, echo_edu_text=True) == S.delimited_edus(edus)
    assert sx.parse_input(edus, echo_edu_text=False) == S.numbered_edus(edus)
    # the numbered form is the one that carries indices; the delimited one does not
    assert "1." in S.numbered_edus(edus) and " || " in S.delimited_edus(edus)


def test_parse_input_roundtrips_through_segmentation_form(gum):
    """Because the parse input reuses stage 1's delimiter, stage 2 consumes exactly
    what stage 1 produces -- no reformatting step between the calls."""
    _, trees = gum
    edus = list(trees[0].edu_strings)
    assert S.get_serialization("sr").parse_input(edus) == S.render_segmentation(edus)
    assert S.parse_segmentation(S.get_serialization("sr").parse_input(edus)) == edus


def test_think_budget_config_guards():
    """think_budget only means something behind a lazy grammar, and 0 would mean
    thinking-off rather than budgeted thinking -- both are silent-wrong configs."""
    GrammarConfig(enabled=True, lazy=True, think_budget=49152)  # ok
    GrammarConfig(enabled=True, lazy=True, think_budget=None)  # unbounded: ok
    with pytest.raises(ValueError):  # eager grammar has no think block to budget
        GrammarConfig(enabled=True, lazy=False, think_budget=49152)
    with pytest.raises(ValueError):  # 0 == thinking off, say so explicitly instead
        GrammarConfig(enabled=True, lazy=True, think_budget=0)
    with pytest.raises(ValueError):
        GrammarConfig(enabled=True, lazy=True, think_budget=1024, budget_message="")


def _budget_parser(monkeypatch, **gkw):
    import iudex.rst.parsers.icl.modeling_icl as M

    monkeypatch.setattr(M.IclParser, "_sample_examples", lambda self: [])
    return M.IclParser(
        _cfg(
            provider=ProviderConfig(type="openai_compatible", base_url="http://x/v1"),
            grammar=GrammarConfig(enabled=True, lazy=True, **gkw),
            relation_types=[("elaboration", "mononuclear")],
        )
    )


def test_think_budget_payload_is_client_side(monkeypatch):
    """Budget forcing must NOT go out as llama.cpp's reasoning_budget_* fields:
    when those FORCE the think-end, the grammar is left untriggered and the answer
    comes back unconstrained (measured). The handoff is private and carries the
    BARE grammar for the second call."""
    payload = _budget_parser(monkeypatch, think_budget=49152)._grammar_payload('root ::= "a"')
    assert "reasoning_budget_tokens" not in payload
    assert "generation_prompt" not in payload
    f = payload["_budget_forcing"]
    assert (f["tokens"], f["think_end"], f["answer_grammar"]) == (49152, "</think>", 'root ::= "a"')
    # call 1 is still exactly today's lazy request
    assert payload["grammar_lazy"] is True
    assert payload["grammar_triggers"] == [{"type": 1, "value": "</think>"}]
    assert payload["grammar"].startswith('root ::= "</think>"')

    # Unbounded stays byte-identical to the pre-budget payload.
    assert "_budget_forcing" not in _budget_parser(monkeypatch)._grammar_payload('root ::= "a"')


def test_budget_forcing_second_call_shape(monkeypatch):
    """When call 1 runs out without a tag, call 2 must continue from call 1's text
    with the tag supplied by us and an EAGER grammar rooted at the answer."""
    p = _budget_parser(monkeypatch, think_budget=100)
    payload = p._grammar_payload('root ::= "a"')
    calls = []

    def fake_post(path, body, key):
        calls.append((path, body))
        if path == "/apply-template":
            return "PROMPT"
        return "still thinking" if len(calls) == 2 else "ANSWER"

    monkeypatch.setattr(p.provider, "_post", fake_post)
    out = p.provider._native_complete([{"role": "user", "content": "x"}], payload)

    first, second = calls[1][1], calls[2][1]
    assert first["n_predict"] == 100 and first["grammar_lazy"] is True
    assert second["prompt"] == "PROMPT" + "still thinking" + p.config.grammar.budget_message + "</think>"
    assert second["grammar"] == 'root ::= "a"'  # bare, not the </think>-prefixed one
    assert "grammar_lazy" not in second  # eager: the tag is already in the prompt
    assert out == "still thinking" + p.config.grammar.budget_message + "</think>ANSWER"
    assert p._strip_think_end(out) == "ANSWER" and p.forced_calls == 1


def test_budget_forcing_skips_second_call_on_natural_end(monkeypatch):
    """A model that finishes thinking inside its budget must cost exactly one call
    and be left completely alone."""
    p = _budget_parser(monkeypatch, think_budget=100)
    calls = []

    def fake_post(path, body, key):
        calls.append(path)
        return "PROMPT" if path == "/apply-template" else "done</think>ANSWER"

    monkeypatch.setattr(p.provider, "_post", fake_post)
    out = p.provider._native_complete([{"role": "user", "content": "x"}], p._grammar_payload('root ::= "a"'))
    assert calls == ["/apply-template", "/completion"]
    assert out == "done</think>ANSWER"
    assert p._strip_think_end(out) == "ANSWER" and p.forced_calls == 0


def test_forced_calls_counted_only_when_budget_message_present(monkeypatch):
    """A forced call is a real parse from truncated reasoning, so it must be
    counted -- but only when OUR injected message is what preceded the tag."""
    import iudex.rst.parsers.icl.modeling_icl as M

    monkeypatch.setattr(M.IclParser, "_sample_examples", lambda self: [])
    g = GrammarConfig(enabled=True, lazy=True, think_budget=1024)
    p = M.IclParser(
        _cfg(
            provider=ProviderConfig(type="openai_compatible", base_url="x"),
            grammar=g,
            relation_types=[("elaboration", "mononuclear")],
        )
    )
    assert p._strip_think_end(f"thinking...{g.budget_message}</think>ANSWER") == "ANSWER"
    assert p.forced_calls == 1
    assert p._strip_think_end("thought it through</think>ANSWER") == "ANSWER"
    assert p.forced_calls == 1  # natural completion is not a forced call


def test_forced_attribution_is_per_document_and_thread_local(monkeypatch):
    """A bare count cannot say WHICH documents were truncated, which is the
    question that decides whether a bound rescued a runaway or cut off a document
    that would have finished. Attribution must reset per document and not leak
    between concurrent workers."""
    from concurrent.futures import ThreadPoolExecutor

    p = _budget_parser(monkeypatch, think_budget=100)
    msg = p.config.grammar.budget_message

    def one(forced: bool):
        p.begin_doc()
        p._strip_think_end(f"thinking{msg}</think>A" if forced else "thinking</think>A")
        return p.doc_stats()["forced"]

    assert one(True) == 1
    assert one(False) == 0  # begin_doc cleared the previous document's count
    assert p.forced_calls == 1  # ...while the global tally still accumulates

    # concurrent documents must not see each other's counters
    with ThreadPoolExecutor(max_workers=4) as ex:
        got = list(ex.map(one, [True, False, True, False] * 4))
    assert got == [1, 0, 1, 0] * 4


def test_doc_stats_reports_generated_token_counts(monkeypatch):
    """Per-call token counts are what let a run show how long reasoning actually
    ran, rather than only whether it finished."""
    p = _budget_parser(monkeypatch, think_budget=100)
    p.begin_doc()
    assert p.doc_stats()["call_tokens"] == []
    p.provider._tls.tokens.append(1234)
    assert p.doc_stats()["call_tokens"] == [1234]
    p.begin_doc()
    assert p.doc_stats()["call_tokens"] == []
