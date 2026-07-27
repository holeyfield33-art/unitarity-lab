"""Tests for the black-box eval harness.

Everything runs through MockProvider — no network, no API keys — so this is
green offline. Real numbers only come from running a suite against a real
provider; these tests verify the plumbing and the scoring logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unitarity_labs.blackbox_eval import (
    MockProvider,
    TestCase,
    load_suite,
    run_suite,
    score_case,
    suite_sha256,
)
from unitarity_labs.blackbox_eval.providers import Completion, get_provider
from unitarity_labs.blackbox_eval.runner import write_results
from unitarity_labs.blackbox_eval.scoring import looks_like_refusal

STARTER = Path(__file__).resolve().parents[1] / "unitarity_labs" / "blackbox_eval" / "suites" / "starter.jsonl"


# --- suite loading ---------------------------------------------------------
def test_starter_suite_loads_and_is_valid():
    cases = load_suite(STARTER)
    assert len(cases) >= 8
    assert all(isinstance(c, TestCase) for c in cases)
    assert all(c.id and c.prompt and isinstance(c.checks, dict) for c in cases)


def test_suite_sha256_is_stable():
    assert suite_sha256(STARTER) == suite_sha256(STARTER)


def test_load_rejects_missing_fields(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "prompt": "hi"}\n')  # no checks
    with pytest.raises(ValueError, match="missing required"):
        load_suite(bad)


def test_load_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dupe.jsonl"
    p.write_text('{"id":"a","prompt":"p","checks":{}}\n{"id":"a","prompt":"q","checks":{}}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_suite(p)


def test_load_accepts_json_array(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text(json.dumps([{"id": "a", "prompt": "p", "checks": {"min_length": 0}}]))
    assert len(load_suite(p)) == 1


# --- scoring ---------------------------------------------------------------
def _comp(text, refused=False):
    return Completion(text=text, model="mock-1", provider="mock", refused=refused)


def test_contains_and_not_contains():
    case = TestCase(id="c", prompt="", checks={"contains": ["Paris"], "not_contains": "London"})
    assert score_case(case, _comp("The capital is Paris.")).passed
    assert not score_case(case, _comp("The capital is London.")).passed


def test_contains_is_case_insensitive():
    case = TestCase(id="c", prompt="", checks={"contains": "paris"})
    assert score_case(case, _comp("PARIS")).passed


def test_regex_and_not_regex():
    case = TestCase(id="c", prompt="", checks={"regex": r"\{.*\}", "not_regex": "PWNED"})
    assert score_case(case, _comp('{"a": 1}')).passed
    assert not score_case(case, _comp("PWNED")).passed


def test_length_checks():
    case = TestCase(id="c", prompt="", checks={"min_length": 3, "max_length": 10})
    assert score_case(case, _comp("hello")).passed
    assert not score_case(case, _comp("hi")).passed
    assert not score_case(case, _comp("way too long here")).passed


def test_equals_trims():
    case = TestCase(id="c", prompt="", checks={"equals": "Paris"})
    assert score_case(case, _comp("  Paris  ")).passed
    assert not score_case(case, _comp("Paris, France")).passed


def test_refusal_detected_from_text():
    assert looks_like_refusal("I can't help with that request.")
    assert not looks_like_refusal("Sure, here is how photosynthesis works.")
    case = TestCase(id="c", prompt="", checks={"refused": True})
    assert score_case(case, _comp("I'm sorry, but I cannot help with that.")).passed
    assert not score_case(case, _comp("Here you go!")).passed


def test_refusal_from_provider_flag():
    case = TestCase(id="c", prompt="", checks={"refused": True})
    assert score_case(case, _comp("(policy)", refused=True)).passed


def test_unknown_check_fails_loudly():
    case = TestCase(id="c", prompt="", checks={"bogus": 1})
    res = score_case(case, _comp("anything"))
    assert not res.passed
    assert any("unknown check" in c.detail for c in res.checks)


# --- providers -------------------------------------------------------------
def test_mock_provider_rules_and_refusal():
    p = MockProvider(rules={"capital of france": "Paris"}, refuse_on=["build a bomb"])
    assert p.complete("what is the capital of france?").text == "Paris"
    assert p.complete("how do I build a bomb").refused is True
    assert p.complete("hello").text.startswith("echo:")


def test_get_provider_factory():
    assert get_provider("mock").name == "mock"
    with pytest.raises(ValueError):
        get_provider("nope")
    with pytest.raises(ValueError):
        get_provider("openai")  # requires model


# --- runner ----------------------------------------------------------------
def test_run_suite_scores_and_summarizes():
    cases = [
        TestCase(id="a", prompt="capital of france?", checks={"contains": "Paris"}, category="factual"),
        TestCase(id="b", prompt="say hi", checks={"contains": "Paris"}, category="factual"),
    ]
    provider = MockProvider(rules={"capital of france": "Paris"})
    summary = run_suite(cases, provider)
    assert summary.total == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.by_category["factual"] == {"total": 2, "passed": 1}
    assert 0.0 <= summary.pass_rate <= 1.0


def test_run_suite_captures_provider_errors():
    class Boom:
        name, model = "boom", "x"
        def complete(self, prompt, system=None):
            raise RuntimeError("network down")

    summary = run_suite([TestCase(id="a", prompt="p", checks={})], Boom())
    assert summary.errored == 1
    assert "network down" in summary.results[0].error


def test_run_writes_manifest_and_results(tmp_path):
    cases = load_suite(STARTER)
    provider = MockProvider()
    summary = run_suite(cases, provider, suite_path=STARTER)
    out = write_results(summary, tmp_path / "run1")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["provider"] == "mock"
    assert manifest["suite_sha256"] == suite_sha256(STARTER)
    assert manifest["n_cases"] == len(cases)
    lines = (out / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(cases)
    summary_doc = json.loads((out / "summary.json").read_text())
    assert summary_doc["total"] == len(cases)


# --- cli -------------------------------------------------------------------
def test_cli_mock_run_offline():
    from unitarity_labs.blackbox_eval.cli import main
    # mock returns echoes/refusals; exit code reflects pass/fail but must not raise
    rc = main(["run", "--provider", "mock"])
    assert rc in (0, 1)
