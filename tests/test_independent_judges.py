"""Offline checks for blinding, evidence validation and resumable judge runs."""
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.run_independent_judges import (
    DEFAULT_MODELS, captured_evidence, checkpoint_path, control_cases, digest,
    evaluate_model, judge_case, main, prepare_cases, summarize, validate_judgment, write_json,
)


def judgment():
    return {"diagnosis": "HEALTHY", "grounding": "SUPPORTED", "completeness": "COMPLETE",
            "reason": "The fee matches the captured text.",
            "findings": [{"answer_quote": "The filing fee is 50 units.", "status": "SUPPORTED",
                          "evidence": [{"chunk_id": "fictional", "quote": "The filing fee is 50 units."}],
                          "reason": "Exact support."}]}


def benchmark_case():
    return {**control_cases()[0], "case_id": "eval-1", "kind": "benchmark",
            "comparison": {"trace_id": "trace-1", "xrag_primary_cause": "UNKNOWN"}}


def fixture_files(root):
    traces = root / "traces"
    traces.mkdir()
    trace = {"trace_id": "one", "question": "Fee?", "generated_answer": "50 units.",
             "prompt_snapshot": "--- Context chunk 1 [Chunk-ID: c1] ---\nFee is 50 units.\n\nQuestion: Fee?",
             "retrieved_chunk_references": [{"chunk_id": "c1", "text": "CHANGED REGISTRY TEXT"}],
             "diagnostics": {"primary_cause": "LEAKED_DIAGNOSIS"}}
    write_json(traces / "trace_one.json", trace)
    results = root / "results.json"
    write_json(results, [{"eval_id": "1", "trace_id": "one", "question": "Fee?",
                          "xrag_primary_cause": "UNSUPPORTED_GENERATION",
                          "expected_failure_type": "SPECULATIVE_LABEL"}])
    return results, traces


def test_preparation_blinds_judges_and_uses_original_prompt_text():
    with tempfile.TemporaryDirectory() as directory:
        results, traces = fixture_files(Path(directory))
        case = prepare_cases(results, traces)[-1]
    assert set(case["input"]) == {"question", "answer", "evidence"}
    assert case["input"]["evidence"][0]["text"] == "Fee is 50 units."
    assert "SPECULATIVE_LABEL" not in json.dumps(case)
    assert "LEAKED_DIAGNOSIS" not in json.dumps(case)
    assert "UNSUPPORTED_GENERATION" not in json.dumps(case["input"])


def test_incomplete_captured_evidence_fails_instead_of_reading_registry():
    with pytest.raises(ValueError, match="Missing captured"):
        captured_evidence({"trace_id": "a", "retrieved_chunk_references": [{"chunk_id": "missing"}]})


@pytest.mark.parametrize("change", ["unknown_id", "fabricated_quote", "fabricated_answer", "blank_quote"])
def test_invalid_judge_provenance_is_rejected(change):
    result = judgment()
    finding = result["findings"][0]
    if change == "unknown_id":
        finding["evidence"][0]["chunk_id"] = "invented"
    elif change == "fabricated_quote":
        finding["evidence"][0]["quote"] = "The fee is 90 units."
    elif change == "fabricated_answer":
        finding["answer_quote"] = "A sentence that was never in the answer."
    else:
        finding["evidence"][0]["quote"] = "   "
    with pytest.raises(ValueError):
        validate_judgment(json.dumps(result), control_cases()[0]["input"])


def test_contradiction_requires_contradicted_evidence_and_grounding_status():
    result = judgment()
    result["diagnosis"] = "GROUNDING_FAILURE"
    with pytest.raises(ValueError, match="contradiction diagnosis"):
        validate_judgment(json.dumps(result), control_cases()[0]["input"])


def test_whitespace_normalization_preserves_valid_quote():
    result = judgment()
    result["findings"][0]["evidence"][0]["quote"] = "The filing\nfee is 50 units."
    assert validate_judgment(json.dumps(result), control_cases()[0]["input"])["diagnosis"] == "HEALTHY"


def test_truncated_responses_are_retried_without_becoming_judgments():
    client = MagicMock()
    response = SimpleNamespace(
        id="r1", model=DEFAULT_MODELS[0], usage=None,
        choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=json.dumps(judgment())))])
    client.chat.completions.create.return_value = response
    with patch("scripts.run_independent_judges.rate_limiter.acquire"), patch("time.sleep") as sleep:
        outcome = judge_case(client, DEFAULT_MODELS[0], control_cases()[0], {"max_tokens": 8192}, sleep=sleep)
    assert outcome["status"] == "error"
    assert "judgment" not in outcome
    assert client.chat.completions.create.call_count == 3
    assert sleep.call_count == 2


def test_unavailable_model_stops_without_repeated_requests_or_account_leakage():
    client = MagicMock()
    error = RuntimeError("account=private-identifier")
    error.status_code = 404
    client.chat.completions.create.side_effect = error
    with patch("scripts.run_independent_judges.rate_limiter.acquire"):
        outcome = judge_case(client, DEFAULT_MODELS[0], control_cases()[0], {"max_tokens": 8192})
    assert outcome["status"] == "unavailable"
    assert client.chat.completions.create.call_count == 1
    assert "private-identifier" not in json.dumps(outcome)


def test_resume_reuses_successful_checkpoint_without_inference():
    case = control_cases()[0]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "judgments").mkdir()
        write_json(checkpoint_path(output, DEFAULT_MODELS[0], case), {
            "model": DEFAULT_MODELS[0], "case_id": case["case_id"],
            "input_sha256": digest(case["input"]), "status": "ok", "judgment": judgment(), "attempts": []})
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "offline-test"}), \
                patch("openai.OpenAI"), patch("scripts.run_independent_judges.judge_case") as judge:
            evaluate_model(DEFAULT_MODELS[0], [case], {"timeout_seconds": 240}, output, threading.Event())
        judge.assert_not_called()


def test_failed_initial_control_stops_that_model_before_benchmark_calls():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "judgments").mkdir()
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "offline-test"}), \
                patch("openai.OpenAI"), patch("scripts.run_independent_judges.judge_case") as judge:
            judge.return_value = {"status": "error", "attempts": []}
            evaluate_model(DEFAULT_MODELS[0], control_cases(), {"timeout_seconds": 240}, output, threading.Event())
        assert judge.call_count == 1
        assert len(list((output / "judgments").glob("*.json"))) == 1


def test_missing_or_inconclusive_pairs_do_not_inflate_comparison():
    cases = [benchmark_case(), {**benchmark_case(), "case_id": "eval-2"},
             {**benchmark_case(), "case_id": "eval-3"}]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "judgments").mkdir()
        for model in DEFAULT_MODELS:
            for case in cases[:2]:
                result = judgment()
                if case["case_id"] == "eval-2":
                    result["diagnosis"] = "INCONCLUSIVE"
                write_json(checkpoint_path(output, model, case), {
                    "model": model, "case_id": case["case_id"], "input_sha256": digest(case["input"]),
                    "status": "ok", "judgment": result, "attempts": []})
        summary = summarize(cases, DEFAULT_MODELS, output)
    assert summary["successful_judgments"] == 4
    assert summary["planned_judgments"] == 6
    assert summary["paired_benchmark_cases"] == 2
    assert summary["consensus_counts"] == {"HEALTHY": 1, "INCONCLUSIVE": 1, "DISAGREEMENT_OR_MISSING": 1}
    assert summary["cases"][1]["review_required"]
    assert summary["cases"][2]["review_required"]


def test_default_command_is_offline_and_changed_settings_cannot_reuse_run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results, traces = fixture_files(root)
        args = ["--results", str(results), "--traces", str(traces), "--output", str(root / "run")]
        with patch("openai.OpenAI", side_effect=AssertionError("No network allowed")):
            assert main(args) == 0
            assert main(args) == 0
            with pytest.raises(SystemExit) as exit_info:
                main(args + ["--max-tokens", "4096"])
        assert exit_info.value.code == 2
        assert not (root / "run" / "run.lock").exists()


def test_existing_lock_prevents_another_runner():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results, traces = fixture_files(root)
        output = root / "run"
        output.mkdir()
        (output / "run.lock").write_text("123")
        with pytest.raises(SystemExit) as exit_info:
            main(["--results", str(results), "--traces", str(traces), "--output", str(output)])
        assert exit_info.value.code == 1
        assert (output / "run.lock").read_text() == "123"
