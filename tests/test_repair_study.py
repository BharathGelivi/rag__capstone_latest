"""Offline checks for real-intervention planning and honest repair comparisons."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from scripts.run_repair_study import (
    GROUPS, apply_suggestion, baseline_state, build_report, get_condition,
    evaluate_answer, inject, main, phase_path, read_quality, run_study, scenarios,
)
from scripts.run_independent_judges import DEFAULT_MODELS, digest, write_json


def action(title):
    return {"title": title, "description": "A real engine recommendation"}


def group_case(group):
    return next(s for s in scenarios() if s["group"] == group)


def test_plan_has_exactly_100_distinct_policies_and_balanced_controls():
    plan = scenarios()
    assert len(plan) == 100
    assert len({s["question"] for s in plan}) == 100
    assert len({s["case_id"] for s in plan}) == 100
    assert Counter(s["group"] for s in plan) == {group: 20 for group in GROUPS}
    assert scenarios() == plan
    assert scenarios(seed=4) != plan


def test_document_fault_removes_actual_evidence_without_changing_pristine_source():
    scenario = group_case("missing_document")
    original = deepcopy(scenario)
    broken = inject(scenario)
    assert scenario == original
    assert scenario["rule_id"] not in {c["chunk_id"] for c in broken["corpus"]}
    assert scenario["rule_id"] in {c["chunk_id"] for c in scenario["corpus"]}
    repaired, result = apply_suggestion(broken, [action("Ingest Missing Domain Knowledge")], scenario["corpus"])
    assert result["applied"]
    assert result["oracle_assisted_document_supply"]
    assert set(c["chunk_id"] for c in repaired["corpus"]) == set(c["chunk_id"] for c in scenario["corpus"])


def test_suggested_top_k_fix_keeps_the_ranking_fault_active():
    scenario = group_case("retrieval_demote")
    broken = inject(scenario)
    repaired, result = apply_suggestion(broken, [action("Increase Retrieval Top-K")], scenario["corpus"])
    assert result["applied"]
    assert repaired["top_k"] == 5
    assert repaired["demoted_ids"] == broken["demoted_ids"]
    assert repaired["corpus"] == broken["corpus"]


def test_overlap_repair_combines_real_fragments_without_swapping_in_original_rule():
    scenario = group_case("chunk_boundary")
    broken = inject(scenario)
    repaired, result = apply_suggestion(broken, [action("Increase Chunk Overlap")], scenario["corpus"])
    assert result["applied"]
    assert repaired["top_k"] == broken["top_k"]
    assert {c["chunk_id"] for c in repaired["corpus"]} == {c["chunk_id"] for c in broken["corpus"]}
    assert scenario["rule_id"] not in {c["chunk_id"] for c in repaired["corpus"]}
    assert str(scenario["expected_facts"]["fee_units"]) not in broken["corpus"][0]["text"]
    assert str(scenario["expected_facts"]["fee_units"]) in repaired["corpus"][0]["text"]


def test_prompt_repair_preserves_the_induced_bad_instruction_and_changes_only_suggested_setting():
    scenario = group_case("misleading_prompt")
    broken = inject(scenario)
    repaired, _ = apply_suggestion(broken, [action("Require Explicit Citations")], scenario["corpus"])
    assert repaired["system_prompt"].startswith(broken["system_prompt"])
    assert str(scenario["wrong_fee"]) in repaired["system_prompt"]
    assert repaired["temperature"] == broken["temperature"]
    assert repaired["corpus"] == broken["corpus"]


def test_wrong_recommendation_does_not_magically_repair_the_injected_fault():
    scenario = group_case("missing_document")
    broken = inject(scenario)
    repaired, result = apply_suggestion(broken, [action("Increase Retrieval Top-K")], scenario["corpus"])
    assert result["applied"]
    assert scenario["rule_id"] not in {c["chunk_id"] for c in repaired["corpus"]}


def test_missing_suggestion_remains_unrepaired_and_controls_have_no_induced_fault():
    scenario = group_case("healthy_control")
    broken = inject(scenario)
    assert broken == baseline_state(scenario)
    repaired, result = apply_suggestion(broken, [], scenario["corpus"])
    assert repaired == broken
    assert result["applied"] is False


def test_condition_resume_reuses_success_and_rejects_changed_inputs():
    scenario = scenarios()[0]
    state = baseline_state(scenario)
    runtime = MagicMock()
    runtime.execute.return_value = {"status": "ok", "phase": "baseline", "trace": {}}
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "conditions").mkdir()
        assert get_condition(runtime, scenario, state, "baseline", output)["status"] == "ok"
        assert get_condition(runtime, scenario, state, "baseline", output)["status"] == "ok"
        assert runtime.execute.call_count == 1
        with pytest.raises(ValueError, match="frozen plan"):
            get_condition(runtime, scenario, {**state, "top_k": 1}, "baseline", output)


def test_condition_retries_transient_failure_and_records_it():
    scenario = scenarios()[0]
    runtime = MagicMock()
    runtime.execute.side_effect = [RuntimeError("transient"), {"status": "ok", "phase": "baseline", "trace": {}}]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "conditions").mkdir()
        with patch("scripts.run_repair_study.time.sleep") as sleep:
            condition = get_condition(runtime, scenario, baseline_state(scenario), "baseline", output)
        assert condition["status"] == "ok"
        assert len(condition["failed_attempts"]) == 1
        assert runtime.execute.call_count == 2
        sleep.assert_called_once_with(5)


def write_quality(output, scenario, phase, label, model=None):
    models = [model] if model else DEFAULT_MODELS
    for name in models:
        path = output / "judgments" / f"{scenario['case_id']}-{phase}-{digest(name)[:12]}.json"
        write_json(path, {"status": "ok", "judgment": {"diagnosis": label}})


def write_phase(output, scenario, phase, applied=True):
    write_json(phase_path(output, scenario["case_id"], phase), {
        "status": "ok", "phase": phase, "trace": {"generated_answer": "<script>unsafe</script>"},
        "diagnosis": {"primary_cause": "MISSING_CORPUS"},
        "repair": {"applied": applied, "implementation": "Ingest Missing Domain Knowledge"}})


def test_report_counts_observed_recovery_and_excludes_unobserved_or_disputed_pairs():
    plan = scenarios()[:3]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "conditions").mkdir()
        (output / "judgments").mkdir()
        for scenario in plan:
            for phase in ("baseline", "broken", "repaired"):
                write_phase(output, scenario, phase)
        for phase, label in (("baseline", "HEALTHY"), ("broken", "INCOMPLETE_ANSWER"), ("repaired", "HEALTHY")):
            write_quality(output, plan[0], phase, label)
        write_quality(output, plan[1], "baseline", "HEALTHY")
        write_quality(output, plan[1], "broken", "HEALTHY", DEFAULT_MODELS[0])
        write_quality(output, plan[1], "broken", "GROUNDING_FAILURE", DEFAULT_MODELS[1])
        assert read_quality(output, plan[1], "broken", DEFAULT_MODELS) is None
        summary = build_report(plan, output, DEFAULT_MODELS)
        assert summary["comparable_scenarios"] == 1
        assert summary["faults_that_degraded_healthy_baseline"] == 1
        assert summary["recovered_after_suggested_fix"] == 1
        assert summary["repair_recovery_rate"] == 1
        report = (output / "REPORT.html").read_text()
        assert "<script>unsafe</script>" not in report
        assert "&lt;script&gt;unsafe&lt;/script&gt;" in report


def test_no_suggested_fix_cannot_count_as_recovery_even_if_judge_recheck_changes():
    scenario = scenarios()[0]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "conditions").mkdir()
        (output / "judgments").mkdir()
        for phase, label in (("baseline", "HEALTHY"), ("broken", "UNSUPPORTED_GENERATION"), ("repaired", "HEALTHY")):
            write_phase(output, scenario, phase, applied=False)
            write_quality(output, scenario, phase, label)
        summary = build_report([scenario], output, DEFAULT_MODELS)
        assert summary["recovered_after_suggested_fix"] == 0
        assert summary["scenarios"][0]["quality_delta"] == 0


def test_healthy_control_regression_is_visible():
    scenario = group_case("healthy_control")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "conditions").mkdir()
        (output / "judgments").mkdir()
        for phase, label in (("baseline", "HEALTHY"), ("broken", "HEALTHY"), ("repaired", "GROUNDING_FAILURE")):
            write_phase(output, scenario, phase)
            write_quality(output, scenario, phase, label)
        summary = build_report([scenario], output, DEFAULT_MODELS)
        assert summary["healthy_control_regressions"] == 1
        assert summary["scenarios"][0]["quality_delta"] == -1


def test_degraded_case_with_missing_after_result_stays_in_recovery_denominator():
    scenario = scenarios()[0]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "conditions").mkdir()
        (output / "judgments").mkdir()
        write_quality(output, scenario, "baseline", "HEALTHY")
        write_quality(output, scenario, "broken", "INCOMPLETE_ANSWER")
        summary = build_report([scenario], output, DEFAULT_MODELS)
        assert summary["faults_that_degraded_healthy_baseline"] == 1
        assert summary["degraded_cases_missing_after_judgment"] == 1
        assert summary["repair_recovery_rate"] == 0
        assert summary["comparable_scenarios"] == 0


def test_judges_are_blinded_and_successful_peer_is_checkpointed_on_failure():
    scenario = scenarios()[0]
    captured = []

    def fake_judge(client, model, case, settings):
        captured.append(deepcopy(case))
        return {"status": "error" if model == DEFAULT_MODELS[0] else "ok",
                "attempts": [], "judgment": {"diagnosis": "HEALTHY"}}

    condition = {"status": "ok", "phase": "baseline", "trace": {"generated_answer": "answer"}}
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "judgments").mkdir()
        with ThreadPoolExecutor(max_workers=2) as pool, \
                patch("scripts.run_repair_study.judge_case", side_effect=fake_judge):
            with pytest.raises(RuntimeError, match="failed preflight"):
                evaluate_answer(dict.fromkeys(DEFAULT_MODELS), DEFAULT_MODELS, scenario, condition, output, {}, pool)
        assert len(list((output / "judgments").glob("*.json"))) == 2
        assert all(set(c["input"]) == {"question", "answer", "evidence"} for c in captured)
        assert all(c["input"]["evidence"] == [{"chunk_id": d["chunk_id"], "text": d["text"]}
                                               for d in scenario["corpus"]] for c in captured)


def test_full_orchestration_applies_actual_suggestion_and_resumes_without_repeating_calls():
    scenario = group_case("missing_document")
    runtime = MagicMock()

    def fake_execute(scenario, state, phase):
        present = scenario["rule_id"] in {c["chunk_id"] for c in state["corpus"]}
        return {"status": "ok", "phase": phase,
                "trace": {"generated_answer": "correct answer" if present else "cannot answer"},
                "diagnosis": {"primary_cause": "RETRIEVAL_MISS", "corrective_plan": {
                    "immediate_actions": [action("Increase Retrieval Top-K")]}}}

    def fake_judge(client, model, case, settings):
        return {"status": "ok", "attempts": [], "judgment": {
            "diagnosis": "HEALTHY" if case["input"]["answer"] == "correct answer" else "INCOMPLETE_ANSWER"}}

    runtime.execute.side_effect = fake_execute
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "conditions").mkdir()
        (output / "judgments").mkdir()
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "offline-test"}), patch("openai.OpenAI"), \
                patch("scripts.run_repair_study.StudyRuntime", return_value=runtime), \
                patch("scripts.run_repair_study.judge_case", side_effect=fake_judge) as judge:
            settings = {"models": DEFAULT_MODELS, "timeout_seconds": 240, "max_tokens": 8192}
            run_study([scenario], output, settings)
            run_study([scenario], output, settings)
            assert runtime.execute.call_count == 3
            assert judge.call_count == 6
        after = json.loads(phase_path(output, scenario["case_id"], "repaired").read_text())
        assert after["repair"]["implementation"] == "Increase Retrieval Top-K"
        assert scenario["rule_id"] not in {c["chunk_id"] for c in after["state"]["corpus"]}
        summary = json.loads((output / "summary.json").read_text())
        assert summary["recovered_after_suggested_fix"] == 0


def test_preparation_never_constructs_runtime_or_contacts_models():
    with tempfile.TemporaryDirectory() as directory:
        with patch("scripts.run_repair_study.StudyRuntime", side_effect=AssertionError("No model loading")), \
                patch("scripts.run_repair_study.run_study", side_effect=AssertionError("No inference")):
            assert main(["--output", directory]) == 0
            assert main(["--output", directory]) == 0
        plan = json.loads((Path(directory) / "scenarios.json").read_text())
        assert len(plan) == 100
        assert not list((Path(directory) / "conditions").glob("*.json"))
        assert not (Path(directory) / "run.lock").exists()
