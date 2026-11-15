"""Execution references must never be inferred from question-level expectations."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.diagnostic_labels import reviewed_reference, load_reviewed_references
from experiments.exp08_live_diagnostic_accuracy import LiveDiagnosticAccuracyExperiment


def row(reference=None):
    return {"trace_id": "trace-a", "eval_id": "1", "expected_failure_type": "RETRIEVAL_MISS",
            "xrag_primary_cause": "RETRIEVAL_MISS", "diagnostic_reference": reference}


def score(value):
    experiment = LiveDiagnosticAccuracyExperiment()
    record = experiment.run_example(SimpleNamespace(payload={"row": value}), None)
    return experiment.summarize([record], None)


def test_scenario_expectation_never_establishes_execution_accuracy():
    assert reviewed_reference(row()) is None
    headline = score(row())["headline"]
    assert headline["scenario_expectation_agreement"] == 1
    assert headline["n_unreviewed_rows"] == 1
    assert headline["reviewed_accuracy"] is None
    assert headline["overall_accuracy_all_40_rows"] is None
    assert headline["false_positive_rate_on_healthy"] is None


def test_model_consensus_is_provisional_agreement():
    value = row({"trace_id": "trace-a", "label": "RETRIEVAL_MISS", "source": "model_consensus"})
    headline = score(value)["headline"]
    assert headline["n_reviewed_rows"] == 1
    assert headline["reviewed_reference_agreement"] == 1
    assert headline["n_accuracy_eligible_rows"] == 0
    assert headline["reviewed_accuracy"] is None


def test_observed_review_can_override_question_expectation():
    value = row({"trace_id": "trace-a", "label": "HEALTHY", "source": "human_review"})
    value["xrag_primary_cause"] = "UNKNOWN"
    headline = score(value)["headline"]
    assert headline["reviewed_accuracy"] == 1
    assert headline["n_healthy"] == 1
    assert headline["scenario_expectation_agreement"] == 0


@pytest.mark.parametrize("reference", [
    {"trace_id": "another-trace", "label": "HEALTHY", "source": "human_review"},
    {"trace_id": "trace-a", "label": "UNKNOWN", "source": "human_review"},
    {"trace_id": "trace-a", "label": "HEALTHY", "source": "guess"},
])
def test_ambiguous_or_mismatched_references_are_unreviewed(reference):
    assert reviewed_reference(row(reference)) is None


def test_reference_file_rejects_duplicates_and_invalid_labels(tmp_path):
    path = tmp_path / "reviews.json"
    reference = {"trace_id": "trace-a", "label": "HEALTHY", "source": "controlled_intervention"}
    path.write_text(json.dumps([reference]))
    assert load_reviewed_references(path) == {"trace-a": reference}
    path.write_text(json.dumps([reference, reference]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_reviewed_references(path)
    path.write_text(json.dumps([{**reference, "label": "INCONCLUSIVE"}]))
    with pytest.raises(ValueError, match="Each reference"):
        load_reviewed_references(path)


def test_cli_keeps_speculative_agreement_out_of_accuracy(tmp_path, monkeypatch):
    from scripts import evaluate_diagnostic_accuracy as cli
    dataset = tmp_path / "questions.csv"
    dataset.write_text("id,expected_failure_type\n1,RETRIEVAL_MISS\n")
    results = tmp_path / "results.json"
    results.write_text(json.dumps([row()]))
    monkeypatch.setattr(cli, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["scorer", "--eval-dataset", str(dataset), "--results", str(results),
                                   "--reviewed-labels", str(tmp_path / "absent.json")])
    cli.main()
    output = json.loads((tmp_path / "diagnostic_accuracy.json").read_text())
    assert output["overall_accuracy"] is None
    assert output["scenario_expectation_agreement"] == 1
    assert output["n_examples_evaluated"] == 0
