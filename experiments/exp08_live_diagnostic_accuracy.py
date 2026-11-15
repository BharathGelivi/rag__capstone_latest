"""E8 scores diagnoses against explicit references for completed execution traces.

Question-level expected_failure_type values are scenario expectations. They are
reported as expectation agreement and never used as execution ground truth.
Optional eval/diagnostic_reviews.json entries identify a trace_id, an explicit
label and a source (human_review, controlled_intervention or model_consensus).
Model consensus contributes to provisional agreement only. Unreviewed rows are
excluded from diagnostic accuracy, with coverage reported in the headline.
The experiment re-scores saved artifacts and makes no new inference calls."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
from src.diagnostic_labels import reviewed_reference, load_reviewed_references

from experiments.common import (
    ExampleSpec,
    Experiment,
    ExperimentContext,
    accuracy,
    confusion_matrix,
    macro_f1,
    per_label_prf,
    wilson_interval,
)
from experiments.exp05_diagnostic_agreement import load_real_rows
from src.root_cause_reasoner import FailureType


class LiveDiagnosticAccuracyExperiment(Experiment):
    key = "exp08_live_diagnostic_accuracy"
    number = 8
    title = "Diagnostic agreement against reviewed execution traces"
    claim = (
        "Compare diagnoses with trace-specific reviewed references. Question-level "
        "failure expectations are reported separately and cannot establish accuracy."
    )
    #: This experiment re-scores already-collected E5 output; it performs no
    #: new pipeline calls regardless of mode.
    supported_modes = ("offline", "live")

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        rows = load_real_rows()
        reviews_path = Path("eval/diagnostic_reviews.json")
        reviews = load_reviewed_references(reviews_path) if reviews_path.exists() else {}
        rows = [{**row, "diagnostic_reference": reviews.get(row.get("trace_id"), row.get("diagnostic_reference"))}
                for row in rows]
        if not rows:
            raise RuntimeError(
                "exp08: artifacts/benchmark_comparison/results.json not found or "
                "empty. Run E5's baseline comparison first -- exp08 re-scores its "
                "output rather than calling the pipeline itself."
            )
        return [
            ExampleSpec(example_id=f"real/{row.get('eval_id', i)}", payload={"row": row})
            for i, row in enumerate(rows)
        ]

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        row = spec.payload["row"]
        reference = reviewed_reference(row)
        true_cause = reference["cause"] if reference else None
        xrag_cause = row.get("xrag_primary_cause") or FailureType.UNKNOWN.value
        expected = row.get("expected_failure_type") or FailureType.UNKNOWN.value
        return {
            "eval_id": row.get("eval_id"),
            "question": row.get("question"),
            "true_cause": true_cause,
            "reference_source": reference["source"] if reference else None,
            "xrag_cause": xrag_cause,
            "is_labeled_failure": true_cause is not None and true_cause != FailureType.UNKNOWN.value,
            "correct": true_cause == xrag_cause if reference and reference["source"] != "model_consensus" else None,
            "matches_scenario_expectation": expected == xrag_cause,
        }

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        reviewed = [r for r in records if r["true_cause"] is not None]
        validated = [r for r in reviewed if r["reference_source"] != "model_consensus"]
        failures = [r for r in validated if r["is_labeled_failure"]]
        healthy = [r for r in validated if not r["is_labeled_failure"]]

        truth_all = [r["true_cause"] for r in validated]
        pred_all = [r["xrag_cause"] for r in validated]

        detection_rate = (
            sum(1 for r in failures if r["xrag_cause"] != FailureType.UNKNOWN.value) / len(failures)
            if failures else None
        )
        attribution_accuracy = accuracy(
            [r["true_cause"] for r in failures], [r["xrag_cause"] for r in failures]
        ) if failures else None

        false_positives = sum(1 for r in healthy if r["xrag_cause"] != FailureType.UNKNOWN.value)
        false_positive_rate = false_positives / len(healthy) if healthy else None
        fpr_ci = wilson_interval(false_positives, len(healthy)) if healthy else (None, None)

        return {
            "headline": {
                "n_real_rows": len(records),
                "n_reviewed_rows": len(reviewed),
                "n_unreviewed_rows": len(records) - len(reviewed),
                "n_accuracy_eligible_rows": len(validated),
                "reviewed_reference_agreement": accuracy([r["true_cause"] for r in reviewed], [r["xrag_cause"] for r in reviewed]) if reviewed else None,
                "scenario_expectation_agreement": sum(r["matches_scenario_expectation"] for r in records) / len(records) if records else None,
                "reviewed_accuracy": accuracy(truth_all, pred_all) if validated else None,
                "n_labeled_failures": len(failures),
                "n_healthy": len(healthy),
                "detection_rate": detection_rate,
                "attribution_accuracy_on_labeled_failures": attribution_accuracy,
                "false_positive_rate_on_healthy": false_positive_rate,
                "false_positive_rate_95ci": fpr_ci,
                "overall_accuracy_all_40_rows": accuracy(truth_all, pred_all) if len(validated) == len(records) and validated else None,
                "macro_f1": macro_f1(truth_all, pred_all) if validated else None,
            },
            "per_label_prf": per_label_prf(truth_all, pred_all),
            "confusion_truth_rows_vs_xrag_columns": confusion_matrix(truth_all, pred_all),
            "reference_scope": "Accuracy uses explicit human-reviewed or controlled-intervention execution references. Model consensus is provisional agreement.",
            "interpretation_notes": [
                "Question-level expected_failure_type and blank labels are scenario expectations. They do not label observed trace failures or healthy executions.",
                "Unreviewed traces are excluded from accuracy. Model-consensus references contribute only to reference agreement.",
                "This experiment performs no new pipeline calls; it re-scores "
                "artifacts/benchmark_comparison/results.json, which was produced by "
                "E5's live run. If that file is regenerated, re-run this experiment "
                "with --force to refresh the scoring against the new rows.",
            ],
        }


EXPERIMENT = LiveDiagnosticAccuracyExperiment()
