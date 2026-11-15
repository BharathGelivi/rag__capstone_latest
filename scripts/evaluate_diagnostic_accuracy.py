"""Score saved diagnoses against trace-specific reviewed references.

Accuracy uses human_review or controlled_intervention references. Model-consensus
references contribute to provisional agreement. The question dataset's expected
failure labels are reported separately as scenario expectation agreement.

Optional --reviewed-labels accepts a JSON array of objects containing trace_id,
label and source. Missing references leave execution accuracy unmeasured."""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional
from pathlib import Path
from src.diagnostic_labels import reviewed_reference, load_reviewed_references

RESULTS_DIR = "artifacts/benchmark_comparison"

# X-RAG reports "UNKNOWN" for a healthy trace (no failure detected); the eval
# dataset represents the same thing as an empty expected_failure_type string.
HEALTHY_LABEL = "UNKNOWN"


def load_eval_dataset(path: str = "eval/eval_dataset.csv") -> Dict[str, str]:
    """Returns {eval_id: normalized_expected_label}."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["id"]: (row.get("expected_failure_type") or HEALTHY_LABEL) for row in rows}


def load_results(path: str) -> Dict[str, str]:
    """Returns {eval_id: xrag_primary_cause}."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {row["eval_id"]: row.get("xrag_primary_cause", HEALTHY_LABEL) for row in rows}


def join_labels(expected: Dict[str, str], predicted: Dict[str, str]) -> List[tuple]:
    """Pairs (expected_label, predicted_label) for every eval_id present in both."""
    return [(expected[eid], predicted[eid]) for eid in expected if eid in predicted]


def compute_confusion_matrix(pairs: List[tuple]) -> Dict[str, Dict[str, int]]:
    """{expected_label: {predicted_label: count}}, only over labels actually observed."""
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for expected, predicted in pairs:
        matrix[expected][predicted] += 1
    return {k: dict(v) for k, v in matrix.items()}


def compute_per_category_metrics(pairs: List[tuple]) -> Dict[str, Dict[str, Optional[float]]]:
    """Precision/recall/F1 for each label that appears as either expected or predicted."""
    categories = sorted({label for pair in pairs for label in pair})
    metrics = {}
    for category in categories:
        tp = sum(1 for e, p in pairs if e == category and p == category)
        fp = sum(1 for e, p in pairs if e != category and p == category)
        fn = sum(1 for e, p in pairs if e == category and p != category)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None

        support = sum(1 for e, _ in pairs if e == category)
        metrics[category] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return metrics


def compute_overall_accuracy(pairs: List[tuple]) -> Optional[float]:
    if not pairs:
        return None
    correct = sum(1 for e, p in pairs if e == p)
    return correct / len(pairs)


def find_mismatches(expected: Dict[str, str], predicted: Dict[str, str]) -> List[Dict[str, Any]]:
    mismatches = []
    for eid in expected:
        if eid not in predicted:
            continue
        if expected[eid] != predicted[eid]:
            mismatches.append({"eval_id": eid, "expected": expected[eid], "predicted": predicted[eid]})
    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Score trace-specific diagnostic references and report scenario expectations separately.")
    parser.add_argument("--eval-dataset", default="eval/eval_dataset.csv")
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "results.json"))
    parser.add_argument("--reviewed-labels", type=Path, default=Path("eval/diagnostic_reviews.json"))
    args = parser.parse_args()

    scenario_expected = load_eval_dataset(args.eval_dataset)
    predicted = load_results(args.results)
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    references = load_reviewed_references(args.reviewed_labels) if args.reviewed_labels.exists() else {}
    reviewed = {}
    for row in rows:
        row = {**row, "diagnostic_reference": references.get(row.get("trace_id"), row.get("diagnostic_reference"))}
        reference = reviewed_reference(row)
        if reference:
            reviewed[row["eval_id"]] = reference
    expected = {eid: ref["cause"] for eid, ref in reviewed.items() if ref["source"] != "model_consensus"}
    pairs = join_labels(expected, predicted)
    reference_pairs = join_labels({eid: ref["cause"] for eid, ref in reviewed.items()}, predicted)

    confusion_matrix = compute_confusion_matrix(pairs)
    per_category_metrics = compute_per_category_metrics(pairs)
    overall_accuracy = compute_overall_accuracy(pairs)
    mismatches = find_mismatches(
        {eid: label for eid, label in expected.items() if eid in predicted},
        predicted
    )

    output = {
        "n_examples_evaluated": len(pairs),
        "n_total_eval_dataset": len(scenario_expected),
        "n_reviewed_traces": len(reviewed),
        "n_unreviewed_traces": len(predicted) - len(reviewed),
        "reference_agreement": compute_overall_accuracy(reference_pairs),
        "scenario_expectation_agreement": compute_overall_accuracy(join_labels(scenario_expected, predicted)),
        "reference_scope": "Accuracy uses human_review or controlled_intervention references tied to trace IDs. Model consensus contributes to provisional agreement only. Question expectations do not establish execution accuracy.",
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": confusion_matrix,
        "per_category_metrics": per_category_metrics,
        "mismatches": mismatches,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "diagnostic_accuracy.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
