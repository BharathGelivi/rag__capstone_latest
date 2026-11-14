"""CI-safe assertion-engine tests: pydantic + stdlib, no model imports or calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from src.regression_cases import Expectations, RegressionCase, evaluate_case, load_cases, run_case, save_case
from scripts.run_regression_cases import main


def case(**expectations):
    return RegressionCase(name="Penalty", source_trace_id="original", expectations=Expectations(**expectations),
        snapshot={"question": "Penalty?", "generated_answer": "A fine of 50 units.",
                  "retrieved_chunk_references": [{"chunk_id": "c1", "text": "Penalty: 50 units."}]})


class TestRegressionCases(unittest.TestCase):
    def test_text_expectations_pass_and_failure_reports_each_assertion(self):
        c = case(expected_evidence=["c1"], required_facts=["FINE OF  50"], forbidden_claims=["prison"])
        self.assertTrue(evaluate_case(c, c.snapshot)["passed"])
        result = evaluate_case(c, {"generated_answer": "Prison", "retrieved_chunk_references": []})
        self.assertFalse(result["passed"])
        self.assertEqual(sum(r["status"] == "failed" for r in result["checks"]), 3)

    def test_citing_an_id_does_not_satisfy_expected_prompt_evidence(self):
        c = case(expected_evidence=["missing"])
        result = evaluate_case(c, {"generated_answer": "Answer [Chunk-ID: missing]", "retrieved_chunk_references": []})
        self.assertFalse(result["passed"])

    def test_refusal_required_allowed_and_forbidden(self):
        refused = {"generated_answer": "I don't know.", "retrieved_chunk_references": []}
        self.assertTrue(evaluate_case(case(refusal="required", required_facts=["50"]), refused)["passed"])
        self.assertFalse(evaluate_case(case(refusal="forbidden"), refused)["passed"])
        self.assertTrue(evaluate_case(case(refusal="allowed", required_facts=["50"]), refused)["passed"])

    def test_semantic_required_and_forbidden_assertions(self):
        c = case(fact_match="nli", required_facts=["a penalty"], forbidden_claims=["jail"])
        self.assertTrue(evaluate_case(c, c.snapshot, lambda answer, fact: {"entailment": .95 if fact == "a penalty" else .05})["passed"])
        uncertain = evaluate_case(c, c.snapshot, lambda *_: {"entailment": .5})
        self.assertFalse(uncertain["passed"])
        self.assertIn("uncertain", [r["status"] for r in uncertain["checks"]])
        self.assertFalse(evaluate_case(c, c.snapshot)["passed"])

    def test_invalid_and_failed_nli_judgments_do_not_pass(self):
        c = case(fact_match="nli", forbidden_claims=["jail"])
        for score in (float("nan"), float("inf"), -1, 2):
            self.assertFalse(evaluate_case(c, c.snapshot, lambda *_, score=score: {"entailment": score})["passed"])

    def test_save_is_exclusive_and_reload_preserves_expectations(self):
        with tempfile.TemporaryDirectory() as directory:
            c = case(required_facts=["50"])
            save_case(c, directory)
            self.assertEqual(load_cases(directory)[0], c)
            with self.assertRaises(FileExistsError): save_case(c, directory)

    def test_invalid_case_and_empty_assertions_are_rejected(self):
        for expectations in ({"required_facts": [" "]}, {"refusal": "allowed"}, {"fact_match": "magic"}):
            with self.assertRaises(ValidationError): case(**expectations)
        with self.assertRaises(ValidationError): RegressionCase.model_validate(case().model_dump() | {"case_id": "../escape"})

    def test_cli_exits_nonzero_for_failures_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            save_case(case(required_facts=["missing fact"]), directory)
            report = str(Path(directory)/"results.json")
            self.assertEqual(main(["--cases", directory, "--mode", "snapshot", "--output", report]), 1)
            self.assertFalse(json.loads(Path(report).read_text())["passed"])

    def test_execution_errors_are_unavailable_not_passed(self):
        with patch("src.regression_cases.generate_candidate", side_effect=RuntimeError("private server error")):
            result = run_case(case())
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][0]["status"], "unavailable")
        self.assertNotIn("private server", str(result))

    def test_shipped_case_passes_its_snapshot(self):
        for value in load_cases("eval/regression_cases"):
            self.assertTrue(evaluate_case(value, value.snapshot)["passed"])

    def test_new_quality_failure_is_flagged_as_a_regression(self):
        c = case(required_facts=["50"])
        with patch("src.regression_cases.generate_candidate", return_value={"generated_answer": "A fine of 100 units.", "retrieved_chunk_references": []}):
            result = run_case(c)
        self.assertEqual(result["change"], "regressed")
        self.assertIn("100", result["answer"])
