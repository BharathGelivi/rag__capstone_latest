import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import trace_metrics
from src.rag_trace import RAGTrace


class TestSavedFrameworkPanel(unittest.TestCase):
    def test_analysis_persists_when_legacy_diagnostics_is_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"diagnostics": None}), encoding="utf-8")
            outcome = {"verification": {"claim_count": 0, "results": []}}
            trace_metrics.chat_service.persist_deep_analysis(str(path), outcome)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["diagnostics"]["claim_verification"], outcome["verification"])

    def test_persisted_verification_is_reused_for_both_trace_formats(self):
        trace = RAGTrace(
            trace_id="saved", trace_version="1.0", pipeline_version="1.0",
            framework_version="1.0", timestamp="2026-11-14T00:00:00Z",
            question="Q", generated_answer="A", prompt_snapshot="", prompt_length=0,
            retrieved_chunk_references=[], configuration_snapshot={},
            execution_statistics={}, pipeline_stage_status={},
        )
        verification = {"claim_count": 1, "results": [
            {"status": "SUPPORTED", "entailment_score": 0.98}]}
        for canonical in (False, True):
            with self.subTest(canonical=canonical), tempfile.TemporaryDirectory() as directory:
                payload = json.loads(trace.to_json())
                if canonical:
                    payload["diagnostics"] = {"claim_verification": verification}
                else:
                    payload["claim_verification"] = verification
                path = Path(directory) / "trace.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                trace_metrics._CACHE.clear()
                with patch.object(trace_metrics, "_trace_file", return_value=str(path)), \
                     patch.object(trace_metrics, "_rebuild_chunks", return_value=[]), \
                     patch.object(trace_metrics, "_ragas_evaluator"), \
                     patch.object(trace_metrics, "compute_metric_panel", return_value={}), \
                     patch.object(trace_metrics.chat_service, "deep_analyze") as analyze:
                    panel = trace_metrics.compute_framework_panel("saved")
                    self.assertEqual(panel["verification"]["supported"], 1)
                    self.assertFalse(panel["computed_claims"])
                    self.assertTrue(trace_metrics.compute_framework_panel("saved")["cached"])
                    analyze.assert_not_called()
        trace_metrics._CACHE.clear()
