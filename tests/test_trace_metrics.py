import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from src import trace_metrics
from src.rag_trace import RAGTrace


class TestSavedFrameworkPanel(unittest.TestCase):
    def test_reanalysis_uses_captured_memory_without_current_store(self):
        trace = RAGTrace("memory", "1", "1", "1", "2026-11-15", "Fee?", "50 units", "", 0,
                         [], {}, {}, {}, {"memory_ids": ["m1"], "memory_evidence": [{
                             "chunk_id": "memory:m1", "chunk_text": "The fee was 50 units.",
                             "similarity_score": 1.0, "rank": 1, "page_number": "", "source_file": "memory", "chunk_index": 0}]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(trace.to_json())
            trace_metrics._CACHE.clear()
            with patch.object(trace_metrics, "_trace_file", return_value=str(path)), \
                    patch.object(trace_metrics, "_ragas_evaluator"), \
                    patch.object(trace_metrics, "compute_metric_panel", return_value={}), \
                    patch.object(trace_metrics.chat_service, "get_memory_manager", side_effect=AssertionError("mutable memory")), \
                    patch.object(trace_metrics.chat_service, "deep_analyze", return_value={"verification": {"claim_count": 0, "results": []}}) as analyze:
                trace_metrics.compute_framework_panel("memory")
                self.assertEqual(analyze.call_args.args[1][0].chunk_text, "The fee was 50 units.")
            trace_metrics._CACHE.clear()

    def test_legacy_missing_memory_is_unavailable_without_reverification(self):
        trace = RAGTrace("legacy", "1", "1", "1", "2026-11-15", "Fee?", "50 units", "", 0,
                         [], {}, {}, {}, {"memory_ids": ["deleted"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(trace.to_json())
            trace_metrics._CACHE.clear()
            with patch.object(trace_metrics, "_trace_file", return_value=str(path)), \
                    patch.object(trace_metrics, "_ragas_evaluator"), \
                    patch.object(trace_metrics, "compute_metric_panel", return_value={}), \
                    patch.object(trace_metrics.chat_service, "deep_analyze") as analyze:
                result = trace_metrics.compute_framework_panel("legacy")
                analyze.assert_not_called()
                self.assertIsNone(result["verification"])
                self.assertIn("not captured", result["analysis_error"])
            trace_metrics._CACHE.clear()

    def test_failed_analysis_write_preserves_original_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            original = {"trace_id": "original", "diagnostics": {"retained": "evidence"}}
            path.write_text(json.dumps(original), encoding="utf-8")
            def interrupted(value, handle, **kwargs):
                handle.write('{"incomplete":')
                raise OSError("Interrupted write")
            with patch("src.memory.memory_utils.json.dump", side_effect=interrupted):
                self.assertFalse(trace_metrics.chat_service.persist_deep_analysis(str(path), {"verification": {"results": []}}))
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_concurrent_analysis_updates_preserve_valid_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"diagnostics": {"retained": "evidence"}}))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(trace_metrics.chat_service.persist_deep_analysis, str(path), outcome)
                           for outcome in ({"verification": {"results": []}}, {"error": "Unavailable"})]
                self.assertTrue(all(f.result() for f in futures))
            diagnostics = json.loads(path.read_text())["diagnostics"]
            self.assertEqual(diagnostics["retained"], "evidence")
            self.assertIn("claim_verification", diagnostics)

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
