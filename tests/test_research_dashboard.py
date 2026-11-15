import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.research_dashboard import experiment_directory, research_summary


class TestResearchSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "artifacts/experiments"
        self.active = self.base / "runs/run_test"
        self.active.mkdir(parents=True)
        (self.base / "active_run.json").write_text('{"run_id":"run_test"}')

    def study(self, directory, name="exp01_fault_injection", **overrides):
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        value = {"title": "Injected", "mode": "live", "n_examples": 40,
                 "headline": {"overall_recovery_accuracy": .75}, **overrides}
        (target / "summary.json").write_text(json.dumps(value))

    def test_keeps_runs_separate_and_returns_recorded_provenance(self):
        self.study(self.active)
        self.study(self.base, n_examples=158)
        data = research_summary(self.root)
        self.assertEqual(len(data["studies"]), 1)
        self.assertEqual(data["studies"][0]["n_examples"], 40)
        self.assertEqual(data["source"], "active")
        self.assertIn("runs/run_test/", data["studies"][0]["source_path"])
        self.assertEqual(research_summary(self.root, "legacy")["studies"][0]["n_examples"], 158)

    def test_unknown_source_cannot_silently_show_legacy_results(self):
        with self.assertRaises(ValueError):
            experiment_directory(self.root, "actvie")

    def test_corrupt_or_missing_active_run_is_explicit(self):
        pointer = self.base / "active_run.json"
        for contents in ('{', '[]', '{}', '{"run_id":"../private"}', '{"run_id":"run_missing"}'):
            with self.subTest(contents=contents):
                pointer.write_text(contents)
                with self.assertRaises(RuntimeError):
                    experiment_directory(self.root, "active")

    def test_legacy_fallback_is_labeled(self):
        (self.base / "active_run.json").unlink()
        self.assertEqual(research_summary(self.root)["source"], "legacy")

    def test_one_corrupt_summary_does_not_hide_other_studies(self):
        self.study(self.active)
        bad = self.active / "exp02_invalid"
        bad.mkdir()
        (bad / "summary.json").write_text('{"headline":')
        data = research_summary(self.root)
        self.assertEqual(len(data["studies"]), 1)
        self.assertEqual(len(data["warnings"]), 1)

    def test_non_finite_metrics_become_missing_not_zero(self):
        self.study(self.active, headline={"accuracy": float("nan")})
        self.assertIsNone(research_summary(self.root)["studies"][0]["headline"]["accuracy"])


class TestResearchApi(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from src.api_ui import app
        self.client = TestClient(app)

    def test_invalid_metric_id_rejected_before_any_model_call(self):
        with patch("src.api_ui.compute_framework_panel", side_effect=AssertionError("model call")):
            for trace_id in ("wild*card", "bracket[id]"):
                self.assertEqual(self.client.get(f"/ui/trace/{trace_id}/metrics").status_code, 422)

    def test_captured_trace_does_not_load_mutable_registries(self):
        from tests.test_trace_replay import source_trace
        trace = source_trace()
        for ref in trace.retrieved_chunk_references:
            ref["text"] = "Captured evidence"
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "artifacts/rag_traces/recorded/trace_captured.json"
            path.parent.mkdir(parents=True)
            path.write_text(trace.to_json(), encoding="utf-8")
            with patch("src.api_ui.PROJECT_ROOT", root), patch("src.chat_service.load_registry", side_effect=AssertionError("unnecessary I/O")):
                self.assertEqual(self.client.get("/ui/trace/captured").status_code, 200)

    def test_corrupt_monitor_does_not_break_saved_experiments(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root) / "artifacts/experiments"
            base.mkdir(parents=True)
            (base / "monitor.json").write_text('{"status":')
            with patch("src.api_ui.PROJECT_ROOT", root):
                response = self.client.get("/ui/experiments?source=legacy")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["monitor"]["status"], "unreadable")

    def test_research_endpoint_is_read_only_and_validates_source(self):
        with tempfile.TemporaryDirectory() as root, patch("src.api_ui.PROJECT_ROOT", root), \
                patch("src.chat_service.deep_analyze", side_effect=AssertionError("model call")):
            response = self.client.get("/ui/research-summary?source=legacy")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["studies"], [])
            self.assertEqual(self.client.get("/ui/research-summary?source=typo").status_code, 422)
