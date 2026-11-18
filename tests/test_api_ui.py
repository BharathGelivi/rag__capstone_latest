"""Fast, no-browser checks for src/api_ui.py -- run before the Playwright e2e suite."""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api_ui import app


class TestApiUI(unittest.TestCase):
    def test_experiments_list_and_open_saved_run_with_real_catalog(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as root:
            base = Path(root) / "artifacts" / "experiments"
            run = base / "runs" / "run40_test"
            study = run / "exp01_fault_injection"
            study.mkdir(parents=True)
            (base / "active_run.json").write_text(json.dumps({"run_id": "run40_test"}), encoding="utf-8")
            (study / "summary.json").write_text(json.dumps({"headline": {"accuracy": 0.75}}), encoding="utf-8")
            (study / "records.jsonl").write_text(json.dumps({"example_id": "saved-example"}) + "\n", encoding="utf-8")
            with patch("src.api_ui.PROJECT_ROOT", root):
                listing = self.client.get("/ui/experiments")
                self.assertEqual(listing.status_code, 200, listing.text)
                self.assertEqual(len(listing.json()["experiments"]), 13)
                detail = self.client.get("/ui/experiments/exp01_fault_injection")
                self.assertEqual(detail.status_code, 200, detail.text)
                self.assertEqual(detail.json()["summary"]["headline"]["accuracy"], 0.75)
                self.assertEqual(detail.json()["records"][0]["example_id"], "saved-example")

    def setUp(self):
        self.client = TestClient(app)
        mm = MagicMock()
        mm.list_sessions.return_value = []
        mm.get_session_memories.return_value = []
        mm.store.get_all_memories.return_value = []
        mm.search_memory.return_value = []
        mm.health.return_value = {"status": "healthy", "memory_count": 0, "malformed_entry_count": 0}
        patcher = patch("src.chat_service.get_memory_manager", return_value=mm)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_config_lists_arms_and_corpora(self):
        r = self.client.get("/ui/config")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("D_ircot", body["arms"])
        self.assertIn("F_graphrag", body["arms"])
        self.assertIn("statutes", body["corpora"])
        self.assertIn("device", body)

    def test_session_deletion_failure_returns_retryable_error(self):
        with patch("src.chat_service.get_memory_manager") as manager:
            manager.return_value.delete_session.side_effect = OSError("private database details")
            response = self.client.delete("/ui/sessions/session1")
        self.assertEqual(response.status_code, 503)
        self.assertIn("retry", response.json()["detail"])
        self.assertNotIn("private database", response.text)

    def test_trace_metrics_unknown_trace_is_404(self):
        r = self.client.get("/ui/trace/no-such-trace-id/metrics")
        self.assertEqual(r.status_code, 404)

    def test_trace_metrics_returns_panel_shape(self):
        fake_panel = {
            "trace_id": "TRACE_X",
            "note": "RAGAS-style ...",
            "families": [
                {"family": "RAGAS", "metrics": [
                    {"key": "faithfulness", "label": "Faithfulness", "value": 0.8,
                     "definition": "supported claims / total claims", "judge": "local NLI verifier",
                     "requires": None},
                ]},
            ],
            "verification": {"total": 3, "supported": 2, "partially_supported": 0,
                             "contradicted": 0, "unsupported": 1, "not_verifiable": 0,
                             "average_entailment": 0.7},
            "computed_claims": False,
            "cached": False,
        }
        from unittest.mock import patch
        with patch("src.api_ui.compute_framework_panel", return_value=fake_panel):
            r = self.client.get("/ui/trace/TRACE_X/metrics")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["trace_id"], "TRACE_X")
        fams = {f["family"] for f in body["families"]}
        self.assertIn("RAGAS", fams)
        self.assertEqual(body["verification"]["unsupported"], 1)

    def test_reports_list_and_traversal_protection(self):
        r = self.client.get("/ui/reports")
        self.assertEqual(r.status_code, 200)
        self.assertIn("reports", r.json())
        r = self.client.get("/ui/reports/../.env")
        self.assertEqual(r.status_code, 404)

    def test_session_report_refresh_keeps_one_artifact_and_links_diagnostics(self):
        session = SimpleNamespace(session_id="saved-chat", title="<Session>")
        first = SimpleNamespace(question="<First>", answer="First answer", trace_id="trace1", timestamp="2026-11-17")
        second = SimpleNamespace(question="Second question", answer="Second answer", trace_id=None, timestamp="2026-11-18")
        trace = SimpleNamespace(retrieved_chunk_references=[{"chunk_id": "chunk1", "text": "<Evidence>"}])
        with (
            TemporaryDirectory() as root,
            patch("src.api_ui.PROJECT_ROOT", root),
            patch("src.api_ui.load_trace", return_value=trace),
            patch("src.chat_service.get_memory_manager") as manager,
        ):
            manager.return_value.list_sessions.return_value = [session]
            manager.return_value.get_session_memories.return_value = []
            endpoint = "/ui/sessions/saved-chat/report"
            self.assertEqual(self.client.post(endpoint).status_code, 409)
            self.assertEqual(self.client.post("/ui/sessions/missing/report").status_code, 404)

            report_dir = Path(root) / "artifacts" / "diagnostic_reports"
            report_dir.mkdir(parents=True)
            (report_dir / "trace1.html").write_text("Existing diagnostic", encoding="utf-8")
            manager.return_value.get_session_memories.return_value = [first]
            saved = self.client.post(endpoint)
            self.assertEqual(saved.status_code, 200, saved.text)
            url = saved.json()["url"]
            self.assertNotIn("Second answer", self.client.get(url).text)

            manager.return_value.get_session_memories.return_value = [first, second]
            self.assertEqual(self.client.post(endpoint).json()["url"], url)
            page = self.client.get(url).text
            self.assertIn("&lt;Session&gt;", page)
            self.assertIn("&lt;Evidence&gt;", page)
            self.assertIn('href="trace1.html"', page)
            self.assertLess(page.index("&lt;First&gt;"), page.index("Second answer"))
            listing = self.client.get("/ui/reports").json()["reports"]
            self.assertEqual([item["url"] for item in listing if item["session_id"] == "saved-chat"], [url])
            self.assertEqual((report_dir / "trace1.html").read_text(encoding="utf-8"), "Existing diagnostic")

    def test_graph_memory_scope(self):
        r = self.client.get("/ui/graph", params={"scope": "memory", "max_nodes": 20})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        node_ids = {n["id"] for n in body["nodes"]}
        for edge in body["edges"]:
            self.assertIn(edge["source"], node_ids)
            self.assertIn(edge["target"], node_ids)

    def test_graph_rag_scope(self):
        r = self.client.get("/ui/graph", params={"scope": "rag", "max_nodes": 20})
        self.assertEqual(r.status_code, 200)
        self.assertIn("nodes", r.json())

    def test_graph_bad_scope_rejected(self):
        r = self.client.get("/ui/graph", params={"scope": "bogus"})
        self.assertEqual(r.status_code, 400)

    def test_sessions_list(self):
        r = self.client.get("/ui/sessions")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_memory_list_without_search(self):
        r = self.client.get("/ui/memory", params={"limit": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["searched"])
        self.assertIsInstance(body["memories"], list)

    def test_memory_health_shape(self):
        r = self.client.get("/ui/memory/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn(body["status"], {"healthy", "degraded"})
        self.assertIn("memory_count", body)
        self.assertIn("malformed_entry_count", body)

    def test_memory_search_returns_scored_results(self):
        r = self.client.get("/ui/memory", params={"search": "arrest", "limit": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["searched"])
        for m in body["memories"]:
            self.assertIn("final_score", m)
            self.assertIn("memory", m)

    def test_trace_not_found_is_404(self):
        r = self.client.get("/ui/trace/does-not-exist")
        self.assertEqual(r.status_code, 404)

    def test_latest_trace_matches_shape_when_present(self):
        r = self.client.get("/ui/trace/latest")
        # Empty artifacts/rag_traces/ in a fresh checkout is a legitimate
        # 404, not a bug -- only assert the shape when one exists.
        if r.status_code == 200:
            body = r.json()
            self.assertIn("trace_id", body)
            self.assertIn("prompt_snapshot", body)


class TestChatFailure(unittest.TestCase):
    def test_unexpected_failure_is_logged_and_returns_safe_terminal_event(self):
        from unittest.mock import patch
        from src import chat_service
        with patch.object(chat_service, '_run_chat_turn', side_effect=RuntimeError('secret credential')):
            with self.assertLogs(chat_service.logger, level='ERROR') as logs:
                events = list(chat_service.run_chat_turn('question', 'session'))
        self.assertEqual(events[0]['event'], 'error')
        self.assertIn('reference_id', events[0])
        self.assertNotIn('secret credential', str(events))
        self.assertIn('secret credential', '\n'.join(logs.output))


if __name__ == "__main__":
    unittest.main()
