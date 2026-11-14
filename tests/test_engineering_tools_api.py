import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from fastapi.testclient import TestClient
from src.api_ui import app
from src.regression_cases import capture_case, Expectations
from src.rag_trace import RAGTraceBuilder
from tests.test_trace_replay import source_trace


class TestEngineeringToolsApi(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for target in ("src.chat_service.PROJECT_ROOT", "src.api_ui.PROJECT_ROOT"):
            patcher = patch(target, self.directory.name); patcher.start(); self.addCleanup(patcher.stop)
        self.trace = source_trace()
        self.path = RAGTraceBuilder.save_to_json(self.trace, str(self.root / "artifacts/rag_traces"))
        self.client = TestClient(app)

    def test_save_case_captures_expectations_without_memory_or_private_prompt(self):
        self.trace.diagnostics["memory_ids"] = ["private"]
        case = capture_case(self.trace, "Case", Expectations(required_facts=["punishment"]))
        self.assertEqual(case.snapshot["diagnostics"], {})
        self.assertEqual(case.snapshot["prompt_snapshot"], "")
        self.assertNotIn("Earlier question", case.model_dump_json())
        response = self.client.post(f"/ui/trace/{self.trace.trace_id}/regression-cases", json={"name": "Failure", "expectations": {"required_facts": ["missing fact"]}})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["baseline"]["passed"])
        saved = self.root / response.json()["path"]
        self.assertTrue(saved.exists())
        cid = response.json()["case_id"]
        run = self.client.post("/ui/regression-cases/run", json={"case_ids": [cid], "mode": "snapshot"})
        self.assertEqual(run.status_code, 200)
        self.assertFalse(run.json()["passed"])
