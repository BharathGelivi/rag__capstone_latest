import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from fastapi.testclient import TestClient
from src.api_ui import app
from src.evidence_sources import resolve_source
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

    def test_debugger_does_not_call_models_on_read(self):
        with patch("src.chat_service.deep_analyze", side_effect=AssertionError("model call")):
            response = self.client.get(f"/ui/trace/{self.trace.trace_id}/evidence")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["sources"]), 2)

    def test_unsafe_source_paths_and_ids_cannot_expose_other_files(self):
        self.assertIsNone(resolve_source(self.root, "../private.pdf"))
        self.assertIsNone(resolve_source(self.root, ".env"))
        outside = self.root / "private.pdf"; outside.write_bytes(b"%PDF-fake")
        self.assertIsNone(resolve_source(self.root, str(outside)))
        data = self.root / "data"; data.mkdir()
        pdf = data / "document.pdf"; pdf.write_bytes(b"%PDF-test")
        self.assertEqual(resolve_source(self.root, str(pdf)), pdf)
        self.trace.retrieved_chunk_references[0]["source_file"] = str(pdf)
        RAGTraceBuilder.save_to_json(self.trace, str(self.root / "artifacts/rag_traces"))
        response = self.client.get(f"/ui/trace/{self.trace.trace_id}/source/c1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"%PDF-test")
        self.assertEqual(self.client.get("/ui/trace/bad*id/evidence").status_code, 422)

    def test_analysis_updates_the_existing_trace_without_overwriting_answer(self):
        outcome = {"verification": {"claim_count": 0, "results": []}}
        with patch("src.chat_service.deep_analyze", return_value=outcome):
            response = self.client.post(f"/ui/trace/{self.trace.trace_id}/evidence/analyze")
        self.assertEqual(response.status_code, 200)
        saved = json.loads(Path(self.path).read_text())
        self.assertEqual(saved["generated_answer"], self.trace.generated_answer)
        self.assertEqual(saved["diagnostics"]["claim_verification"], outcome["verification"])

    def test_analysis_save_failure_is_not_reported_as_success(self):
        with patch("src.chat_service.deep_analyze", return_value={"verification": {"results": []}}), \
             patch("src.chat_service.persist_deep_analysis", return_value=False):
            response = self.client.post(f"/ui/trace/{self.trace.trace_id}/evidence/analyze")
        self.assertEqual(response.status_code, 500)

    def test_deep_analysis_records_offsets_and_checks_the_cited_passage(self):
        from src.chat_service import deep_analyze
        from src.claim_decomposer import CandidateClaim, CandidateClaimSet
        from src.trace_replay import captured_chunks
        from src.claim_verifier import VerificationStatus
        answer = "A fine applies. [Chunk-ID: c2]"
        self.trace.generated_answer = answer
        candidate = CandidateClaim("claim", self.trace.trace_id, "A penalty applies.", "S1", 0, 0, 15)
        claims = CandidateClaimSet(trace_id=self.trace.trace_id)
        claims.add_claim(candidate)
        result = SimpleNamespace(claim_id="claim", claim_text=candidate.claim_text,
            verification_status=VerificationStatus.SUPPORTED, verification_reason="Supported by best evidence",
            confidence=.95, entailment_score=.95, contradiction_score=.01, neutral_score=.04,
            best_chunk_id="c1", best_chunk_rank=1, evidence_text="Text for c1.", verified_by="nli")
        with patch("src.chat_service.load_decomposer") as decomposer, patch("src.chat_service.load_verifier") as verifier:
            decomposer.return_value.decompose.return_value = claims
            verifier.return_value.verify_all.return_value = SimpleNamespace(results=[result])
            verifier.return_value.check_cited_chunk.return_value = {"chunk_id": "c2", "status": "UNSUPPORTED"}
            output = deep_analyze(self.trace, captured_chunks(self.trace))
        recorded = output["verification"]["results"][0]
        self.assertEqual(recorded["character_start"], 0)
        self.assertEqual(recorded["citation_checks"][0]["status"], "UNSUPPORTED")
        self.assertEqual(verifier.return_value.check_cited_chunk.call_args.args[1].chunk_id, "c2")
