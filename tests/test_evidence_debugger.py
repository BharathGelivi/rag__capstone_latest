import unittest
from src.evidence_debugger import answer_sentences, build_evidence_debugger, sentence_for_claim


class TestEvidenceDebugger(unittest.TestCase):
    def test_known_bare_citation_after_blank_line_belongs_to_previous_sentence(self):
        answer = "A fine applies.\n\n[fixture-penalty]"
        rows = sentence_for_claim(answer, {"claim_text": "A fine applies."}, ["fixture-penalty"])
        self.assertEqual(rows[0]["citations"], ["fixture-penalty"])
        self.assertEqual(len(answer_sentences(answer, ["fixture-penalty"])), 1)

    def test_offsets_and_citations_stay_with_answer_sentence(self):
        answer = "A fine applies. [Chunk-ID: c1] Jail is excluded. [Chunk-ID: c2]"
        rows = answer_sentences(answer)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["citations"], ["c1"])
        for row in rows: self.assertEqual(answer[row["start"]:row["end"]], row["text"])
        self.assertEqual(sentence_for_claim(answer, {"claim_text": "A fine applies."})[0]["sentence_id"], "S1")

    def test_paraphrases_without_offsets_are_unmapped_not_guessed(self):
        self.assertEqual(sentence_for_claim("Fifty units are due.", {"claim_text": "Pay 50."}), [])
        self.assertEqual(sentence_for_claim("Fine. Fine.", {"claim_text": "Fine."}), [])

    def test_excluded_evidence_and_diagnosis_do_not_invent_retrieval_failure(self):
        trace = {"generated_answer": "Fine.", "retrieved_chunk_references": [{"chunk_id": "c1"}],
                 "diagnostics": {"retrieval_metadata": {"candidate_chunks": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]},
                 "claim_verification": {"results": [{"claim_id": "claim", "claim_text": "Fine.", "status": "UNSUPPORTED"}]}}}
        result = build_evidence_debugger(trace)
        self.assertEqual([c["chunk_id"] for c in result["excluded_candidates"]], ["c2"])
        self.assertIn("unconfirmed", next(s["message"] for s in result["signals"] if s["kind"] == "retrieval"))
        self.assertEqual(result["sentences"][0]["claims"][0]["claim_id"], "claim")

    def test_legacy_candidates_are_unavailable_and_uncertainty_is_explicit(self):
        value = build_evidence_debugger({"generated_answer": "Answer.", "diagnostics": {}})
        self.assertEqual(value["candidate_capture"], "unavailable")
        self.assertIn("uncertain", [s["kind"] for s in value["signals"]])

    def test_engineer_labels_identify_prompt_evidence_misses_only_for_live_retrieval(self):
        trace = {"generated_answer": "Answer.", "retrieved_chunk_references": [], "diagnostics": {
            "regression_case": {"expected_evidence": ["gold"], "scope": "retrieve"}}}
        value = build_evidence_debugger(trace)
        self.assertIn("gold", next(s["message"] for s in value["signals"] if s["kind"] == "retrieval"))
        trace["diagnostics"]["regression_case"]["scope"] = "generate"
        value = build_evidence_debugger(trace)
        self.assertIn("not measured", next(s["message"] for s in value["signals"] if s["kind"] == "retrieval"))
