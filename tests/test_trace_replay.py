from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from llama_index.core.llms import ChatMessage
from pydantic import ValidationError

from src.api_ui import app
from src.generator import PromptBuilder
from src.rag_trace import RAGTraceBuilder
from src.trace_replay import ReplayOptions, captured_messages, load_trace, replay_trace
from tests.test_rag_trace import make_generation_result, make_retrieval_result


def source_trace(structured=True):
    retrieval = make_retrieval_result()
    retrieval.retrieval_metadata.update(corpus="statutes", arm="C_hybrid_rerank")
    generation = make_generation_result()
    messages = PromptBuilder.build_messages(generation.question, retrieval.retrieved_chunks,
                                            system_instructions="Use only evidence. Remember captured fact.",
                                            chat_history=[{"role": "user", "content": "Earlier question"}])
    generation.prompt = "\n".join(f"[{m.role.value.upper()}]: {m.content}" for m in messages)
    if structured:
        generation.generation_metadata["messages"] = [{"role": m.role.value, "content": m.content} for m in messages]
    return RAGTraceBuilder.build(retrieval, generation, .8)


class TestTraceReplay(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = patch("src.chat_service.PROJECT_ROOT", self.directory.name)
        root.start(); self.addCleanup(root.stop)
        llm = patch("src.generator.OpenAILike")
        self.llm = llm.start(); self.addCleanup(llm.stop)
        self.llm.return_value.chat.return_value = SimpleNamespace(
            message=ChatMessage(role="assistant", content="Replay answer."),
            raw={"choices": [{"finish_reason": "stop"}]})
        limiter = patch("src.generator.rate_limiter.call", side_effect=lambda fn, *a, **kw: fn(*a, **kw))
        limiter.start(); self.addCleanup(limiter.stop)

    def test_frozen_replay_preserves_evidence_history_and_original_file(self):
        source = source_trace()
        source_path = RAGTraceBuilder.save_to_json(source, str(Path(self.directory.name)/"artifacts/rag_traces"))
        original_bytes = Path(source_path).read_bytes()
        with patch("src.chat_service.load_pipeline", side_effect=AssertionError("fresh retrieval")), \
             patch("src.chat_service.get_memory_manager", side_effect=AssertionError("mutable memory")):
            result = replay_trace(load_trace(source.trace_id), ReplayOptions(verify_claims=False))
        self.assertNotEqual(result["replay"]["trace_id"], source.trace_id)
        self.assertEqual(original_bytes, Path(source_path).read_bytes())
        sent = self.llm.return_value.chat.call_args.args[0]
        self.assertEqual([m.model_dump() for m in sent], [m.model_dump() for m in captured_messages(source)])
        self.assertEqual(result["comparison"]["evidence_added"], [])
        self.assertFalse(result["comparison"]["evidence_order_changed"])
        self.assertEqual(result["replay"]["execution_statistics"]["retrieval_time"], 0)
        child = load_trace(result["replay"]["trace_id"])
        self.assertEqual(child.diagnostics["replay"]["source_trace_id"], source.trace_id)
        self.assertIsNone(result["comparison"]["scores"]["replay"]["supported_fraction"])

    def test_model_and_prompt_overrides_preserve_other_messages(self):
        source = source_trace()
        result = replay_trace(source, ReplayOptions(model="test/other", system_prompt="New instructions", verify_claims=False))
        self.assertEqual(self.llm.call_args.kwargs["model"], "test/other")
        messages = self.llm.return_value.chat.call_args.args[0]
        self.assertEqual(messages[1].content, "New instructions")
        self.assertEqual(messages[-1].content, captured_messages(source)[-1].content)
        self.assertEqual(result["replay"]["configuration_snapshot"]["llm_temperature"], source.configuration_snapshot["llm_temperature"])

    def test_legacy_prompt_recovers_full_evidence_without_registry(self):
        source = source_trace(False)
        for ref in source.retrieved_chunk_references:
            ref.pop("text")
        result = replay_trace(source, ReplayOptions(verify_claims=False))
        self.assertEqual(result["original"]["retrieved_chunk_references"][0]["text"], "Text for c1.")
        self.assertIn("Legacy prompt", " ".join(result["warnings"]))

    def test_incomplete_frozen_evidence_is_rejected_before_inference(self):
        source = source_trace()
        source.retrieved_chunk_references[0].pop("text")
        source.prompt_snapshot = "[USER]: Q"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            replay_trace(source, ReplayOptions(verify_claims=False))
        self.llm.assert_not_called()

    def test_fresh_retrieval_uses_requested_context_limit_and_diffs_ids(self):
        source = source_trace()
        retrieval = make_retrieval_result(("new",))
        retriever = MagicMock()
        retriever.retrieve.return_value = retrieval
        with patch("src.chat_service.load_pipeline", return_value=(retriever, MagicMock(), MagicMock(), None)):
            result = replay_trace(source, ReplayOptions(mode="retrieve", arm="A_vector", top_k=1, verify_claims=False))
        retriever.retrieve.assert_called_once_with(source.question, mode="vector", rerank=False, top_n=1)
        self.assertEqual(result["comparison"]["evidence_added"], ["new"])
        self.assertEqual(result["comparison"]["evidence_removed"], ["c1", "c2"])
        self.assertIn("Text for new.", self.llm.return_value.chat.call_args.args[0][-1].content)

    def test_failed_generation_does_not_save_a_child_trace(self):
        with patch("src.generator.Generator.generate", return_value=SimpleNamespace(ok=False)), \
             patch("src.trace_replay.RAGTraceBuilder.save_to_json") as save:
            with self.assertRaises(RuntimeError):
                replay_trace(source_trace(), ReplayOptions())
            save.assert_not_called()

    def test_verification_results_are_persisted_and_compared(self):
        source = source_trace()
        source.diagnostics["claim_verification"] = {"claim_count": 1, "results": [
            {"status": "UNSUPPORTED", "claim_text": "Replay answer."}]}
        payload = {"claim_count": 1, "results": [{"status": "SUPPORTED", "claim_text": "Replay answer."}]}
        with patch("src.chat_service.deep_analyze", return_value={"verification": payload}) as analyze:
            result = replay_trace(source, ReplayOptions())
        self.assertEqual(result["comparison"]["scores"]["replay"]["supported_fraction"], 1)
        self.assertEqual(load_trace(result["replay"]["trace_id"]).diagnostics["claim_verification"], payload)
        self.assertEqual(len(analyze.call_args.args[1]), 2)
        self.assertEqual(result["comparison"]["claim_changes"], [{"claim_text": "Replay answer.",
                          "original_status": "UNSUPPORTED", "replay_status": "SUPPORTED"}])

    def test_options_and_api_validate_before_execution(self):
        for value in ({"top_k": 0}, {"mode": "frozen", "arm": "A_vector"},
                      {"mode": "retrieve", "top_k": 1.5}, {"model": "http://bad url"}, {"system_prompt": "  "}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ReplayOptions(**value)
        client = TestClient(app)
        self.assertEqual(client.post("/ui/trace/missing/replay", json={}).status_code, 404)
        self.assertEqual(client.post("/ui/trace/bad*id/replay", json={}).status_code, 422)
        self.assertEqual(client.post("/ui/trace/missing/replay", json={"top_k": 0}).status_code, 422)

    def test_invalid_corpus_strategy_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "judgments"):
            replay_trace(source_trace(), ReplayOptions(mode="retrieve", arm="F_graphrag", corpus="statutes"))
        self.llm.assert_not_called()

    def test_streaming_replay_preserves_answer_extraction(self):
        source = source_trace()
        source.diagnostics["generation_metadata"].update(streamed=True, reasoning_mode=True)
        result = make_generation_result()
        result.generated_answer = "Extracted answer"
        def stream(generator, retrieval, **kwargs):
            self.assertTrue(kwargs["reasoning"])
            self.assertEqual(kwargs["messages_override"][-1].content, captured_messages(source)[-1].content)
            generator.last_stream_result = result
            yield ("answer", "Extracted answer")
        with patch("src.generator.Generator.generate_stream", stream):
            value = replay_trace(source, ReplayOptions(verify_claims=False))
        self.assertEqual(value["replay"]["generated_answer"], "Extracted answer")
