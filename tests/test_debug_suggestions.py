from dataclasses import asdict
from unittest.mock import patch

from src.debug_suggestions import suggest_tuning, summarize_chat_tuning
from src.memory.memory_utils import save_json
from src.rag_trace import RAGTrace


def make_trace(trace_id, diagnostics):
    return RAGTrace(
        trace_id=trace_id, trace_version="1", pipeline_version="1", framework_version="1",
        timestamp="", question="Private question", generated_answer="Private answer",
        prompt_snapshot="", prompt_length=0, retrieved_chunk_references=[],
        configuration_snapshot={"arm": "C_hybrid_rerank", "rerank_enabled": True},
        execution_statistics={"retrieval_time": 1.25}, pipeline_stage_status={},
        diagnostics=diagnostics,
    )


def test_suggestions_cover_saved_chats_without_running_model_for_history(tmp_path):
    current = make_trace("current", {"trace_source": "chat"})
    legacy = make_trace("legacy", {"memory_ids": [], "memory_kinds": [], "retrieval_metadata": {}})
    replay = make_trace("replay", {**legacy.diagnostics, "replay": {}})
    benchmark = make_trace("benchmark", {})
    for trace in (current, legacy, replay, benchmark):
        save_json(asdict(trace), tmp_path / "rag_traces" / f"trace_{trace.trace_id}.json")
    save_json(asdict(current), tmp_path / "rag_traces" / "nested" / "trace_current.json")
    save_json({
        "primary_cause": "RETRIEVAL_MISS",
        "immediate_actions": [{
            "title": "Increase Retrieval Top-K", "description": "Try a wider window",
            "success_metric": "Evidence recall@k",
        }],
    }, tmp_path / "corrective_action_plan" / "TRACE_legacy.json")

    proposed = {"title": "RERANKER_TOP_N", "experiment": "Try 6 to 8", "measure": "Evidence recall@k"}
    with patch("src.debug_suggestions._ask_tuning_model", return_value=proposed) as model:
        first = suggest_tuning(current, tmp_path, use_llm=True)
        assert first["source"] == "llm"
        assert suggest_tuning(current, tmp_path, use_llm=True) == first
        summary = summarize_chat_tuning(tmp_path)
        model.assert_called_once()
        assert "BASE_TOP_N" not in model.call_args.args[0]
        assert model.call_args.args[1] == {"retrieval_time": 1.25, "returned_chunks": 0}
    assert summary["trace_count"] == 2
    assert summary["ranked"][0]["trace_count"] == 2
    assert summary["ranked"][0]["evidence_count"] == 2


def test_tuning_model_failure_returns_retryable_trial(tmp_path):
    trace = make_trace("chat", {"trace_source": "chat"})
    with patch("src.debug_suggestions._ask_tuning_model", side_effect=RuntimeError):
        suggestion = suggest_tuning(trace, tmp_path, use_llm=True)
    assert suggestion["source"] == "fallback"
    assert suggestion["title"] == "RERANK_INPUT_SIZE"
    assert not (tmp_path / "debug_suggestions" / "chat.json").exists()
