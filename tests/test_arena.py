from unittest.mock import patch
import pytest
from src.arena import ArenaRequest, paired, run_arena


def test_paired_excludes_missing_judgments_and_keeps_question_pairs():
    rows = [{"source_trace_id": str(i), "results": [{"arm": "a", "supported_fraction": x}, {"arm": "b", "supported_fraction": y}]} for i, (x,y) in enumerate([(0, 1), (.5, 1), (None, 0)])]
    result = paired(rows, ["a", "b"])[0]
    assert result["n"] == 2 and result["mean_difference"] == .75
    assert result["bootstrap_95_percent"] == [.5, 1]


def test_invalid_arms_and_nan_rejected():
    for args in ({"arms": ["bad", "bad"]}, {"arms": ["A_vector", "B_hybrid"], "input_per_million": float("nan")}):
        with pytest.raises(ValueError): ArenaRequest(trace_ids=["x"], **args)


def test_arena_isolates_arm_errors_and_resumes_completed_results():
    request = ArenaRequest(trace_ids=["x"], arms=["A_vector", "B_hybrid"])
    from types import SimpleNamespace
    checkpoint = [{"source_trace_id": "x", "question": "q", "results": [{"arm": "A_vector", "supported_fraction": 1}]}]
    with patch("src.arena.load_trace", return_value=SimpleNamespace(question="q")), patch("src.arena.replay_trace", side_effect=ValueError("incompatible")) as replay:
        result = run_arena(request, checkpoint=checkpoint)
    assert replay.call_count == 1
    assert result["rows"][0]["results"][1]["error"] == "incompatible"
    assert len(checkpoint[0]["results"]) == 1


def test_arena_compares_same_question_and_reports_estimate_scope():
    from types import SimpleNamespace
    source = SimpleNamespace(question="same question")
    trace = {"execution_statistics": {"replay_total_time": 2}, "prompt_snapshot": "x" * 40, "generated_answer": "y" * 20}
    output = {"replay": trace, "comparison": {"scores": {"replay": {"supported_fraction": .5}}}}
    request = ArenaRequest(trace_ids=["x"], arms=["A_vector", "B_hybrid"], input_per_million=2, output_per_million=4)
    with patch("src.arena.load_trace", return_value=source), patch("src.arena.replay_trace", return_value=output) as replay:
        result = run_arena(request)
    assert [call.args[0] for call in replay.call_args_list] == [source, source]
    assert [call.args[1].arm for call in replay.call_args_list] == request.arms
    item = result["rows"][0]["results"][0]
    assert item["estimated_generation_cost_usd"] == .00004
    assert "excludes strategy and judge" in item["cost_scope"]
    assert result["paired"][0]["bootstrap_95_percent"] is None
