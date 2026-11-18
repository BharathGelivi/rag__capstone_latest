"""Trace-based tuning suggestions for the Debug page."""

import json
import logging
import os
from pathlib import Path
import re
from threading import Lock

from configs.pipeline import CHUNK_OVERLAP, RETRIEVAL_TOP_K, RERANK_INPUT_SIZE, RERANKER_TOP_N
from src.memory.memory_utils import save_json
from src.strategy_arms import ARMS, BASE_TOP_N

logger = logging.getLogger(__name__)
llm_lock = Lock()


def _is_chat(diagnostics):
    return "replay" not in diagnostics and (
        diagnostics.get("trace_source") == "chat"
        or {"memory_ids", "memory_kinds", "retrieval_metadata"}.issubset(diagnostics)
    )


def suggest_tuning(trace, artifact_dir: Path, *, use_llm=False):
    """Return one experiment using existing diagnostics, or ask the tuning model."""
    config = trace.configuration_snapshot or {}
    stats = trace.execution_statistics or {}
    diagnostics = trace.diagnostics or {}
    retrieval = diagnostics.get("retrieval_metadata") or {}
    arm = config.get("arm")
    rerank = config.get("reranker_enabled") or config.get("rerank_enabled")
    combined = config.get("corpus") == "both"
    is_direct_chat = arm == "C_hybrid_rerank" and _is_chat(diagnostics)
    if combined:
        top_setting, top_value = "RETRIEVAL_TOP_K", RETRIEVAL_TOP_K
    elif not is_direct_chat and ARMS.get(arm, {}).get("strategy") in ("plain", "ircot"):
        top_setting, top_value = "BASE_TOP_N", BASE_TOP_N
    elif rerank:
        top_setting = "RERANKER_TOP_N"
        top_value = retrieval.get("reranker_top_n", config.get("reranker_top_n", RERANKER_TOP_N))
    else:
        top_setting, top_value = "RETRIEVAL_TOP_K", config.get("retrieval_top_k", RETRIEVAL_TOP_K)
    settings = {top_setting: top_value, "CHUNK_OVERLAP": config.get("chunk_overlap", CHUNK_OVERLAP)}
    if rerank:
        settings["RERANK_INPUT_SIZE"] = retrieval.get("rerank_input_size", RERANK_INPUT_SIZE)

    plan_path = artifact_dir / "corrective_action_plan" / f"TRACE_{trace.trace_id}.json"
    plan = _read_object(plan_path)
    if plan.get("primary_cause") and plan["primary_cause"] != "UNKNOWN":
        for tier in ("immediate_actions", "short_term_actions", "experimental_actions"):
            for action in plan.get(tier) or []:
                if not isinstance(action, dict) or not all(
                    isinstance(action.get(key), str) and action[key].strip()
                    for key in ("title", "description", "success_metric")
                ):
                    continue
                return {
                    "title": top_setting if action["title"] == "Increase Retrieval Top-K" else action["title"],
                    "experiment": action["description"],
                    "measure": action["success_metric"],
                    "source": "diagnostic",
                }

    # Only operational metadata goes to the tuning model. Keep document passages,
    # chat messages and generated answers out of this request.
    signals = {
        key: stats[key] for key in (
            "retrieval_time", "generation_time", "pre_rerank_candidate_pool_size",
            "pre_rerank_min_dense_distance",
        ) if isinstance(stats.get(key), (int, float))
    }
    signals["returned_chunks"] = len(trace.retrieved_chunk_references)
    verification = diagnostics.get("claim_verification") or {}
    results = verification.get("results") or []
    if results:
        signals["checked_claims"] = len(results)
        signals["unsupported_or_contradicted_claims"] = sum(
            item.get("status") in ("UNSUPPORTED", "CONTRADICTED") for item in results
        )
    settings = {key: value for key, value in settings.items() if type(value) is int and value > 0}
    cache_path = artifact_dir / "debug_suggestions" / f"{trace.trace_id}.json"
    cached = _read_object(cache_path)
    if cached:
        return cached
    if use_llm:
        with llm_lock:
            cached = _read_object(cache_path)
            if cached:
                return cached
            try:
                suggestion = {**_ask_tuning_model(settings, signals), "source": "llm"}
            except Exception as exc:
                logger.warning("Debug tuning model unavailable (%s)", type(exc).__name__)
            else:
                try:
                    save_json(suggestion, str(cache_path))
                except OSError:
                    logger.warning("Could not save Debug tuning suggestion")
                return suggestion
    setting = "RERANK_INPUT_SIZE" if "RERANK_INPUT_SIZE" in settings else top_setting
    current = settings.get(setting)
    target = f" from {current} to {current + max(1, current // 4)}" if current is not None else ""
    return {
        "title": setting,
        "experiment": f"Compare the baseline with a trial increase in {setting}{target}.",
        "measure": "Compare evidence recall@k and retrieval latency on the same questions.",
        "source": "fallback" if use_llm else "trace_rule",
    }


def _read_object(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def summarize_chat_tuning(artifact_dir: Path, focused=None, suggestion=None):
    from src.rag_trace import RAGTrace

    root = artifact_dir / "rag_traces"
    paths = sorted(set(root.glob("*/trace_*.json")) | set(root.glob("*.json")))
    seen = set()
    groups = {}
    for path in paths:
        data = _read_object(path)
        trace_id = data.get("trace_id")
        if not isinstance(trace_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id) or trace_id in seen:
            continue
        diagnostics = data.get("diagnostics") or {}
        # Older chat traces are recognizable by the memory-provenance fields.
        if not _is_chat(diagnostics):
            continue
        try:
            trace = RAGTrace.from_json(data)
        except (TypeError, ValueError):
            continue
        seen.add(trace_id)
        item = suggestion if trace_id == focused and suggestion else suggest_tuning(trace, artifact_dir)
        title = item["title"]
        group = groups.setdefault(title, {"suggestion": item, "trace_count": 0, "evidence_count": 0})
        group["trace_count"] += 1
        if item["source"] in ("diagnostic", "llm"):
            group["evidence_count"] += 1
            if group["suggestion"]["source"] not in ("diagnostic", "llm"):
                group["suggestion"] = item
    ranked = sorted(groups.values(), key=lambda row: (
        -row["evidence_count"], -row["trace_count"], row["suggestion"]["title"],
    ))
    return {"trace_count": len(seen), "ranked": ranked}


def _ask_tuning_model(settings, signals):
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("Tuning model credentials unavailable")
    from llama_index.core.llms import ChatMessage, MessageRole
    from src import rate_limiter
    from src.generator import Generator

    response = rate_limiter.call(Generator(temperature=0.0, max_tokens=350).llm.chat, [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Suggest one controlled RAG tuning experiment using the provided telemetry. "
                "Return only a JSON object with title, experiment and measure string fields. "
                "The title must be exactly one supplied setting name. In experiment, propose a "
                "small test change and explain why it is worth testing. In measure, state how to "
                "compare evidence coverage and cost or latency against baseline. "
                "Treat the change as a hypothesis. Do not claim a failure or outcome was proven."
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=json.dumps({"settings": settings, "telemetry": signals})),
    ])
    suggestion = json.loads(str(response.message.content).strip())
    if (
        set(suggestion) != {"title", "experiment", "measure"}
        or suggestion["title"] not in settings
        or any(not isinstance(value, str) or not 0 < len(value.strip()) <= 500 for value in suggestion.values())
    ):
        raise ValueError("Invalid tuning suggestion")
    return suggestion
