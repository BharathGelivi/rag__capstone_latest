"""
FastAPI backend for the React UI: chat (SSE), graphs, sessions.

Thin transport over existing implementations -- chat_service.run_chat_turn,
ui/components/graph.py's build_graph_data, src/memory_graph.py's
build_memory_graph_data, and MemoryManager. No pipeline logic lives here.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import chat_service
from src.chat_evaluation import evaluate_answer
from src.research_dashboard import experiment_directory, research_summary
from src.device import describe_device
from src.debug_suggestions import suggest_tuning, summarize_chat_tuning
from src.memory_graph import build_memory_graph_data
from src.trace_metrics import compute_framework_panel
from src.trace_replay import ReplayOptions, captured_messages, editable_prompt, load_trace, replay_trace, trace_choices
from src.regression_cases import Expectations, capture_case, evaluate_case, load_cases, run_case, save_case
from src.session_report import save_session_report
from ui.components.graph import build_graph_data, build_legal_graph_data

app = FastAPI(title="X-RAG UI backend")
session_report_lock = Lock()
from src.arena import router as arena_router
app.include_router(arena_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    arm: str = "C_hybrid_rerank"
    corpus: str = "statutes"
    chat_history: List[Dict[str, str]] = Field(default_factory=list, max_length=100)
    memory_enabled: bool = True
    deep_analysis: bool = False

    @model_validator(mode="after")
    def validate_options(self):
        from src.strategy_arms import ARMS
        if not self.question.strip():
            raise ValueError("question must not be blank")
        if self.corpus not in chat_service.CORPUS_OPTIONS or self.arm not in ARMS:
            raise ValueError("Unknown corpus or retrieval arm")
        if self.corpus == "both" and self.arm != "C_hybrid_rerank":
            raise ValueError("Combined corpus requires C_hybrid_rerank")
        if self.corpus == "statutes" and (ARMS[self.arm].get("graph") or ARMS[self.arm].get("graph_expand")
                                         or self.arm == "C_fixed_chunking"):
            raise ValueError("This arm requires the judgments corpus")
        if any(t.get("role") not in ("user", "assistant") or not isinstance(t.get("content"), str)
               or len(t["content"]) > 20000 for t in self.chat_history):
            raise ValueError("Invalid conversation history")
        return self


class SessionCreateRequest(BaseModel):
    title: str = "New Session"


@app.get("/ui/config")
def get_config():
    from src.strategy_arms import ARMS

    return {
        "arms": list(ARMS.keys()),
        "corpora": chat_service.CORPUS_OPTIONS,
        "device": describe_device(),
        "arms_by_corpus": {c: [a for a, options in ARMS.items()
                              if (c != "both" or a == "C_hybrid_rerank")
                              and (c != "statutes" or not (options.get("graph") or options.get("graph_expand") or a == "C_fixed_chunking"))]
                           for c in chat_service.CORPUS_OPTIONS},
    }


@app.post("/ui/chat")
def chat(req: ChatRequest):
    """SSE stream. Declared as a sync `def` (not `async def`) so the blocking
    pipeline runs in Starlette's threadpool instead of the event loop --
    otherwise one slow turn would freeze every other route.
    """
    def sse_events():
        for event in chat_service.run_chat_turn(
            question=req.question,
            session_id=req.session_id,
            arm=req.arm,
            corpus=req.corpus,
            chat_history=req.chat_history,
            memory_enabled=req.memory_enabled,
            deep_analysis=req.deep_analysis,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse_events(), media_type="text/event-stream")


@app.get("/ui/graph")
def graph(scope: str = "rag", session_id: Optional[str] = None, max_nodes: int = Query(150, ge=1, le=1000)):
    if scope == "memory":
        return build_memory_graph_data(max_observations=max_nodes)

    if scope == "legal":
        return build_legal_graph_data(chat_service.load_knowledge_graph(), max_nodes=max_nodes)

    if scope != "rag":
        raise HTTPException(status_code=400, detail="scope must be 'rag', 'legal' or 'memory'")

    from src.chunk_registry import ChunkRegistry

    registry_path = os.path.join(PROJECT_ROOT, "artifacts", "chunk_registry.json")
    registry = ChunkRegistry.load_from_json(registry_path) if os.path.exists(registry_path) else None

    mm = chat_service.get_memory_manager()
    sessions = mm.list_sessions()
    memories = mm.get_session_memories(session_id) if session_id else [
        m for s in sessions for m in mm.get_session_memories(s.session_id)
    ]

    return build_graph_data(
        registry=registry, memories=memories, sessions=sessions,
        last_result=None, max_chunks=max_nodes,
    )


@app.get("/ui/sessions")
def list_sessions():
    mm = chat_service.get_memory_manager()
    return [s.to_dict() for s in mm.list_sessions()]


@app.post("/ui/sessions")
def create_session(req: SessionCreateRequest):
    mm = chat_service.get_memory_manager()
    return mm.create_session(req.title).to_dict()


class SessionRenameRequest(BaseModel):
    title: str


@app.patch("/ui/sessions/{session_id}")
def rename_session(session_id: str, req: SessionRenameRequest):
    mm = chat_service.get_memory_manager()
    if not mm.rename_session(session_id, req.title):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "title": req.title}


@app.get("/ui/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """Reload a session's turns for the message list.

    Only ``role``/``content`` survive a turn's live SSE stream into permanent
    storage -- the chunks, timings and eval scores the UI showed while the
    turn was in flight only ever existed in the frontend's in-memory state,
    so a page reload or navigating away from Chat and back used to make a
    turn's Details panel vanish for good even though the answer's own
    citations were still sitting right there in the text. ``trace_id`` is
    the one durable link back to that data (see ``/ui/trace/{trace_id}``,
    which reads the full RAGTrace JSON saved alongside it) -- returning it
    here lets the frontend re-fetch and rebuild the Details panel on demand.
    """
    mm = chat_service.get_memory_manager()
    memories = mm.get_session_memories(session_id)
    messages = []
    for mem in memories:
        messages.append({"role": "user", "content": mem.question})
        messages.append({
            "role": "assistant",
            "content": mem.answer,
            "trace_id": mem.trace_id or None,
        })
    return messages


@app.delete("/ui/sessions/{session_id}")
def delete_session(session_id: str):
    mm = chat_service.get_memory_manager()
    try:
        deleted = mm.delete_session(session_id)
    except Exception as exc:
        logging.getLogger(__name__).exception("Session deletion failed")
        raise HTTPException(503, "Session deletion could not complete. Please retry.") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.get("/ui/memory")
def list_memory(search: Optional[str] = None, session_id: Optional[str] = None, limit: int = Query(30, ge=1, le=1000)):
    """Memory page data. With ``search``, ranks via MemoryManager.search_memory
    (semantic/recency/frequency/importance all populated). Without it, this is
    a plain recency listing -- those per-query scores don't exist outside a
    search, so only ``importance_score`` (intrinsic to the memory) is set and
    the rest come back 0.
    """
    mm = chat_service.get_memory_manager()

    if search:
        results = mm.search_memory(search, top_k=limit, session_id=session_id)
        return {"memories": [r.to_dict() for r in results], "searched": True}

    memories = (
        mm.get_session_memories(session_id) if session_id
        else mm.store.get_all_memories(limit=limit)
    )
    memories = sorted(memories, key=lambda m: m.timestamp, reverse=True)[:limit]
    return {
        "memories": [
            {
                "memory": m.to_dict(),
                "semantic_score": 0.0, "recency_score": 0.0, "frequency_score": 0.0,
                "importance_score": m.importance_score, "final_score": 0.0,
                "retrieval_reason": "",
            }
            for m in memories
        ],
        "searched": False,
    }


@app.get("/ui/memory/health")
def memory_health():
    """Return memory persistence health without mutating the store."""
    return chat_service.get_memory_manager().health()


@app.get("/ui/trace/latest")
def latest_trace():
    trace = _read_latest_trace_file()
    if trace is None:
        raise HTTPException(status_code=404, detail="No traces recorded yet")
    return trace


class DebugSuggestionRequest(BaseModel):
    trace_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")


@app.post("/ui/debug/suggestions")
def debug_suggestions(request: DebugSuggestionRequest):
    artifact_dir = Path(PROJECT_ROOT) / "artifacts"
    suggestion = None
    if request.trace_id:
        try:
            trace = load_trace(request.trace_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Trace not found") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "Could not read the saved trace") from exc
        suggestion = suggest_tuning(trace, artifact_dir, use_llm=True)
    return {
        "current": suggestion,
        **summarize_chat_tuning(artifact_dir, request.trace_id, suggestion),
    }


@app.get("/ui/trace/{trace_id}")
def get_trace(trace_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id):
        raise HTTPException(status_code=422, detail="Invalid trace identifier")
    trace_dir = os.path.join(PROJECT_ROOT, "artifacts", "rag_traces")
    matches = glob.glob(os.path.join(trace_dir, "*", f"trace_{trace_id}.json"))
    matches += [p for p in (os.path.join(trace_dir, f"{trace_id}.json"),
                            os.path.join(trace_dir, f"trace_{trace_id}.json")) if os.path.isfile(p)]
    if not matches:
        raise HTTPException(status_code=404, detail="Trace not found")
    with open(matches[0], "r", encoding="utf-8") as f:
        trace = json.load(f)
    diagnostics = trace.get("diagnostics") or {}
    # Legacy UI wire fields are maintained while storage uses canonical diagnostics.
    for key in ("claim_verification", "claim_error"):
        if key in diagnostics:
            trace[key] = diagnostics[key]

    # New traces capture full evidence; backfill previews only for legacy
    # traces so a current registry can never overwrite captured text.
    refs = trace.get("retrieved_chunk_references") or []
    # Recover historical scores from captured evidence before adding previews
    # from current registries, which may have changed since the answer.
    snapshot = dict(re.findall(
        r"--- Context chunk \d+ \[Chunk-ID: ([^\]]*)\] ---\n(.*?)(?=\n--- Context chunk |\n\nQuestion: |\Z)",
        trace.get("prompt_snapshot") or "", re.DOTALL)) if any(not ref.get("text") for ref in refs) else {}
    evaluation_refs = [
        {**ref, "text": ref.get("text") or ref.get("chunk_text") or snapshot.get(ref.get("chunk_id"), "").strip()}
        for ref in refs
    ]
    trace["evaluation"] = {
        **evaluate_answer(trace.get("generated_answer") or "", evaluation_refs, trace.get("claim_verification")),
        **(diagnostics.get("evaluation") or {}),
    }
    if refs:
        for ref in refs:
            if not ref.get("text") and ref.get("chunk_id") in snapshot:
                ref["text"] = snapshot[ref["chunk_id"]].strip()[:400]
        corpus = (trace.get("configuration_snapshot") or {}).get("corpus")
        corpora = [corpus] if corpus in chat_service.CORPORA else chat_service.CORPORA
        for corpus in corpora:
            if all(ref.get("text") for ref in refs):
                break
            registry = chat_service.load_registry(corpus)
            if registry is None:
                continue
            for ref in refs:
                if not ref.get("text"):
                    record = registry.get_chunk(ref["chunk_id"])
                    if record:
                        ref["text"] = record.text[:400]
        for ref in refs:
            ref.setdefault("text", "")

    return trace


@app.get("/ui/trace/{trace_id}/metrics")
def trace_metrics(trace_id: str):
    """On-demand framework comparison panel for one saved turn.

    Computes the 13-metric panel (RAGAS-/RAGChecker-/ARES-style families,
    each with per-metric provenance) for the saved trace, running claim
    decomposition + NLI verification on demand if the turn wasn't produced
    with deep analysis. Results are cached per trace id -- each call after
    the first is free.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id):
        raise HTTPException(422, "Invalid trace identifier")
    panel = compute_framework_panel(trace_id)
    if panel is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return panel


@app.get("/ui/traces")
def saved_traces(limit: int = Query(50, ge=1, le=100)):
    return trace_choices(limit)


class SaveRegressionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    expectations: Expectations


@app.post("/ui/trace/{trace_id}/regression-cases")
def save_regression_case(trace_id: str, request: SaveRegressionRequest):
    try:
        case = capture_case(load_trace(trace_id), request.name, request.expectations)
        directory = os.path.join(PROJECT_ROOT, "eval", "regression_cases")
        save_case(case, directory)
        return {"case_id": case.case_id, "path": f"eval/regression_cases/{case.case_id}.json",
                "baseline": evaluate_case(case, case.snapshot)}
    except FileNotFoundError as exc:
        raise HTTPException(404, "Trace not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/ui/regression-cases")
def regression_case_list():
    return [{"case_id": c.case_id, "name": c.name, "source_trace_id": c.source_trace_id,
             "expectations": c.expectations.model_dump()} for c in
            load_cases(os.path.join(PROJECT_ROOT, "eval", "regression_cases"))]


class RunRegressionRequest(BaseModel):
    mode: str = Field(pattern=r"^(generate|retrieve|snapshot)$", default="generate")
    case_ids: List[str] = Field(min_length=1, max_length=20)


@app.post("/ui/regression-cases/run")
def run_regression_suite(request: RunRegressionRequest):
    cases = {c.case_id: c for c in load_cases(os.path.join(PROJECT_ROOT, "eval", "regression_cases"))}
    if any(cid not in cases for cid in request.case_ids):
        raise HTTPException(404, "Regression case not found")
    results = [run_case(cases[cid], request.mode) for cid in dict.fromkeys(request.case_ids)]
    return {"mode": request.mode, "passed": all(r["passed"] for r in results), "cases": results}


def _evidence_view(trace_id):
    from dataclasses import asdict
    from src.trace_replay import captured_chunks
    trace = load_trace(trace_id)
    value = asdict(trace)
    try:
        texts = {c.chunk_id: c.chunk_text for c in captured_chunks(trace)}
        for ref in value["retrieved_chunk_references"]:
            ref["text"] = texts[ref["chunk_id"]]
    except ValueError:
        pass
    return value


@app.get("/ui/trace/{trace_id}/evidence")
def evidence_debugger(trace_id: str):
    from src.evidence_debugger import build_evidence_debugger
    try:
        value = _evidence_view(trace_id)
        result = build_evidence_debugger(value)
        result["sources"] = value["retrieved_chunk_references"] + [
            {**m, "text": m.get("chunk_text", "")} for m in (value.get("diagnostics") or {}).get("memory_evidence", [])]
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, "Trace not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/ui/trace/{trace_id}/evidence/analyze")
def analyze_evidence(trace_id: str):
    from src.trace_replay import captured_chunks
    try:
        trace = load_trace(trace_id)
        memory = chat_service.captured_memory_chunks(trace)
        outcome = chat_service.deep_analyze(trace, captured_chunks(trace) + memory)
        if "error" in outcome:
            raise HTTPException(502, outcome["error"])
        # Save only analysis into the same recorded trace file, not a new run.
        root = os.path.join(PROJECT_ROOT, "artifacts", "rag_traces")
        paths = glob.glob(os.path.join(root, "*", f"trace_{trace_id}.json"))
        paths += [p for p in (os.path.join(root, f"{trace_id}.json"), os.path.join(root, f"trace_{trace_id}.json")) if os.path.isfile(p)]
        if not chat_service.persist_deep_analysis(paths[0], outcome):
            raise HTTPException(500, "Analysis completed but could not be saved. Please retry.")
        from src.trace_metrics import invalidate_framework_panel
        invalidate_framework_panel(trace_id)
        return evidence_debugger(trace_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Trace not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/ui/trace/{trace_id}/source/{chunk_id}")
def evidence_source(trace_id: str, chunk_id: str):
    from src.evidence_sources import resolve_source
    try:
        trace = load_trace(trace_id)
        refs = trace.retrieved_chunk_references + ((trace.diagnostics or {}).get("retrieval_metadata") or {}).get("candidate_chunks", [])
        ref = next((c for c in refs if c["chunk_id"] == chunk_id), None)
        path = resolve_source(PROJECT_ROOT, ref.get("source_file", "")) if ref else None
        if path is None:
            raise HTTPException(404, "Source PDF is unavailable; use the captured passage")
        return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")
    except FileNotFoundError as exc:
        raise HTTPException(404, "Trace not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/ui/trace/{trace_id}/replay")
def replay_settings(trace_id: str):
    try:
        trace = load_trace(trace_id)
        return {"system_prompt": editable_prompt(captured_messages(trace)),
                "model": trace.configuration_snapshot.get("llm_model", ""),
                "arm": trace.configuration_snapshot.get("arm", "C_hybrid_rerank"),
                "corpus": trace.configuration_snapshot.get("corpus", ""),
                "top_k": len(trace.retrieved_chunk_references) or 6}
    except FileNotFoundError as exc:
        raise HTTPException(404, "Trace not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/ui/trace/{trace_id}/replay")
def replay_saved_trace(trace_id: str, options: ReplayOptions):
    try:
        return replay_trace(load_trace(trace_id), options)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Trace not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        import logging
        import uuid
        reference = uuid.uuid4().hex
        logging.getLogger(__name__).exception("Trace replay failed [ref=%s]", reference)
        raise HTTPException(502, f"Replay could not complete. Please retry. Reference: {reference}") from exc


@app.get("/ui/reports")
def list_diagnostic_reports(limit: int = 50):
    """List generated HTML diagnostic reports for the report browser."""
    report_dir = os.path.join(PROJECT_ROOT, "artifacts", "diagnostic_reports")
    if not os.path.isdir(report_dir):
        return {"reports": []}
    paths = sorted(glob.glob(os.path.join(report_dir, "*.html")),
                   key=os.path.getmtime, reverse=True)[:max(1, min(limit, 200))]
    return {"reports": [{
        "name": os.path.basename(path),
        "trace_id": os.path.splitext(os.path.basename(path))[0],
        "session_id": os.path.basename(path).removeprefix("session.")[:-5]
        if os.path.basename(path).startswith("session.") else None,
        "size_bytes": os.path.getsize(path),
        "modified_at": os.path.getmtime(path),
        "url": f"/ui/reports/{os.path.basename(path)}",
    } for path in paths]}


@app.post("/ui/sessions/{session_id}/report")
def create_session_report(session_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        raise HTTPException(404, "Session not found")
    mm = chat_service.get_memory_manager()
    session = next((entry for entry in mm.list_sessions() if entry.session_id == session_id), None)
    if session is None:
        raise HTTPException(404, "Session not found")
    with session_report_lock:
        turns = [entry for entry in mm.get_session_memories(session_id) if entry.kind == "turn"]
        if not turns:
            raise HTTPException(409, "This session has no saved turns to report.")
        path = save_session_report(
            Path(PROJECT_ROOT) / "artifacts" / "diagnostic_reports", session, turns, load_trace
        )
    return {"url": f"/ui/reports/{path.name}"}


@app.get("/ui/reports/{name}")
def download_diagnostic_report(name: str):
    """Download one report while rejecting traversal and non-HTML files."""
    report_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "artifacts", "diagnostic_reports"))
    candidate = os.path.abspath(os.path.join(report_dir, name))
    if os.path.dirname(candidate) != report_dir or not candidate.endswith(".html") or not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="Diagnostic report not found")
    return FileResponse(
        candidate, media_type="text/html", content_disposition_type="inline",
        headers={"Cache-Control": "no-store"} if name.startswith("session.") else None,
    )


@app.get("/ui/experiments")
def list_experiment_runs(source: str = "active"):
    """Return ordered experiment status and the latest reproducibility envelope."""
    base_dir = _experiment_directory(source)
    if not os.path.isdir(base_dir):
        return {"run": None, "experiments": []}
    try:
        from experiments import run_all
        experiments = run_all.load_experiments()
        statuses = [run_all.experiment_status(experiment, base_dir) for experiment in experiments]
        manifest = run_all.load_manifest(base_dir)
        for status in statuses:
            saved = manifest.get("experiments", {}).get(status["key"], {})
            if saved.get("last_error"):
                status["status"] = "failed"
                status["last_error"] = saved["last_error"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read experiment status: {exc}") from exc
    run = None
    manifest_path = os.path.join(base_dir, run_all.RUN_MANIFEST_NAME)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                run = json.load(handle)
        except (OSError, json.JSONDecodeError):
            run = None
    monitor = None
    monitor_path = os.path.join(base_dir, "monitor.json")
    if os.path.isfile(monitor_path):
        try:
            with open(monitor_path, encoding="utf-8-sig") as handle:
                monitor = json.load(handle)
        except (OSError, json.JSONDecodeError):
            monitor = {"status": "unreadable"}
            logging.getLogger(__name__).warning("Experiment monitor could not be read")
    return {"run": run, "experiments": statuses, "monitor": monitor}


def _experiment_directory(source: str):
    try:
        return str(experiment_directory(PROJECT_ROOT, source))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/ui/research-summary")
def get_research_summary(source: str = "active"):
    try:
        return research_summary(PROJECT_ROOT, source)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/ui/experiments/{key}")
def experiment_detail(key: str, source: str = "active"):
    from experiments import run_all
    experiment = next((e for e in run_all.load_experiments() if e.key == key), None)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    directory = os.path.join(_experiment_directory(source), key)
    summary = None
    records = []
    try:
        path = os.path.join(directory, "summary.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                summary = json.load(handle, parse_constant=lambda value: None)
        path = os.path.join(directory, "records.jsonl")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line, parse_constant=lambda value: None))
                    if len(records) >= 100:
                        break
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Experiment artifacts could not be read") from exc
    return {"key": key, "title": experiment.title, "summary": summary, "records": records}


def _read_latest_trace_file() -> Optional[Dict[str, Any]]:
    trace_dir = os.path.join(PROJECT_ROOT, "artifacts", "rag_traces")
    files = glob.glob(os.path.join(trace_dir, "*", "trace_*.json"))
    files += glob.glob(os.path.join(trace_dir, "*.json"))
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)

