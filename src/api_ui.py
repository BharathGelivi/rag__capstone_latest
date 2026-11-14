"""
FastAPI backend for the React UI: chat (SSE), graphs, sessions.

Thin transport over existing implementations -- chat_service.run_chat_turn,
ui/components/graph.py's build_graph_data, src/memory_graph.py's
build_memory_graph_data, and MemoryManager. No pipeline logic lives here.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import chat_service
from src.device import describe_device
from src.memory_graph import build_memory_graph_data
from src.trace_metrics import compute_framework_panel
from ui.components.graph import build_graph_data, build_legal_graph_data

app = FastAPI(title="X-RAG UI backend")

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
    question: str
    session_id: str
    arm: str = "C_hybrid_rerank"
    corpus: str = "statutes"
    chat_history: List[Dict[str, str]] = []
    memory_enabled: bool = True
    deep_analysis: bool = False


class SessionCreateRequest(BaseModel):
    title: str = "New Session"


@app.get("/ui/config")
def get_config():
    from src.strategy_arms import ARMS

    return {
        "arms": list(ARMS.keys()),
        "corpora": chat_service.CORPUS_OPTIONS,
        # Actual device string (was previously hardcoded in the UI as
        # "RTX 4060" on every machine).
        "device": describe_device(),
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
def graph(scope: str = "rag", session_id: Optional[str] = None, max_nodes: int = 150):
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
    if not mm.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.get("/ui/memory")
def list_memory(search: Optional[str] = None, session_id: Optional[str] = None, limit: int = 30):
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


@app.get("/ui/trace/{trace_id}")
def get_trace(trace_id: str):
    trace_dir = os.path.join(PROJECT_ROOT, "artifacts", "rag_traces")
    matches = glob.glob(os.path.join(trace_dir, "*", f"trace_{trace_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="Trace not found")
    with open(matches[0], "r", encoding="utf-8") as f:
        trace = json.load(f)

    # RAGTraceBuilder records chunk_id/scores/provenance but never the chunk
    # text itself (keeps trace files small) -- the live SSE `chunks` event is
    # the only place that ever carried it, so a re-fetched trace would show
    # empty chunk cards. The trace doesn't record which corpus it ran
    # against either, so check both registries; a chunk_id collision across
    # corpora is not realistic (they're per-ingestion UUIDs).
    refs = trace.get("retrieved_chunk_references") or []
    if refs:
        for corpus in chat_service.CORPORA:
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
    panel = compute_framework_panel(trace_id)
    if panel is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return panel


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
        "size_bytes": os.path.getsize(path),
        "modified_at": os.path.getmtime(path),
        "url": f"/ui/reports/{os.path.basename(path)}",
    } for path in paths]}


@app.get("/ui/reports/{name}")
def download_diagnostic_report(name: str):
    """Download one report while rejecting traversal and non-HTML files."""
    report_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "artifacts", "diagnostic_reports"))
    candidate = os.path.abspath(os.path.join(report_dir, name))
    if os.path.dirname(candidate) != report_dir or not candidate.endswith(".html") or not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="Diagnostic report not found")
    return FileResponse(candidate, media_type="text/html", content_disposition_type="inline")


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
        with open(monitor_path, encoding="utf-8-sig") as handle:
            monitor = json.load(handle)
    return {"run": run, "experiments": statuses, "monitor": monitor}


def _experiment_directory(source: str):
    base_dir = os.path.join(PROJECT_ROOT, "artifacts", "experiments")
    pointer = os.path.join(base_dir, "active_run.json")
    if source == "active" and os.path.isfile(pointer):
        with open(pointer, encoding="utf-8") as handle:
            run_id = json.load(handle)["run_id"]
        if not isinstance(run_id, str) or not run_id.startswith("run40_") or not all(c.isalnum() or c == "_" for c in run_id):
            raise HTTPException(status_code=500, detail="Invalid active run")
        return os.path.join(base_dir, "runs", run_id)
    return base_dir


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
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)

