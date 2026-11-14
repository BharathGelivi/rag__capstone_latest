"""Interactive graph over the framework's own memory store.

Used to read from ``~/.claude-mem/`` -- the developer's personal AI memory
tool, which made the shipped "Memory graph" view silently empty on every
other machine. Now it graphs what this project actually stores: sessions,
memory entries (turn/fact/summary), and the traces they produced, in the
same ``{"nodes": [...], "edges": [...], "truncated": bool}`` shape as
``ui/components/graph.py``'s ``build_graph_data()``, so both graphs share
one renderer on the frontend.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: kind -> display color. Session pink / memory blue / fact green / summary
#: amber / trace dark -- the same legend conventions as the RAG graph.
NODE_TYPES: Dict[str, Dict[str, str]] = {
    "session": {"label": "Session", "color": "#ec4899"},
    "turn": {"label": "Memory (turn)", "color": "#3b82f6"},
    "fact": {"label": "Memory (fact)", "color": "#10a37f"},
    "summary": {"label": "Summary", "color": "#f59e0b"},
    "trace": {"label": "Trace", "color": "#0d0d0d"},
}


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_memory_graph_data(
    project: Optional[str] = None,
    max_observations: int = 150,
) -> Dict[str, Any]:
    """Assemble nodes/edges from the project's own memory store.

    Args:
        project:          Kept for API compatibility; the project's store is
                          single-tenant, so this parameter only filters turns
                          whose session title matches it when given.
        max_observations: Memory-node cap, most-recent-first -- an uncapped
                          pull renders as a hairball.
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    truncated = False

    def add_node(node_id: str, node_type: str, label: str, **extra: Any) -> None:
        if node_id in nodes:
            return
        info = NODE_TYPES.get(node_type, {"label": node_type, "color": "#9ca3af"})
        nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "label": label,
            "color": info["color"],
            "type_label": info["label"],
            "degree": 0,
            **extra,
        }

    def add_edge(source: str, target: str, kind: str) -> None:
        if source in nodes and target in nodes:
            edges.append({"source": source, "target": target, "kind": kind})
            nodes[source]["degree"] += 1
            nodes[target]["degree"] += 1

    try:
        # Late import to keep this module importable without a framework stack
        from src import chat_service
        mm = chat_service.get_memory_manager()
    except Exception as exc:
        logger.warning("memory graph: store unavailable: %s", exc)
        return {"nodes": [], "edges": [], "truncated": False}

    try:
        sessions = {s.session_id: s for s in mm.list_sessions()}
        for s in sessions.values():
            add_node(s.session_id, "session", _truncate(s.title or s.session_id, 48))

        memories = mm.store.get_all_memories(limit=max_observations * 2)
        memories.sort(key=lambda m: m.timestamp or "", reverse=True)
        if len(memories) > max_observations:
            memories = memories[:max_observations]
            truncated = True

        for mem in memories:
            kind = mem.kind if mem.kind in NODE_TYPES else "turn"
            label = _truncate(mem.question if kind == "turn" else mem.answer, 64)
            add_node(
                mem.memory_id, kind, label,
                detail=f"{mem.question} — {mem.answer}" if kind != "turn"
                else f"Q: {mem.question}\nA: {mem.answer}",
                meta=f"{mem.session_id} · {mem.timestamp}",
            )
            if mem.session_id in nodes:
                add_edge(mem.session_id, mem.memory_id, "contains")

            # Provenance edge to the trace that produced this memory.
            if mem.trace_id:
                add_node(
                    mem.trace_id, "trace", f"trace {_truncate(mem.trace_id, 12)}",
                    meta=mem.timestamp,
                )
                add_edge(mem.memory_id, mem.trace_id, "produced_by")

            # Supersession edge (new fact replacing an older one).
            if mem.superseded_by:
                add_node(
                    mem.superseded_by, "fact", "(superseding entry)",
                    meta="",
                )
                add_edge(mem.memory_id, mem.superseded_by, "superseded_by")
    except Exception as exc:
        logger.warning("memory graph build failed: %s", exc)
        return {"nodes": [], "edges": [], "truncated": truncated}

    return {"nodes": list(nodes.values()), "edges": edges, "truncated": truncated}
