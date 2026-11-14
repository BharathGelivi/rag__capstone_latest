"""Retrieval-strategy arms: the single catalogue of retrieval strategies the
system can run.

This module is why a chat turn and the E6 ablation cannot drift apart: the UI
(``src/chat_service.py``) and the experiment suite (``experiments/exp06…``)
both execute arms through the one ``execute_arm`` defined here, over one
``ARMS`` table. The table used to live inside ``experiments/exp06…``, which
forced production code (the UI backend) to import from the experiment
package -- wrong dependency direction, and exactly the kind of coupling that
lets "the shipped pipeline" and "the measured pipeline" silently become two
different things.

Arms
----
    A_vector                dense bi-encoder only
    A_bm25                  lexical only
    B_hybrid                dense + BM25, RRF fused, no reranker
    C_hybrid_rerank         + cross-encoder reranker      (the shipped pipeline)
    C_fixed_chunking        identical to C, over fixed-size chunks
    D_ircot                 interleaved retrieval/reasoning over C
    E_agentic               bounded controller over C, no graph
    F_graphrag              C, then citation-graph expansion
    G_ircot_graph           D, then citation-graph expansion
    H_agentic_graph         controller with graph actions enabled
    I_full                  IRCoT + graph expansion + contradiction search

Every arm but ``C_fixed_chunking`` runs over the legal-chunked corpus, so
chunking is isolated to exactly one comparison (C vs C_fixed_chunking) rather
than confounded with everything else.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: ``arm -> configuration``. One dict per arm keeps the whole design readable in
#: one screen and makes "what exactly differs between C and D" answerable
#: without reading the executor.
ARMS: Dict[str, Dict[str, Any]] = {
    "A_vector":         {"chunking": "legal", "mode": "vector", "rerank": False, "strategy": "plain"},
    "A_bm25":           {"chunking": "legal", "mode": "bm25",   "rerank": False, "strategy": "plain"},
    "B_hybrid":         {"chunking": "legal", "mode": "hybrid", "rerank": False, "strategy": "plain"},
    "C_hybrid_rerank":  {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "plain"},
    "C_fixed_chunking": {"chunking": "fixed", "mode": "hybrid", "rerank": True,  "strategy": "plain"},
    "D_ircot":          {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "ircot"},
    "E_agentic":        {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "agentic",
                         "graph": False},
    "F_graphrag":       {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "plain",
                         "graph_expand": True},
    "G_ircot_graph":    {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "ircot",
                         "graph_expand": True},
    "H_agentic_graph":  {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "agentic",
                         "graph": True},
    "I_full":           {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "ircot",
                         "graph_expand": True, "contradiction": True},
}

#: Chunks every arm's first retrieval returns. Held constant across arms so a
#: recall difference is a ranking difference, not a budget difference. Wider
#: than the shipped RERANKER_TOP_N because the ablation is a retrieval study,
#: and recall@10 cannot exceed what was retrieved.
BASE_TOP_N = 10

GRAPH_EXPAND_CHUNKS = 6
CONTRADICTION_CHUNKS = 4


# ---------------------------------------------------------------------------
# Arm execution -- shared by E6, E7 and the chat UI so they cannot drift apart
# ---------------------------------------------------------------------------

def graph_expand(graph, chunk_ids, registry, relations=None, limit: int = GRAPH_EXPAND_CHUNKS):
    if graph is None:
        return []
    from src.legal_graph import expand

    return expand(graph, list(chunk_ids), registry, max_hops=2,
                  max_chunks=limit, relations=relations)


def materialise(chunk_ids, registry):
    """Turn registry records into RetrievedChunks so graph results are the same
    type as retrieval results and need no special-casing downstream."""
    from src.retriever import RetrievedChunk

    out = []
    for chunk_id in chunk_ids:
        record = registry.get_chunk(chunk_id)
        if record is None:
            continue
        out.append(RetrievedChunk(
            chunk_id=chunk_id, similarity_score=0.0, rank=0,
            page_number=str(record.metadata.get("page_number", "")),
            source_file=record.source_file, chunk_index=record.chunk_index,
            chunk_text=record.text, parent_document_id=record.parent_document_id,
        ))
    return out


def execute_arm(arm: str, question_text: str, registry, retriever, graph, llm, top_n=None):
    """Run one arm's retrieval. Returns ``(chunks, retrieval_metadata, graph_added)``.

    Single definition of what each arm *is*, imported by E6 (retrieval
    scoring), E7 (generation scoring) and the chat UI. Two copies of this
    logic would let them quietly measure different things under the same arm
    names.
    """
    from src.agentic import ADVERSE_RELATIONS, CONTRADICTION_TERMS, AgenticRetriever
    from src.ircot import IRCoTRetriever

    config = ARMS[arm]
    graph_added = 0

    if config["strategy"] == "plain":
        result = retriever.retrieve(question_text, mode=config["mode"],
                                    rerank=config["rerank"], top_n=top_n or BASE_TOP_N)
    elif config["strategy"] == "ircot":
        result = IRCoTRetriever(retriever, llm, retrieval_mode=config["mode"],
                                rerank=config["rerank"],
                                first_hop_top_n=top_n or BASE_TOP_N).retrieve(question_text)
    elif config["strategy"] == "agentic":
        result = AgenticRetriever(
            retriever, llm, graph=graph if config.get("graph") else None,
            registry=registry, retrieval_mode=config["mode"], rerank=config["rerank"],
        ).retrieve(question_text)
    else:
        raise ValueError(f"unknown strategy in arm {arm}")

    chunks = list(result.retrieved_chunks)

    if config.get("graph_expand"):
        hits = graph_expand(graph, [c.chunk_id for c in chunks[:6]], registry)
        known = {c.chunk_id for c in chunks}
        new = [h.chunk_id for h in hits if h.chunk_id not in known]
        chunks += materialise(new, registry)
        graph_added += len(new)

    if config.get("contradiction"):
        adverse = retriever.retrieve(f"{question_text} {CONTRADICTION_TERMS}",
                                     mode="bm25", rerank=False, top_n=CONTRADICTION_CHUNKS)
        known = {c.chunk_id for c in chunks}
        chunks += [c for c in adverse.retrieved_chunks if c.chunk_id not in known]
        hits = graph_expand(graph, [c.chunk_id for c in chunks[:6]], registry,
                            relations=ADVERSE_RELATIONS, limit=CONTRADICTION_CHUNKS)
        known = {c.chunk_id for c in chunks}
        new = [h.chunk_id for h in hits if h.chunk_id not in known]
        chunks += materialise(new, registry)
        graph_added += len(new)

    return chunks[:top_n] if top_n is not None else chunks, result.retrieval_metadata, graph_added
