"""On-demand evaluation-framework metric panel for a saved chat trace.

The UI's "Compare frameworks" button is served from here. Given a trace id,
this module rebuilds the retrieved chunks from the chunk registries, ensures
claim verification exists (running the same deep analysis as a live turn via
``chat_service.deep_analyze`` when the turn was produced without it), computes
the thirteen-metric panel (``src/rag_eval.compute_metric_panel``), and groups
the scores by evaluation-framework family together with their provenance.

Honesty contract (rendered in the UI, and in the module-level ``PANEL_NOTE``):
these are RAGAS-, RAGChecker- and ARES-*style* scores computed by *this
project's own judges against its own definitions of what those metrics mean*.
They are comparable across turns within this app, and they are NOT the numbers
the upstream packages would return on the same turn -- the byte-for-byte
cross-framework comparison is the offline benchmark suite
(``scripts/run_baseline_comparison.py``), not a chat turn.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from src import chat_service
from src.claim_verifier import ClaimVerifier, VerificationResult, VerificationStatus, VerificationSummary
from src.rag_trace import RAGTrace
from src.rag_eval import METRIC_PROVENANCE, compute_metric_panel

logger = logging.getLogger(__name__)

PANEL_NOTE = (
    "RAGAS-style, RAGChecker-style and ARES-style scores below are computed by "
    "X-RAG's own judges against its own definitions of these metrics -- "
    "comparable across answers in this app, but not byte-identical to the "
    "upstream packages' outputs. The real cross-framework comparison lives in "
    "the offline benchmark (artifacts/benchmark_comparison/)."
)

#: Display order and human labels for the framework families.
FAMILY_ORDER = ["RAGAS", "RAGChecker", "ARES", "this project"]

_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_correctness": "Answer Correctness",
    "precision": "Claim Precision",
    "recall": "Claim Recall",
    "f1": "Claim F1",
    "hallucination": "Hallucination Rate",
    "context_relevance": "Context Relevance",
    "answer_relevance": "Answer Relevance",
    "answer_faithfulness": "Answer Faithfulness",
    "citation_correctness": "Citation Correctness",
}

# In-process cache: a panel costs ~15 LLM-judge calls plus a full NLI
# verification pass, so recomputing on every tab open would burn the shared
# rate budget for nothing.
_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX = 32


def invalidate_framework_panel(trace_id):
    _CACHE.pop(trace_id, None)


def _trace_file(trace_id: str) -> Optional[str]:
    trace_dir = os.path.join(chat_service.PROJECT_ROOT, "artifacts", "rag_traces")
    matches = glob.glob(os.path.join(trace_dir, "*", f"trace_{trace_id}.json"))
    matches += [p for p in (os.path.join(trace_dir, f"{trace_id}.json"),
                            os.path.join(trace_dir, f"trace_{trace_id}.json")) if os.path.isfile(p)]
    return matches[0] if matches else None


def _rebuild_chunks(trace: RAGTrace) -> list:
    """RetrievedChunks with real text, resolved against BOTH corpus registries
    (the trace doesn't record which corpus it ran against; chunk ids are
    per-ingestion UUIDs, so a collision is not realistic)."""
    captured = ClaimVerifier.build_retrieved_chunks_from_trace(trace, None)
    if len(captured) == len(trace.retrieved_chunk_references):
        return captured
    for corpus in chat_service.CORPORA:
        registry = chat_service.load_registry(corpus)
        if registry is None:
            continue
        chunks = ClaimVerifier.build_retrieved_chunks_from_trace(trace, registry)
        if chunks:
            return chunks
    return captured


def stored_verification_summary(trace_json: Dict[str, Any]) -> Optional[VerificationSummary]:
    """Rebuild a VerificationSummary from the claim payload a live turn (or a
    previous on-demand run) persisted into the trace file."""
    payload = trace_json.get("claim_verification") or (trace_json.get("diagnostics") or {}).get("claim_verification")
    if not payload:
        return None
    results = payload.get("results") or []
    counts = {s: 0 for s in (
        "SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "NOT_VERIFIABLE")}
    for r in results:
        if r.get("status") in counts:
            counts[r["status"]] += 1
    total = payload.get("claim_count") or len(results)
    entailments = [r.get("entailment_score", 0.0) or 0.0 for r in results]
    return VerificationSummary(
        trace_id=trace_json["trace_id"],
        total_claims=total,
        supported_claims=counts["SUPPORTED"],
        partially_supported_claims=counts["PARTIALLY_SUPPORTED"],
        contradicted_claims=counts["CONTRADICTED"],
        unsupported_claims=counts["UNSUPPORTED"],
        not_verifiable_claims=counts["NOT_VERIFIABLE"],
        average_entailment_score=round(sum(entailments) / len(entailments), 4)
        if entailments else 0.0,
        total_verification_latency_ms=0.0,  # not persisted per-claim; unknown here
        results=[
            VerificationResult(
                verification_id=r.get("verification_id") or r["claim_id"],
                trace_id=trace_json["trace_id"],
                claim_id=r["claim_id"],
                claim_text=r.get("claim_text") or "",
                verification_status=VerificationStatus(r["status"]),
                verification_reason=r.get("reason") or "",
                confidence=r.get("confidence") or 0.0,
                best_chunk_id=r.get("best_chunk_id"),
                best_chunk_rank=r.get("best_chunk_rank"),
                best_chunk_score=None,
                best_sentence_id=None,
                evidence_text=r.get("evidence_text"),
                entailment_score=r.get("entailment_score") or 0.0,
                contradiction_score=r.get("contradiction_score") or 0.0,
                neutral_score=r.get("neutral_score") or 0.0,
                verified_by=r.get("verified_by") or "nli",
                boundary_check=r.get("boundary_check"),
            )
            for r in results if r.get("status") in counts and r.get("claim_id")
        ],
    )


def _summary_to_dict(summary: VerificationSummary) -> Dict[str, Any]:
    return {
        "total": summary.total_claims,
        "supported": summary.supported_claims,
        "partially_supported": summary.partially_supported_claims,
        "contradicted": summary.contradicted_claims,
        "unsupported": summary.unsupported_claims,
        "not_verifiable": summary.not_verifiable_claims,
        "average_entailment": summary.average_entailment_score,
    }


@lru_cache(maxsize=1)
def _ragas_evaluator():
    """Shared evaluator (LLM judge + embed model + NLI verifier), loaded once."""
    from configs.models import EMBEDDING_MODEL_NAME
    from src.embedding_engine import get_shared_embed_model
    from src.generator import Generator
    from src.ragas_metrics import RagasEvaluator

    return RagasEvaluator(
        llm=Generator().llm,
        embed_model=get_shared_embed_model(EMBEDDING_MODEL_NAME),
        claim_verifier=chat_service.load_verifier(),
    )


def compute_framework_panel(trace_id: str) -> Optional[Dict[str, Any]]:
    """The full framework-comparison payload for one trace, or None if the
    trace id is unknown. Sync and heavy; callers should run it in a threadpool."""
    cached = _CACHE.get(trace_id)
    if cached is not None:
        return {**cached, "cached": True}

    path = _trace_file(trace_id)
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        trace_json = json.load(f)
    trace = RAGTrace.from_json(trace_json)

    chunks = _rebuild_chunks(trace)

    # Claim verification: reuse what the live turn persisted, else run the
    # same deep analysis now and persist it back (computed once, cached once).
    computed_claims = False
    analysis_error = None
    summary = stored_verification_summary(trace_json)
    if summary is None:
        try:
            evidence = chunks + chat_service.captured_memory_chunks(trace)
        except ValueError as exc:
            outcome = {"error": str(exc)}
        else:
            outcome = chat_service.deep_analyze(trace, evidence)
        if "error" in outcome:
            analysis_error = outcome["error"]
            logger.warning("On-demand deep analysis failed for %s: %s", trace_id, outcome["error"])
            summary = None
        else:
            trace_json["claim_verification"] = outcome["verification"]
            chat_service.persist_deep_analysis(path, outcome)
            summary = stored_verification_summary(trace_json)
            computed_claims = True

    panel = compute_metric_panel(
        question=trace.question,
        answer=trace.generated_answer,
        retrieved_chunks=chunks,
        verification=summary,
        ragas_evaluator=_ragas_evaluator(),
    )

    families: Dict[str, List[Dict[str, Any]]] = {}
    for key, value in panel.items():
        info = METRIC_PROVENANCE[key]
        families.setdefault(info["family"], []).append({
            "key": key,
            "label": _METRIC_LABELS.get(key, key),
            "value": value,
            "definition": info["definition"],
            "judge": info["judge"],
            "requires": info.get("requires"),
        })
    ordered = [
        {"family": fam, "metrics": families[fam]}
        for fam in FAMILY_ORDER if fam in families
    ]

    out = {
        "trace_id": trace_id,
        "note": PANEL_NOTE + (f" Claim verification unavailable: {analysis_error}" if analysis_error else ""),
        "analysis_error": analysis_error,
        "families": ordered,
        "verification": _summary_to_dict(summary) if summary else None,
        "computed_claims": computed_claims,
        "cached": False,
    }
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[trace_id] = out
    return {**out, "cached": False}
