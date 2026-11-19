"""Export saved chat turns with the standalone diagnostic HTML renderer."""

import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile

from scripts.generate_diagnostic_report import (
    STYLE,
    esc,
    load_answer_correctness,
    load_baseline_row,
    load_chunk_records,
    load_report,
    render_report_body,
)
from src.claims import ClaimSet
from src.corrective_action_engine import CorrectiveActionEngine
from src.pipeline_state_analyzer import PipelineStateAnalyzer
from src.ragas_metrics import RagasMetrics, compute_faithfulness_strict
from src.report_builder import ReportBuilder
from src.root_cause_reasoner import RootCauseReasoner
from src.trace_metrics import stored_verification_summary

logger = logging.getLogger(__name__)


def _chat_diagnostics(trace):
    diagnostics = trace.get("diagnostics") or {}
    verification = diagnostics.get("claim_verification")
    claims = (verification or {}).get("results", [])
    entailment = [claim["entailment_score"] for claim in claims if claim.get("entailment_score") is not None]
    return {
        "executive_summary": {
            "primary_issue": "Unavailable",
            "total_claims": verification.get("claim_count", len(claims)) if verification is not None else None,
            "supported_claims": sum(claim.get("status") == "SUPPORTED" for claim in claims),
        },
        "evaluation_metrics": {"average_entailment": mean(entailment) if entailment else None},
        "root_cause_analysis": {},
        "pipeline_overview": {"pipeline_stages": [
            {
                "stage": stage.replace("_", " ").upper(),
                "status": {"completed": "COMPLETED", "failed": "FAIL", "partial": "PARTIAL"}.get(status, "UNKNOWN"),
                "observation": "Recorded execution status",
            }
            for stage, status in trace.get("pipeline_stage_status", {}).items()
        ]},
        "evidence_analysis": [
            {
                "claim_id": claim.get("claim_id"),
                "claim_text": claim.get("claim_text"),
                "verification_status": claim.get("status"),
                "supporting_chunk_id": claim.get("best_chunk_id"),
                "supporting_chunk_rank": claim.get("best_chunk_rank"),
                "supporting_evidence": claim.get("evidence_text"),
            }
            for claim in claims
        ],
    }


def _build_diagnostics(trace, verification, panel):
    states = PipelineStateAnalyzer().analyze(trace, ClaimSet(trace_id=trace.trace_id), verification)
    cause = RootCauseReasoner().analyze(states)
    actions = CorrectiveActionEngine().generate(cause, psm=states, config_snapshot=trace.configuration_snapshot)
    scores = {metric["key"]: metric["value"] for family in (panel or {}).get("families", [])
              if family["family"] == "RAGAS" for metric in family["metrics"]}
    metrics = RagasMetrics(
        faithfulness=scores.get("faithfulness", compute_faithfulness_strict(verification)),
        answer_relevancy=scores.get("answer_relevancy"),
        context_precision=scores.get("context_precision"),
    )
    context_scores = {metric["key"]: metric["value"] for family in (panel or {}).get("families", [])
                      if family["family"] == "ARES" for metric in family["metrics"]}
    metrics.context_relevancy = context_scores.get("context_relevance")
    return asdict(ReportBuilder().build(trace=trace, psm=states, rca=cause, cap=actions,
                                       verification=verification, ragas_metrics=metrics))


def save_session_report(report_dir, session, turns, load_trace, compute_framework_panel):
    artifacts = report_dir.parent
    sections = []
    for number, turn in enumerate(turns, 1):
        trace_id = turn.trace_id or ""
        trace = {"trace_id": "", "timestamp": turn.timestamp}
        report = None
        baseline = correctness = None
        panel = None
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id):
            try:
                captured = load_trace(trace_id)
                trace = asdict(captured)
            except (FileNotFoundError, TypeError, ValueError):
                pass
            if trace["trace_id"] == trace_id:
                try:
                    report = load_report(trace_id, artifacts / "reports")
                except FileNotFoundError:
                    pass
                baseline = load_baseline_row(trace_id, artifacts / "benchmark_comparison" / "results.json")
                correctness = load_answer_correctness(trace_id, artifacts / "answer_correctness")
                if report is None:
                    verification = stored_verification_summary(trace)
                    try:
                        panel = compute_framework_panel(trace_id)
                        if verification is None and panel and panel.get("verification") is not None:
                            captured = load_trace(trace_id)
                            trace = asdict(captured)
                            verification = stored_verification_summary(trace)
                    except Exception:
                        logger.exception("Could not compute report metrics for trace %s", trace_id)
                    if verification is not None:
                        report = _build_diagnostics(captured, verification, panel)

        partial = report is None
        if partial:
            report = _chat_diagnostics(trace)
        report["executive_summary"].update(question=turn.question, generated_answer=turn.answer)
        refs = trace.get("retrieved_chunk_references", [])
        records = load_chunk_records([ref["chunk_id"] for ref in refs], artifacts / "chunk_registry.json") if refs else {}
        body = render_report_body(trace, report, baseline, correctness, partial=partial,
                                  chunk_records=records, framework_panel=panel)
        sections.append(f'<article id="turn-{number}" aria-label="Turn {number}">{body}</article>')

    title = esc(session.title or "Untitled session")
    contents = " · ".join(f'<a href="#turn-{n}">Turn {n}</a>' for n in range(1, len(turns) + 1))
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} | X-RAG Session Report</title>
  <style>{STYLE}
  nav a {{ color: var(--accent); }}
  article {{ margin-top: 40px; }}
  </style>
</head>
<body>
<main class="page">
  <header>
    <div class="eyebrow">X-RAG Session Report</div>
    <h1>{title}</h1>
    <nav aria-label="Conversation turns">{contents}</nav>
  </header>
  {''.join(sections)}
</main>
</body>
</html>
"""
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"session.{session.session_id}.html"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=report_dir, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(page)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
