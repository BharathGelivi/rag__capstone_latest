"""Export the saved turns of a chat session as a standalone HTML report."""

from html import escape
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from scripts.generate_diagnostic_report import STYLE


def save_session_report(report_dir, session, turns, load_trace):
    sections = []
    for number, turn in enumerate(turns, 1):
        trace_id = turn.trace_id or ""
        trace_label = ""
        evidence = []
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id):
            filename = f"{trace_id}.html"
            trace_label = f"Trace <code>{trace_id}</code>"
            if (report_dir / filename).is_file():
                trace_label += f' · <a href="{filename}">Diagnostic report</a>'
            try:
                trace = load_trace(trace_id)
            except (FileNotFoundError, TypeError, ValueError):
                trace = None
            if trace:
                for chunk in trace.retrieved_chunk_references:
                    reference = escape(str(chunk.get("chunk_id", "")))
                    source = escape(str(chunk.get("source_file") or "Source unavailable"))
                    passage = escape(str(chunk.get("text") or "Passage text unavailable"))
                    evidence.append(f"<h4>{source} · {reference}</h4><pre>{passage}</pre>")

        passages = (
            f"<details><summary>Retrieved passages ({len(evidence)})</summary>"
            f"{''.join(evidence)}</details>"
            if evidence else "<p>Captured passages unavailable.</p>"
        )
        sections.append(f"""
<section id="turn-{number}">
  <h2>Turn {number}</h2>
  <p>{escape(str(turn.timestamp))} {trace_label}</p>
  <h3>Question</h3><pre>{escape(turn.question)}</pre>
  <h3>Answer</h3><pre>{escape(turn.answer)}</pre>
  {passages}
</section>""")

    title = escape(session.title or "Untitled session")
    contents = " · ".join(f'<a href="#turn-{n}">Turn {n}</a>' for n in range(1, len(turns) + 1))
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | X-RAG Session Report</title>
<style>{STYLE}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; }}
</style>
<main class="page">
  <header><div class="eyebrow">X-RAG Session Report</div><h1>{title}</h1></header>
  <nav aria-label="Conversation turns">{contents}</nav>
  {''.join(sections)}
</main>
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
