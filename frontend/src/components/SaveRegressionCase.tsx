import { useEffect, useRef, useState } from "react";
import { getTrace, saveRegressionCase, type CaseExpectations, type CaseResult, type TraceData } from "../lib/api";

const lines = (text: string) => text.split("\n").map(v => v.trim()).filter(Boolean);

export default function SaveRegressionCase({traceId, disabled}: {traceId?: string; disabled?: boolean}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [evidence, setEvidence] = useState("");
  const [required, setRequired] = useState("");
  const [forbidden, setForbidden] = useState("");
  const [refusal, setRefusal] = useState<CaseExpectations["refusal"]>("forbidden");
  const [match, setMatch] = useState<CaseExpectations["fact_match"]>("text");
  const [trace, setTrace] = useState<TraceData | null>(null);
  const [saved, setSaved] = useState<{path: string; baseline: CaseResult} | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  useEffect(() => {
    const controller = new AbortController();
    if (open && traceId) getTrace(traceId, controller.signal).then(value => {
      if (controller.signal.aborted) return;
      setTrace(value); setName(current => current || value?.question.slice(0, 150) || "");
    }).catch(e => { if (e.name !== "AbortError") setError(e.message); });
    return () => controller.abort();
  }, [open, traceId]);
  async function save() {
    if (!traceId || busyRef.current) return;
    busyRef.current = true; setBusy(true); setError("");
    try {
      setSaved(await saveRegressionCase(traceId, name, {expected_evidence: lines(evidence),
        required_facts: lines(required), forbidden_claims: lines(forbidden), refusal, fact_match: match}));
    } catch (e) { setError((e as Error).message); }
    finally { busyRef.current = false; setBusy(false); }
  }
  return <div className="regression-save">
    <button type="button" className="retry-btn" disabled={!traceId || disabled} onClick={() => setOpen(v => !v)}>Save as test case</button>
    {open && <section className="card regression-form" aria-label="Save regression case">
      <h3>Save this answer as a regression case</h3>
      <p>Capture the question and document evidence with engineer-defined expectations. The original failure can remain failing until fixed.</p>
      {error && <p role="alert">{error}</p>}
      {saved ? <div role="status"><p>Saved to {saved.path}</p><p>Original snapshot: {saved.baseline.passed ? "passes text assertions" : "fails or needs additional judging"}.</p><p>Review the case file before committing it. Conversation history and recalled memory are excluded; document text and the original answer are included.</p></div> :
      <form onSubmit={e => {e.preventDefault(); void save();}}>
        <fieldset disabled={busy}>
          <label>Case name <input autoFocus aria-label="Case name" required maxLength={160} value={name} onChange={e => setName(e.target.value)} /></label>
          <label>Expected evidence IDs, one per line <textarea aria-label="Expected evidence IDs" rows={2} value={evidence} onChange={e => setEvidence(e.target.value)} /></label>
          <details><summary>Captured passage IDs</summary>{trace?.retrieved_chunk_references.map(c => <p key={c.chunk_id}><code>{c.chunk_id}</code> — {c.source_file}, page {c.page_number}</p>)}</details>
          <label>Required facts, one per line <textarea aria-label="Required facts" rows={3} value={required} onChange={e => setRequired(e.target.value)} /></label>
          <label>Forbidden claims, one per line <textarea aria-label="Forbidden claims" rows={3} value={forbidden} onChange={e => setForbidden(e.target.value)} /></label>
          <label>Fact matching <select aria-label="Fact matching" value={match} onChange={e => setMatch(e.target.value as typeof match)}><option value="text">Literal phrases (case and whitespace normalized)</option><option value="nli">Semantic support with local NLI</option></select></label>
          <p className="text-muted">Literal checks also match quoted or negated phrases. NLI checks can recognize paraphrases; unavailable or ambiguous forbidden-claim judgments do not pass.</p>
          <label>Refusal policy <select aria-label="Refusal policy" value={refusal} onChange={e => setRefusal(e.target.value as typeof refusal)}><option value="forbidden">Must answer</option><option value="allowed">May refuse</option><option value="required">Must refuse</option></select></label>
          <button type="submit" className="btn btn-primary">{busy ? "Saving…" : "Save regression case"}</button>
        </fieldset>
      </form>}
    </section>}
  </div>;
}
