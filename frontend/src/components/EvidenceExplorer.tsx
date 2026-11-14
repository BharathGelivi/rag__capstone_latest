import { useEffect, useRef, useState } from "react";
import {analyzeEvidence, getEvidenceDebug, type EvidenceDebugData, type TraceData} from "../lib/api";
import EvidenceDetails from "./EvidenceDetails";

const verdictLabels = {
  SUPPORTED: "Supported",
  PARTIALLY_SUPPORTED: "Partially supported",
  CONTRADICTED: "Contradicted",
  UNSUPPORTED: "Unsupported",
  NOT_VERIFIABLE: "Not verifiable",
} as const;

export default function EvidenceExplorer({ trace }: { trace: TraceData | null }) {
  const [debug, setDebug] = useState<{id: string; data: EvidenceDebugData} | null>(null);
  const [debugError, setDebugError] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const analyzeRef = useRef(false);
  const currentTrace = useRef(trace?.trace_id);
  useEffect(() => {
    currentTrace.current = trace?.trace_id;
    const controller = new AbortController();
    if (trace?.trace_id) getEvidenceDebug(trace.trace_id, controller.signal).then(data => {
      if (!controller.signal.aborted && Array.isArray(data.sentences)) {setDebug({id: trace.trace_id, data}); setDebugError("");}
    }).catch(e => {if (e.name !== "AbortError") setDebugError(e.message);});
    return () => {controller.abort(); currentTrace.current = undefined;};
  }, [trace?.trace_id]);
  const currentDebug = debug?.id === trace?.trace_id ? debug?.data : null;
  async function analyze() {
    if (!trace || analyzeRef.current) return;
    const id = trace.trace_id;
    analyzeRef.current = true; setAnalyzing(true); setDebugError("");
    try {const data = await analyzeEvidence(id); if (currentTrace.current === id) setDebug({id, data});}
    catch (e) {if (currentTrace.current === id) setDebugError((e as Error).message);}
    finally {analyzeRef.current = false; setAnalyzing(false);}
  }
  const explorerRef = useRef<HTMLElement>(null);
  function setVisiblePanelsOpen(open: boolean) {
    explorerRef.current?.querySelectorAll<HTMLDetailsElement>("details").forEach(panel => {
      panel.open = open;
    });
  }
  const [filter, setFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [copyFeedback, setCopyFeedback] = useState("");
  async function copyEvidence(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopyFeedback("Claim and evidence copied.");
    } catch {
      setCopyFeedback("Could not copy. Select the claim and evidence text and copy manually.");
    }
  }
  const claims = currentDebug?.claims ?? trace?.claim_verification?.results ?? [];
  const query = search.trim().toLowerCase();
  const visibleClaims = claims.filter(claim =>
    (filter === "ALL" || claim.status === filter) &&
    claim.claim_text.toLowerCase().includes(query)
  );
  return <section ref={explorerRef} className="card elev-sm" aria-label="Claim evidence explorer">
    <h2>Claim evidence explorer</h2>
    <p role="status" aria-live="polite">{copyFeedback}</p>
    <p>Inspect each claim against the evidence used by the verifier. Scores are model outputs, not proof of correctness.</p>
    {trace && <>
      <button type="button" disabled={analyzing} onClick={() => void analyze()}>{analyzing ? "Analyzing evidence…" : "Analyze evidence and citations"}</button>
      <p className="text-muted">Analysis uses captured evidence, local NLI, and NVIDIA claim decomposition / ambiguous judging. It updates this trace's saved verification.</p>
    </>}
    {debugError && <p role="alert">Evidence debugger: {debugError}</p>}
    {currentDebug && trace && <EvidenceDetails key={trace.trace_id} traceId={trace.trace_id} data={currentDebug} />}
    {trace?.claim_error && <p role="alert">Verification failed: {trace.claim_error}</p>}
    {!claims.length && <p>No verified claims available. Enable Deep analysis above, then ask a question in Chat.</p>}
    {!!claims.length && <>
      <div role="group" aria-label="Verdict counts for all claims" style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
        {Object.entries(verdictLabels).map(([status, label]) => (
          <span key={status} className="dpill">
            {claims.filter(claim => claim.status === status).length} {label}
          </span>
        ))}
      </div>
      <label>Verdict <select value={filter} onChange={e => setFilter(e.target.value)}>
        <option value="ALL">All claims ({claims.length})</option>
        {[...new Set(claims.map(c => c.status))].map(s => <option key={s} value={s}>{s.replaceAll("_", " ")}</option>)}
      </select></label>
      <label style={{ display: "block", marginTop: 12 }}>Search claims {" "}
        <input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder="Find a word or phrase" />
      </label>
      {search && <button type="button" onClick={() => setSearch("")}>Clear search</button>}
      <p role="status">Showing {visibleClaims.length} of {claims.length} claims</p>
      {!visibleClaims.length && <p>No claims match your search and verdict filter.</p>}
      <div role="group" aria-label="Visible evidence panels" style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button type="button" onClick={() => setVisiblePanelsOpen(true)}>Expand all</button>
        <button type="button" onClick={() => setVisiblePanelsOpen(false)}>Collapse all</button>
      </div>
      {visibleClaims.map(c => {
        const source = (currentDebug?.sources ?? trace?.retrieved_chunk_references)?.find(r => r.chunk_id === c.best_chunk_id);
        const evidence = c.evidence_text || source?.text;
        return <details key={c.claim_id} style={{ marginTop: 16, padding: 16, border: "1px solid #94a3b8", borderRadius: 12 }}>
          <summary style={{ cursor: "pointer" }}><strong>{c.status.replaceAll("_", " ")}</strong> — {c.claim_text}</summary>
          <p>{c.reason || "No explanation supplied by the verifier."}</p>
          {c.citation_checks?.map(check => <p key={check.chunk_id}>Citation {check.chunk_id}: <strong>{check.status}</strong> — {check.reason}</p>)}
          <button type="button" disabled={!evidence} onClick={() => {
            setCopyFeedback("");
            void copyEvidence([
              `Claim: ${c.claim_text}`,
              `Verdict: ${verdictLabels[c.status]}`,
              `Source: ${source?.source_file || c.best_chunk_id || "Unknown"}`,
              ...(source?.page_number != null ? [`Page: ${source.page_number}`] : []),
              `Evidence: ${evidence}`,
            ].join("\n"));
          }}>Copy evidence</button>
          <p><strong>Evidence</strong> · {source?.source_file || c.best_chunk_id || "No source identified"}{source?.page_number != null ? ` · Page ${source.page_number}` : ""}</p>
          {evidence ? <blockquote style={{ borderLeft: "3px solid #10a37f", paddingLeft: 16, whiteSpace: "pre-wrap" }}>{evidence}</blockquote> : <p>No evidence text available. This does not establish that the claim is false.</p>}
          {source?.text && <details><summary>{currentDebug ? "Full captured source passage" : "Source passage preview"} — page {source.page_number}</summary><p className="replay-answer">{source.text}</p></details>}
          <table style={{ width: "100%", textAlign: "left" }}><thead><tr><th>Model signal</th><th>Score</th></tr></thead><tbody>
            {[["Entailment", c.entailment_score], ["Contradiction", c.contradiction_score], ["Neutral", c.neutral_score]].map(([label, value]) => <tr key={String(label)}><td>{label}</td><td>{typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "Unavailable"}</td></tr>)}
          </tbody></table>
          <p>Verified by: {c.verified_by === "llm" ? "LLM judge" : "NLI model"}</p>
        </details>;
      })}
    </>}
  </section>;
}
