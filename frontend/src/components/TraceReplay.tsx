import { useEffect, useRef, useState } from "react";
import { getConfig, getReplaySettings, getSavedTraces, replayTrace, type Config, type ReplayResult, type ReplaySettings } from "../lib/api";
import EvidenceExplorer from "./EvidenceExplorer";

const timingRows = [["Retrieval", "retrieval_time"], ["Generation", "generation_time"], ["Pipeline", "total_pipeline_time"]] as const;

export default function TraceReplay({ initialTraceId }: { initialTraceId?: string }) {
  const [traceId, setTraceId] = useState(initialTraceId ?? "");
  const [choices, setChoices] = useState<Array<{trace_id: string; question: string}>>([]);
  const [config, setConfig] = useState<Config | null>(null);
  const [baseline, setBaseline] = useState<ReplaySettings | null>(null);
  const [settings, setSettings] = useState<ReplaySettings | null>(null);
  const [mode, setMode] = useState<"frozen" | "retrieve">("frozen");
  const [verify, setVerify] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ReplayResult | null>(null);
  const request = useRef<AbortController | null>(null);
  const busyRef = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    getSavedTraces(controller.signal).then(setChoices).catch(e => { if (e.name !== "AbortError") setError(e.message); });
    getConfig().then(setConfig).catch(() => { /* Frozen replay remains available. */ });
    return () => { controller.abort(); request.current?.abort(); };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    Promise.resolve().then(() => {
      if (controller.signal.aborted) return;
      setBaseline(null); setSettings(null); setResult(null); setError("");
      if (!traceId) return;
      getReplaySettings(traceId, controller.signal).then(value => {
        if (controller.signal.aborted) return;
        setBaseline(value); setSettings(value);
      }).catch(e => { if (e.name !== "AbortError") setError(e.message); });
    });
    return () => controller.abort();
  }, [traceId]);

  async function run() {
    if (!settings || !baseline || busyRef.current) return;
    busyRef.current = true; setBusy(true); setError(""); setResult(null);
    const controller = new AbortController(); request.current = controller;
    try {
      const comparison = await replayTrace(traceId, {
        mode, verify_claims: verify,
        ...(settings.model !== baseline.model && {model: settings.model}),
        ...(settings.system_prompt !== baseline.system_prompt && {system_prompt: settings.system_prompt}),
        ...(mode === "retrieve" && {arm: settings.arm, corpus: settings.corpus, top_k: settings.top_k}),
      }, controller.signal);
      setResult(comparison);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      busyRef.current = false; setBusy(false);
    }
  }

  return <section className="card elev-sm trace-replay" aria-label="Trace replay">
    <h2>Replay a trace</h2>
    <p>Change a setting and compare against the saved answer. The original trace stays unchanged.</p>
    <label>Saved trace
      <select aria-label="Saved trace" value={traceId} disabled={busy} onChange={e => setTraceId(e.target.value)}>
        <option value="">Select a saved trace</option>
        {traceId && !choices.some(c => c.trace_id === traceId) && <option value={traceId}>{traceId}</option>}
        {choices.map(c => <option key={c.trace_id} value={c.trace_id}>{c.question.slice(0, 90)} — {c.trace_id.slice(0, 8)}</option>)}
      </select>
    </label>
    <label>Trace ID <input aria-label="Trace ID" value={traceId} disabled={busy} onChange={e => setTraceId(e.target.value)} placeholder="Or paste any saved trace ID" /></label>
    {traceId && !settings && !error && <p role="status">Loading captured settings…</p>}
    {error && <p role="alert">{error}</p>}
    {settings && <form onSubmit={e => { e.preventDefault(); void run(); }}>
      <fieldset disabled={busy}>
        <label>Evidence mode
          <select aria-label="Evidence mode" value={mode} onChange={e => setMode(e.target.value as typeof mode)}>
            <option value="frozen">Reuse original captured evidence</option>
            <option value="retrieve">Retrieve again from the current index</option>
          </select>
        </label>
        <label>Generation model <input aria-label="Generation model" required value={settings.model} maxLength={160} onChange={e => setSettings({...settings, model: e.target.value})} /></label>
        <label>System prompt <textarea aria-label="System prompt" required rows={6} maxLength={40000} value={settings.system_prompt} onChange={e => setSettings({...settings, system_prompt: e.target.value})} /></label>
        {mode === "retrieve" && <div className="replay-controls">
          <label>Corpus <select aria-label="Corpus" required value={settings.corpus} onChange={e => {
            const corpus = e.target.value;
            const arms = config?.arms_by_corpus[corpus] ?? [];
            setSettings({...settings, corpus, arm: arms.includes(settings.arm) ? settings.arm : arms[0] ?? "C_hybrid_rerank"});
          }}>
            <option value="" disabled>Select corpus</option>
            {config?.corpora.map(c => <option key={c} value={c}>{c}</option>)}
          </select></label>
          <label>Strategy <select aria-label="Strategy" value={settings.arm} onChange={e => setSettings({...settings, arm: e.target.value})}>
            {config?.arms_by_corpus[settings.corpus]?.map(a => <option key={a} value={a}>{a}</option>)}
          </select></label>
          <label>Top-K context limit <input aria-label="Top-K context limit" type="number" min={1} max={30} required value={settings.top_k} onChange={e => setSettings({...settings, top_k: Number(e.target.value)})} /></label>
        </div>}
        <label className="replay-check"><input type="checkbox" checked={verify} onChange={e => setVerify(e.target.checked)} />Verify replay claims</label>
        <p className="text-muted">Replay sends the captured conversation, evidence and edited prompt to the configured NVIDIA service. Claim verification can make additional calls. No new chat memories are created. Provider outputs may vary.</p>
        <button type="submit" className="btn btn-primary">{busy ? "Replaying…" : "Run replay"}</button>
      </fieldset>
    </form>}
    {busy && <p role="status" aria-live="polite">Generating the replay{verify ? " and verifying its claims" : ""}… This can take a few minutes.</p>}
    {result && <div aria-label="Replay comparison">
      <h3>Original vs replay</h3>
      {result.warnings.map(w => <p key={w} className="text-muted">{w}</p>)}
      <p>{result.comparison.answer_changed ? "Answer changed." : "Answer unchanged."} Evidence: {result.comparison.evidence_added.length} added, {result.comparison.evidence_removed.length} removed. {result.comparison.evidence_order_changed ? "Evidence selection or order changed." : "Evidence order preserved."}</p>
      <div className="replay-columns">
        {(["original", "replay"] as const).map(side => {
          const trace = result[side];
          const scores = result.comparison.scores[side];
          return <article key={side}>
            <h4>{side === "original" ? "Original" : "Replay"}</h4>
            <p className="text-muted">Trace: {trace.trace_id}</p>
            <div className="replay-answer">{trace.generated_answer}</div>
            <p>Supported claims: {scores.supported_fraction == null ? "Not measured" : `${Math.round(scores.supported_fraction * 100)}% of ${scores.claim_count}`}</p>
            <details><summary>Evidence ({trace.retrieved_chunk_references.length} passages)</summary>
              {trace.retrieved_chunk_references.map(c => <div key={c.chunk_id} className="replay-evidence"><strong>{c.source_file || c.chunk_id}</strong>
                {side === "original" && result.comparison.evidence_removed.includes(c.chunk_id) && <span> — Removed</span>}
                {side === "replay" && result.comparison.evidence_added.includes(c.chunk_id) && <span> — Added</span>}
                <p>{c.text || "Text not captured in this legacy trace."}</p></div>)}
            </details>
            <details><summary>Claim verification</summary><EvidenceExplorer trace={trace} /></details>
          </article>;
        })}
      </div>
      <h4>Claim changes</h4>
      {result.comparison.claim_changes == null ? <p>Both answers need claim verification for a verdict comparison.</p> :
        result.comparison.claim_changes.length === 0 ? <p>No changes in extracted claim text or verdicts.</p> :
        <ul>{result.comparison.claim_changes.map(c => <li key={c.claim_text}>{c.claim_text}: {c.original_status ?? "Not in original"} → {c.replay_status ?? "Not in replay"}</li>)}</ul>}
      <p className="text-muted">Claims are matched by normalized text; paraphrased claims appear as removed and added.</p>
      <table className="table"><caption>Recorded timings in seconds</caption><thead><tr><th>Stage</th><th>Original</th><th>Replay</th><th>Change</th></tr></thead><tbody>
        {timingRows.map(([label, key]) => {
          const a = result.original.execution_statistics[key], b = result.replay.execution_statistics[key];
          return <tr key={key}><th>{label}</th><td>{typeof a === "number" ? a.toFixed(3) : "Not recorded"}</td><td>{typeof b === "number" ? b.toFixed(3) : "Not recorded"}</td><td>{typeof a === "number" && typeof b === "number" ? `${b - a >= 0 ? "+" : ""}${(b - a).toFixed(3)}` : "—"}</td></tr>;
        })}
      </tbody></table>
      <details><summary>Configuration changes and captured prompts</summary>
        <table className="table"><thead><tr><th>Setting</th><th>Original</th><th>Replay</th></tr></thead><tbody>
          {[...new Set([...Object.keys(result.original.configuration_snapshot), ...Object.keys(result.replay.configuration_snapshot)])].map(key => <tr key={key}><th>{key}</th><td>{JSON.stringify(result.original.configuration_snapshot[key]) ?? "Not recorded"}</td><td>{JSON.stringify(result.replay.configuration_snapshot[key]) ?? "Not recorded"}</td></tr>)}
        </tbody></table>
        <div className="replay-columns">{(["original", "replay"] as const).map(side => <div key={side}><h4>{side}</h4><pre className="debug-pre">{result[side].prompt_snapshot}</pre></div>)}</div>
      </details>
    </div>}
  </section>;
}
