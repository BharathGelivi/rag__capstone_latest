import ResearchDashboard from "./ResearchDashboard";
import "./research-dashboard.css";
import { useEffect, useRef, useState } from "react";
import { API_BASE, getExperimentRuns, type ExperimentRuns } from "../lib/api";

function Metrics({ value }: {value: Record<string, unknown>}) {
  return <div className="experiment-metrics">{Object.entries(value).map(([key, item]) =>
    item && typeof item === "object" && !Array.isArray(item)
      ? <section className="metric-group" key={key}><h3>{key.replaceAll("_", " ")}</h3><Metrics value={item as Record<string, unknown>} /></section>
      : <div className="metric-cell" key={key}><span>{key.replaceAll("_", " ")}</span><strong>{typeof item === "number" ? Number.isInteger(item) ? item : item.toFixed(3) : Array.isArray(item) ? item.map(v => typeof v === "number" ? v.toFixed(3) : String(v)).join(" – ") : String(item ?? "—")}</strong></div>
  )}</div>;
}

export default function Experiments() {
  const [data, setData] = useState<ExperimentRuns | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<{title: string; summary: Record<string, unknown> | null; records: Record<string, unknown>[]} | null>(null);
  const [source, setSource] = useState("active");
  const [recordIndex, setRecordIndex] = useState(0);
  const resultsRef = useRef<HTMLElement>(null);
  const detailRequest = useRef(0);
  useEffect(() => () => { detailRequest.current++; }, [source]);
  useEffect(() => { if (detail) resultsRef.current?.scrollIntoView({behavior: "smooth", block: "start"}); }, [detail]);
  const [opening, setOpening] = useState(false);
  async function openExperiment(key: string) {
    const requestId = ++detailRequest.current;
    setOpening(true); setError(null); setDetail(null); setRecordIndex(0);
    try {
      const response = await fetch(`${API_BASE}/ui/experiments/${key}?source=${source}`);
      if (!response.ok) throw new Error(`Could not open results (HTTP ${response.status})`);
      const result = await response.json();
      if (requestId === detailRequest.current) setDetail(result);
    } catch (e) { if (requestId === detailRequest.current) setError(e instanceof Error ? e.message : "Could not open results"); }
    finally { if (requestId === detailRequest.current) setOpening(false); }
  }
  useEffect(() => {
    let cancelled = false;
    const refresh = () => getExperimentRuns(source).then(result => {if (!cancelled) {setData(result); setError(null);}}).catch(e => {if (!cancelled) setError(e.message);});
    refresh(); const timer = window.setInterval(refresh, 15000);
    return () => {cancelled = true; window.clearInterval(timer);};
  }, [source]);
  return <main className="page-content experiments-page">
    <div className="page-heading"><div><span className="eyebrow">EVALUATION</span><h1>Experiments</h1><p>Ordered runs, claims, and reproducibility metadata.</p></div></div>
    <select className="run-picker" aria-label="Run selection" value={source} onChange={e => {detailRequest.current++; setSource(e.target.value); setData(null); setDetail(null); setOpening(false);}}><option value="active">Latest run</option><option value="legacy">Previous saved results</option></select>
    <ResearchDashboard source={source} onInspect={key => void openExperiment(key)} />
    {data && <div className="experiment-overview"><div><strong>{data.experiments.filter(e => e.status === "complete").length} / {data.experiments.length}</strong><span>studies complete</span></div><div><strong>{data.experiments.reduce((sum, e) => sum + (e.n_recorded || 0), 0)}</strong><span>examples recorded</span></div><div><strong>{data.monitor?.status || "Saved results"}</strong><span>Refreshes every 15 seconds</span></div></div>}
    {error && <div className="empty-state">{error}</div>}
    {!error && !data && <div className="empty-state">Loading experiment status…</div>}
    {opening && <p role="status">Loading saved results...</p>}
    {detail && <section ref={resultsRef} className="experiment-detail" aria-label="Experiment results">
      <h2>{detail.title}</h2><h3>Key findings</h3>
      {detail.summary ? <Metrics value={(detail.summary.headline || {}) as Record<string, unknown>} /> : <p>Summary pending. Recorded examples are available below.</p>}
      {detail.summary && Object.entries(detail.summary).filter(([key, value]) => key !== "headline" && value && typeof value === "object").map(([key, value]) => <details className="study-section" key={key}><summary>{key.replaceAll("_", " ")}</summary>{Array.isArray(value) ? <ul>{value.map((v, i) => <li key={i}>{typeof v === "object" ? JSON.stringify(v) : String(v)}</li>)}</ul> : <Metrics value={value as Record<string, unknown>} />}</details>)}
      <h3>Example explorer</h3>
      {detail.records.length > 0 ? <><div className="record-controls"><button className="btn btn-secondary" disabled={recordIndex === 0} onClick={() => setRecordIndex(i => i - 1)}>Previous example</button><span>{recordIndex + 1} / {detail.records.length}</span><button className="btn btn-secondary" disabled={recordIndex + 1 === detail.records.length} onClick={() => setRecordIndex(i => i + 1)}>Next example</button></div><Metrics value={Object.fromEntries(Object.entries(detail.records[recordIndex]).filter(([, v]) => typeof v !== "object"))} /><details className="study-section"><summary>Full example data</summary><pre>{JSON.stringify(detail.records[recordIndex], null, 2)}</pre></details></> : <p>No examples recorded yet.</p>}
      <details className="study-section"><summary>Raw summary</summary><pre>{JSON.stringify(detail.summary, null, 2)}</pre></details>
    </section>}
    {data && <>
      <div className="experiment-run-meta">{data.run ? `Run ${String(data.run.run_id)} · ${String(data.run.mode || "unknown")} mode` : "No suite run recorded yet"}</div>
      <div className="experiment-list">{data.experiments.map((exp) => <article className="experiment-card" key={exp.key}>
        <div><span className="experiment-number">E{exp.number}</span><strong>{exp.title}</strong></div>
        <p>{exp.claim}</p><span className={`status-tag ${exp.status}`}>{exp.status}</span>
        <small>{exp.n_recorded || 0} / {exp.n_planned || "—"} examples · {exp.mode || "pending"}</small>
        <button className="btn btn-secondary" onClick={() => openExperiment(exp.key)}>View results for E{exp.number}</button>
      </article>)}</div>
    </>}
  </main>;
}
