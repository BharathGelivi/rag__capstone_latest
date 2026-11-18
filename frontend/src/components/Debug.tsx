import EvidenceExplorer from "./EvidenceExplorer";
import TraceReplay from "./TraceReplay";
import RegressionSuite from "./RegressionSuite";
import StrategyArena from "./StrategyArena";
import { useEffect, useState } from "react";
import { getDebugSuggestions, getLatestTrace, getTrace, type DebugSuggestions, type TraceData } from "../lib/api";
import type { RuntimeSettings } from "../App";

interface DebugProps {
  settings: RuntimeSettings;
  onSettingsChange: (s: RuntimeSettings) => void;
  lastTraceId: string | null;
}

function toggle(label: string, on: boolean, onClick: () => void) {
  return (
    <div key={label} className="debug-toggle-row">
      <span>{label}</span>
      <button role="switch" aria-checked={on} aria-label={label} className={`debug-toggle ${on ? "on" : ""}`} onClick={onClick}>
        <span className="debug-toggle-knob" />
      </button>
    </div>
  );
}

function TuningSuggestions({ traceId }: { traceId?: string }) {
  const [data, setData] = useState<DebugSuggestions>();
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    getDebugSuggestions(traceId, controller.signal)
      .then(setData)
      .catch(err => { if (!controller.signal.aborted) setError(err.message); });
    return () => controller.abort();
  }, [traceId, attempt]);

  const priority = data?.ranked[0];
  const current = data?.current;
  return <>
    <section className="card elev-sm" aria-label="Trace tuning suggestion">
      <div className="card-kicker">Trace tuning suggestion</div>
      {!data && !error && <p role="status">Checking tuning options…</p>}
      {error && <p role="alert">{error} Compare a baseline run with a single retrieval setting changed and record evidence coverage and latency.</p>}
      {current && <>
        <strong>{current.title}</strong>
        <p>{current.experiment}</p>
        <p className="text-muted">{current.measure}</p>
        <p className="text-muted">{{
          diagnostic: "From a saved diagnostic plan",
          llm: "LLM-proposed experiment based on trace telemetry",
          trace_rule: "Trace-based baseline experiment",
          fallback: "Baseline experiment. The tuning model was unavailable.",
        }[current.source]}</p>
      </>}
      {data && !current && <p>Run a baseline chat question with known evidence to establish a tuning target.</p>}
      {(error || current?.source === "fallback") && <button className="btn btn-secondary" onClick={() => {
        setData(undefined);
        setError("");
        setAttempt(value => value + 1);
      }}>Retry suggestion</button>}
    </section>
    <section className="card elev-sm" aria-label="Deployment tuning recommendation">
      <div className="card-kicker">Deployment tuning recommendation</div>
      {data && <p className="text-muted">Based on {data.trace_count} saved chat {data.trace_count === 1 ? "trace" : "traces"} in this deployment.</p>}
      {priority && <>
        <strong>{priority.suggestion.title}</strong>
        <p>{priority.suggestion.experiment}</p>
        <p className="text-muted">{priority.suggestion.measure}</p>
        <details>
          <summary>Suggestions across saved chats</summary>
          <ul>{data.ranked.map(row => <li key={row.suggestion.title}>
            {row.suggestion.title} · {row.trace_count} {row.trace_count === 1 ? "trace" : "traces"} · {row.evidence_count} from diagnostics or the tuning model
          </li>)}</ul>
        </details>
      </>}
      {data && !priority && <p>Start with a set of baseline questions and record whether the expected passages are retrieved.</p>}
    </section>
  </>;
}

export default function Debug({ settings, onSettingsChange, lastTraceId }: DebugProps) {
  const [trace, setTrace] = useState<TraceData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    Promise.resolve().then(() => { if (!controller.signal.aborted) { setLoading(true); setError(""); } });
    const load = lastTraceId ? getTrace(lastTraceId, controller.signal) : getLatestTrace(controller.signal);
    load.then(value => { if (!controller.signal.aborted) setTrace(value); })
      .catch(err => { if (err.name !== "AbortError") setError("Unable to load trace"); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [lastTraceId]);

  const stats = (trace?.execution_statistics || {}) as Record<string, number>;
  const total = stats.total_pipeline_time || (stats.retrieval_time || 0) + (stats.generation_time || 0) || 1;
  const traceRows: Array<[string, number | undefined]> = [
    ["Retrieval", stats.retrieval_time],
    ["Generation", stats.generation_time],
  ];

  return (
    <div className="debug-page xrag-scroll">
      <span className="topbar-title">Debug</span>
      {error && <p role="alert">{error}</p>}

      <div className="card elev-sm">
        <div className="card-kicker">Runtime toggles</div>
        <div className="debug-toggles">
          {toggle("Memory recall", settings.memoryEnabled, () =>
            onSettingsChange({ ...settings, memoryEnabled: !settings.memoryEnabled }))}
          {toggle("Deep analysis (claim verification)", settings.deepAnalysis, () =>
            onSettingsChange({ ...settings, deepAnalysis: !settings.deepAnalysis }))}
          {toggle("Auto-expand answer details", settings.autoExpandDetails, () =>
            onSettingsChange({ ...settings, autoExpandDetails: !settings.autoExpandDetails }))}
        </div>
      </div>

      <div className="card elev-sm">
        <div className="card-kicker">
          {trace ? `Last trace — ${trace.trace_id}` : loading ? "Last trace" : "Last trace — none recorded yet"}
        </div>
        {trace && (
          <div className="debug-trace-rows">
            {traceRows.map(([label, val]) => (
              <div key={label} className="debug-trace-row">
                <span className="debug-trace-label">{label}</span>
                <div className="debug-trace-track">
                  <div className="debug-trace-fill" style={{ width: `${Math.min(100, ((val || 0) / total) * 100)}%` }} />
                </div>
                <span className="debug-trace-time">{val != null ? (val < 1 ? `${Math.round(val * 1000)} ms` : `${val.toFixed(2)} s`) : "—"}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {!loading && <TuningSuggestions key={trace?.trace_id ?? "none"} traceId={trace?.trace_id} />}
      <TraceReplay key={trace?.trace_id ?? "none"} initialTraceId={trace?.trace_id} />
      <RegressionSuite />
      <StrategyArena key={`arena-${trace?.trace_id}`} traceId={trace?.trace_id}/>
      <EvidenceExplorer trace={trace} />
      <div className="card elev-sm">
        <div className="card-kicker">Prompt sent to model</div>
        <pre className="debug-pre">{trace?.prompt_snapshot || (loading ? "Loading…" : "No trace available.")}</pre>
      </div>

      <div className="card elev-sm">
        <div className="card-kicker">Raw retrieval metadata</div>
        <pre className="debug-pre">
          {trace ? JSON.stringify(trace.retrieved_chunk_references, null, 2) : (loading ? "Loading…" : "No trace available.")}
        </pre>
      </div>
    </div>
  );
}
