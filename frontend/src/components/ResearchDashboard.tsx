import { useEffect, useState } from "react";
import { API_BASE } from "../lib/api";

interface Study {
  key: string; title: string; mode: string; n_examples: number | null;
  headline: Record<string, unknown>; caveats: string[]; source_path: string; arm_counts: Record<string, number>;
}
interface Evidence {
  source: string; directory: string; note: string; warnings: string[];
  run: {created_at?: string; seed?: number; mode?: string} | null; studies: Study[];
}
const object = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const number = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const percent = (value: number | null) => value === null ? "Not measured" : `${(value * 100).toFixed(1)}%`;

function Bars({title, rows, unit = "%", maximum = 1}: {title: string; rows: {label: string; value: number | null}[]; unit?: string; maximum?: number}) {
  const measured = rows.filter((r): r is {label: string; value: number} => r.value !== null);
  if (!measured.length) return null;
  return <div className="research-chart"><h3>{title}</h3><div className="research-axis">{unit === "%" ? "Rate (%) · higher is better" : `Mean duration (${unit}) · lower is better`}</div>
    {measured.map(row => <div className="research-bar-row" key={row.label}>
      <span>{row.label}</span><div className="research-track"><div style={{width: `${Math.max(0, Math.min(100, row.value / maximum * 100))}%`}} /></div>
      <strong>{unit === "%" ? percent(row.value) : `${row.value.toFixed(1)} ${unit}`}</strong>
    </div>)}<div className="research-scale"><span>0 {unit}</span><span>{unit === "%" ? 100 : maximum.toFixed(0)} {unit}</span></div>
  </div>;
}

export default function ResearchDashboard({source, onInspect}: {source: string; onInspect: (key: string) => void}) {
  const [state, setState] = useState<{source: string; data?: Evidence; error?: string}>({source});
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/ui/research-summary?source=${encodeURIComponent(source)}`, {signal: controller.signal})
      .then(async response => {if (!response.ok) throw new Error(`Evidence could not load (HTTP ${response.status})`); return response.json() as Promise<Evidence>;})
      .then(data => {if (!controller.signal.aborted) setState({source, data});})
      .catch(error => {if (!controller.signal.aborted) setState({source, error: String(error.message)});});
    return () => controller.abort();
  }, [source]);
  const data = state.source === source ? state.data : undefined;
  if (state.source === source && state.error) return <p role="alert">{state.error}</p>;
  if (!data) return <p role="status">Reading research evidence…</p>;
  if (!data.studies.length) return <p>No saved study summaries exist for this run. Run the experiment suite to collect evidence.</p>;
  const find = (prefix: string) => data.studies.find(s => s.key.startsWith(prefix));
  const h = (prefix: string) => find(prefix)?.headline ?? {};
  const injected = number(h("exp01").overall_recovery_accuracy);
  const reviewed = number(h("exp08").n_accuracy_eligible_rows) !== null;
  const real = number(reviewed ? h("exp08").reviewed_accuracy : h("exp08").overall_accuracy_all_40_rows);
  const realLabel = reviewed ? "Reviewed diagnostic accuracy" : "Legacy E8 reference agreement";
  const falsePositive = number(h("exp08").false_positive_rate_on_healthy);
  const external = number(h("exp13").overall_collapsed_accuracy);
  const bestRecall = object(h("exp06").best_arm_by_document_recall);
  const bestSupport = h("exp07").best_claim_support_rate;
  const cost = h("exp12");
  const latency = [
    {label: "X-RAG · NLI stage only", value: number(cost.xrag_verification_mean_latency_ms)},
    {label: "RAGAS · saved evaluation", value: number(cost.ragas_mean_latency_ms)},
    {label: "ARES · saved evaluation", value: number(cost.ares_mean_latency_ms)},
  ].map(row => ({...row, value: row.value === null ? null : row.value / 1000}));
  const small = data.studies.filter(s => s.n_examples !== null && s.n_examples < 50);
  return <section className="research-dashboard" aria-label="Research evidence dashboard">
    <header className="research-heading"><div><span className="eyebrow">RESEARCH EVIDENCE</span><h2>What works. What still needs proof.</h2></div><span className="tag">{data.studies.length} saved studies</span></header>
    <p className="research-provenance">{data.directory} · {data.run?.created_at?.slice(0, 10) ?? "Legacy snapshot"} · seed {data.run?.seed ?? "not recorded"}</p>
    <p className="research-note">{data.note}</p>
    {!reviewed && find("exp08") && <p className="research-warning">E8 is a historical summary without explicit trace-review coverage. Its agreement and false-positive figures are legacy reference-label comparisons, not established diagnostic accuracy. Re-run E8 with independently reviewed trace labels before making accuracy claims.</p>}
    {small.length > 0 && <p className="research-warning" role="note">{small.length} studies have fewer than the suite’s 50-example research floor. Treat these as pilot results; records within a study may share questions or fixtures.</p>}
    {data.warnings.map(warning => <p key={warning} role="status">{warning}</p>)}
    <div className="research-kpis">
      {[{label:"Injected-fault recovery", value:injected, prefix:"exp01", scope:"Controlled stage signals", n:find("exp01")?.n_examples}, {label:realLabel,value:real,prefix:"exp08",scope:reviewed ? "Reviewed execution references" : "Unvalidated historical labels",n:reviewed ? number(h("exp08").n_accuracy_eligible_rows) : find("exp08")?.n_examples}, {label:"Healthy-label false positives",value:falsePositive,prefix:"exp08",scope:"Lower is better; E8 reference labels",n:number(h("exp08").n_healthy)}, {label:"External NLI accuracy",value:external,prefix:"exp13",scope:"FEVER subset; verifier only",n:find("exp13")?.n_examples}].filter(k => k.value !== null).map(k => <button className="research-kpi" key={k.label} onClick={() => onInspect(find(k.prefix)!.key)}><span>{k.label}</span><strong>{percent(k.value)}</strong><span>{k.scope} · n={k.n ?? "unknown"}</span></button>)}
    </div>
    <div className="research-panels">
      <section><Bars title="Diagnosis: controlled vs. execution references" rows={[{label:"Injected-fault recovery (E1)",value:injected},{label:`${realLabel} (E8)`,value:real}]} /><p>Different tasks and reference quality; not a paired comparison or evidence of general diagnostic accuracy.</p>{Array.isArray(h("exp01").accuracy_ci95) && <p>E1 reported 95% interval: {(h("exp01").accuracy_ci95 as number[]).map(v => percent(v)).join("–")}.</p>}</section>
      <section><Bars title="Document recall by retrieval strategy (E6)" rows={[{label:`${String(h("exp06").baseline_arm ?? "Baseline")} · n=${find("exp06")?.arm_counts[String(h("exp06").baseline_arm)] ?? "?"}`,value:number(h("exp06").baseline_document_recall)},{label:`${String(bestRecall.arm ?? "Best saved arm")} · n=${find("exp06")?.arm_counts[String(bestRecall.arm)] ?? "?"}`,value:number(bestRecall.document_recall)}]} /><Bars title="Supported-claim fraction (E7)" rows={[{label:`${String(h("exp07").baseline_arm ?? "Baseline")} · n=${find("exp07")?.arm_counts[String(h("exp07").baseline_arm)] ?? "?"}`,value:number(h("exp07").baseline_claim_support_rate)},{label:Array.isArray(bestSupport) ? `${String(bestSupport[0])} · n=${find("exp07")?.arm_counts[String(bestSupport[0])] ?? "?"}` : "Best saved arm",value:Array.isArray(bestSupport) ? number(bestSupport[1]) : null}]} /><p>Best arm selected on this saved set; arm coverage may differ. These descriptive means are not a paired superiority test. E6 and E7 are separate studies; support is not factual accuracy.</p></section>
    </div>
    <div className="research-panels">
      <section><Bars title="Saved evaluation time (E12)" rows={latency} maximum={Math.max(1,...latency.map(r => r.value ?? 0))} unit="s" /><p>Scopes differ: X-RAG excludes claim decomposition and other pipeline stages. These bars do not establish an end-to-end speedup or dollar-cost saving.</p></section>
      <section><h3>Refusal can improve the metric and lose the answer (E3)</h3><dl className="research-deltas">{[{label:"Faithfulness change when refusing",value:number(h("exp03").faithfulness_advantage_of_refusing)},{label:"Answer recall change when refusing",value:number(h("exp03").recall_cost_of_refusing)},{label:"Silence-adjusted score change",value:number(h("exp03").silence_adjusted_advantage_of_refusing)}].filter(r => r.value !== null).map(r => <div key={r.label}><dt>{r.label}</dt><dd>{r.value! > 0 ? "+" : ""}{(r.value! * 100).toFixed(1)} pp</dd></div>)}</dl><p>Percentage-point deltas from the saved study. Report coverage and refusals alongside faithfulness.</p></section>
    </div>
    <details className="research-sources"><summary>Inspect sources, sample sizes and study caveats</summary><div className="research-table-wrap"><table><thead><tr><th>Study</th><th>Mode label</th><th>Records</th><th>Evidence</th></tr></thead><tbody>{data.studies.map(study => <tr key={study.key}><td><button className="research-link" onClick={() => onInspect(study.key)}>{study.title}</button></td><td>{study.mode}</td><td>{study.n_examples ?? "Unknown"}</td><td><code>{study.source_path}</code>{study.caveats.map(c => <p key={c}>{c}</p>)}</td></tr>)}</tbody></table></div></details>
  </section>;
}
