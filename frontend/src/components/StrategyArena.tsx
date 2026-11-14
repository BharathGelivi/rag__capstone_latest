import {useRef, useState} from "react";
import {toolsRequest} from "../lib/api";
import EvidenceExplorer from "./EvidenceExplorer";
import type {TraceData} from "../lib/api";

export interface ArenaResult {
  rows: {source_trace_id: string; question: string; results: {arm: string; error?: string; trace?: TraceData; supported_fraction?: number; latency_seconds?: number; estimated_input_tokens?: number; estimated_output_tokens?: number; estimated_generation_cost_usd?: number; cost_scope?: string}[]}[];
  paired: {baseline: string; challenger: string; n: number; mean_difference: number | null; bootstrap_95_percent: number[] | null}[];
  uncertainty_note: string;
}
export function ArenaResults({result}: {result: ArenaResult}) {
  return <div aria-label="Strategy comparison results"><p>{result.uncertainty_note}</p>
    {result.paired.map(p => <p key={p.baseline + p.challenger}>{p.challenger} minus {p.baseline}: {p.mean_difference?.toFixed(3) ?? "unavailable"}; paired n={p.n}; 95% interval {p.bootstrap_95_percent?.map(x => x.toFixed(3)).join(" to ") ?? "unavailable"}</p>)}
    {result.rows.map(row => <section key={row.source_trace_id}><h3>{row.question}</h3><div className="arena-grid">{row.results.map(r => <article className="card" key={r.arm}><h4>{r.arm}</h4>{r.error ? <p role="alert">{r.error}</p> : <><p className="replay-answer">{r.trace?.generated_answer}</p><p>{r.latency_seconds?.toFixed(2)} s; supported claims {r.supported_fraction?.toFixed(3) ?? "unavailable"}</p><p>Estimated tokens {r.estimated_input_tokens} in / {r.estimated_output_tokens} out; estimated ${r.estimated_generation_cost_usd?.toFixed(6)}</p><p>{r.cost_scope}</p><EvidenceExplorer trace={r.trace ?? null}/></>}</article>)}</div></section>)}
  </div>;
}
export default function StrategyArena({traceId}: {traceId?: string}) {
  const [ids, setIds] = useState(traceId ?? "");
  const [arms, setArms] = useState("A_vector\nB_hybrid\nC_hybrid_rerank");
  const [inputRate, setInputRate] = useState(0), [outputRate, setOutputRate] = useState(0);
  const [result, setResult] = useState<ArenaResult | null>(null), [error, setError] = useState("");
  const [busy, setBusy] = useState(false); const lock = useRef(false);
  async function run() {
    if (lock.current) return; lock.current = true; setBusy(true); setError("");
    try {setResult(await toolsRequest<ArenaResult>("/ui/arena/run", {trace_ids: ids.split(/\s+/).filter(Boolean), arms: arms.split(/\s+/).filter(Boolean), input_per_million: inputRate, output_per_million: outputRate}));}
    catch(e) {setError((e as Error).message);} finally {lock.current = false; setBusy(false);}
  }
  return <section className="card engineering-tool" aria-label="Strategy comparison arena"><h2>Strategy comparison arena</h2><p>Same saved questions across strategies. Runs fresh retrieval and sends captured conversation and new evidence to configured inference. Up to 20 questions and 6 strategies; incompatible arms report errors.</p><fieldset disabled={busy}>
    <label>Question trace IDs<textarea value={ids} onChange={e => setIds(e.target.value)}/></label><label>Strategy IDs<textarea value={arms} onChange={e => setArms(e.target.value)}/></label>
    <p>A_vector, A_bm25, B_hybrid, C_hybrid_rerank; judgments also allow C_fixed_chunking, D_ircot, E_agentic, F_graphrag, G_ircot_graph, H_agentic_graph, I_full. Graph and fixed chunking require judgments; both corpora support C_hybrid_rerank only.</p>
    <label>Input USD per million tokens<input type="number" min="0" value={inputRate} onChange={e => setInputRate(Number(e.target.value))}/></label><label>Output USD per million tokens<input type="number" min="0" value={outputRate} onChange={e => setOutputRate(Number(e.target.value))}/></label><p>Rates default to zero until entered. Estimates exclude planner and judge usage.</p><button onClick={() => void run()}>{busy ? "Comparing…" : "Compare strategies"}</button></fieldset>{error && <p role="alert">{error}</p>}{result && <ArenaResults result={result}/>}</section>;
}
