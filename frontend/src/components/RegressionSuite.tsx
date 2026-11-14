import {useEffect, useRef, useState} from "react";
import {getRegressionCases, runRegressionCases, type CaseResult, type SavedCase} from "../lib/api";

export default function RegressionSuite() {
  const [cases, setCases] = useState<SavedCase[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [mode, setMode] = useState("generate");
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<CaseResult[] | null>(null);
  function load(signal?: AbortSignal) {
    getRegressionCases(signal).then(value => {
      if (signal?.aborted) return;
      if (!Array.isArray(value)) throw new Error("Invalid regression-case response");
      setCases(value); setSelected(value.slice(0, 20).map(c => c.case_id)); setError("");
    }).catch(e => { if (e.name !== "AbortError") setError(e.message); });
  }
  useEffect(() => {const controller = new AbortController(); load(controller.signal); return () => controller.abort();}, []);
  async function run() {
    if (busyRef.current) return;
    busyRef.current = true; setBusy(true); setError(""); setResults(null);
    try {setResults((await runRegressionCases(selected, mode)).cases);}
    catch (e) {setError((e as Error).message);}
    finally {busyRef.current = false; setBusy(false);}
  }
  return <section className="card regression-suite" aria-label="Regression test cases">
    <h2>Regression cases</h2>
    <p>Save expectations beside an answer, review the case files in <code>eval/regression_cases</code>, then commit them for PR checks.</p>
    <button type="button" disabled={busy} onClick={() => load()}>Refresh cases</button>
    {error && <p role="alert">{error}</p>}
    {!cases.length && <p>No saved cases.</p>}
    <fieldset disabled={busy}>
      {cases.map(c => <label className="replay-check" key={c.case_id}><input type="checkbox" checked={selected.includes(c.case_id)} onChange={e => setSelected(v => e.target.checked ? [...v, c.case_id] : v.filter(id => id !== c.case_id))} />{c.name}</label>)}
      <label>Evaluation mode <select aria-label="Evaluation mode" value={mode} onChange={e => setMode(e.target.value)}><option value="generate">Regenerate on captured document evidence</option><option value="retrieve">Run current retrieval and generation (requires corpus)</option><option value="snapshot">Check saved answer only</option></select></label>
      <p>{mode === "snapshot" ? "Snapshot checks do not exercise current generation code. Semantic assertions can load local NLI." : "This sends the case question and document evidence to NVIDIA. Full retrieval also uses the current corpus. New evaluation traces are saved; chat memory is unchanged."}</p>
      <button className="btn btn-primary" type="button" disabled={selected.length === 0 || selected.length > 20} onClick={() => void run()}>{busy ? "Running cases…" : `Run selected cases (${selected.length})`}</button>
    </fieldset>
    {busy && <p role="status">Evaluating… This may take several minutes.</p>}
    {results && <div aria-label="Regression results"><p role="status">{results.filter(r => r.passed).length}/{results.length} cases passed.</p>
      {results.map(r => <details key={r.case_id} open={!r.passed}><summary>{r.passed ? "PASS" : "FAIL / UNAVAILABLE"} — {r.name}</summary>
        {r.change && <p>Compared with the saved answer: {r.change}</p>}
        <ul>{r.checks.map((c, i) => <li key={i}>{c.status}: {c.kind} {c.assertion ?? ""} — {c.detail}</li>)}</ul>
        {r.answer != null && <details><summary>Evaluated answer and evidence IDs</summary><p className="replay-answer">{r.answer}</p><p>{r.evidence_ids?.join(", ")}</p></details>}
      </details>)}
    </div>}
  </section>;
}
