import {useEffect, useState} from "react";
import {getTrace, type TraceData} from "../lib/api";
import SaveRegressionCase from "./SaveRegressionCase";
import EvidenceExplorer from "./EvidenceExplorer";

export default function AnswerTools({traceId, busy}: {traceId?: string; busy: boolean}) {
  const [open, setOpen] = useState(false);
  const [trace, setTrace] = useState<TraceData | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    if (open && traceId) getTrace(traceId, controller.signal).then(value => {
      if (!controller.signal.aborted) setTrace(value);
    }).catch(e => {if (e.name !== "AbortError") setError(e.message);});
    return () => controller.abort();
  }, [open, traceId]);
  return <div className="answer-tools">
    <SaveRegressionCase traceId={traceId} disabled={busy} />
    <button type="button" className="retry-btn" disabled={!traceId || busy} onClick={() => setOpen(v => !v)}>Evidence debugger</button>
    {open && <div className="answer-evidence-panel">{error && <p role="alert">{error}</p>}{trace ? <EvidenceExplorer trace={trace} /> : <p>Loading evidence…</p>}</div>}
  </div>;
}
