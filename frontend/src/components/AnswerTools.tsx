import SaveRegressionCase from "./SaveRegressionCase";

export default function AnswerTools({traceId, busy}: {traceId?: string; busy: boolean}) {
  return <div className="answer-tools"><SaveRegressionCase traceId={traceId} disabled={busy} /></div>;
}
