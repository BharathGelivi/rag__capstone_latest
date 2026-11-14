import {useEffect, useRef, useState} from "react";
import {API_BASE, type EvidenceDebugData, type TraceChunkReference} from "../lib/api";

export default function EvidenceDetails({traceId, data}: {traceId: string; data: EvidenceDebugData}) {
  const [selected, setSelected] = useState<string | null>(null);
  const sourcePanel = useRef<HTMLElement>(null);
  useEffect(() => {
    if (selected) {
      sourcePanel.current?.scrollIntoView({behavior: "smooth", block: "nearest"});
      sourcePanel.current?.focus({preventScroll: true});
    }
  }, [selected]);
  const allSources = [...data.sources, ...data.excluded_candidates];
  const source = allSources.find(c => c.chunk_id === selected) as (TraceChunkReference & {chunk_text?: string}) | undefined;
  const text = source?.text || source?.chunk_text;
  return <div className="evidence-debugger" aria-label="Sentence evidence debugger">
    <h3>Answer sentences → claims → passages</h3>
    <p className="text-muted">{data.note}</p>
    <ul>{data.signals.map(s => <li key={s.kind}><strong>{s.kind}</strong>: {s.message}</li>)}</ul>
    {data.sentences.map(s => <article className="evidence-sentence" key={s.sentence_id}>
      <h4>{s.sentence_id}</h4><p className="replay-answer">{s.text}</p>
      {!s.claims.length && <p>No claim mapping available. This sentence has not been established as supported.</p>}
      {s.claims.map(c => <div key={c.claim_id}>
        <button type="button" onClick={() => setSelected(c.best_chunk_id)} disabled={!c.best_chunk_id}>{c.status}: {c.claim_text}</button>
        <p className="text-muted">Opens the verifier's best passage. It can differ from the cited passage.</p>
        {c.citation_checks?.map(check => <p key={check.chunk_id}>
          <button type="button" onClick={() => setSelected(check.chunk_id)}>Cited passage {check.chunk_id}</button>: <strong>{check.status}</strong> — {check.reason}
        </p>)}
      </div>)}
      {s.citations.length > 0 && !s.claims.some(c => c.citation_checks?.length) && <p>Citation support has not been assessed for this sentence.</p>}
      {s.citations.map(cid => <button key={cid} type="button" onClick={() => setSelected(cid)}>Open cited source {cid}</button>)}
    </article>)}
    {!!data.unmapped_claim_ids.length && <p>{data.unmapped_claim_ids.length} claims could not be tied to a unique answer span. They remain visible in the claim list.</p>}
    {selected && <section ref={sourcePanel} className="card captured-source" aria-label="Selected source passage" tabIndex={0}>
      <h3>Captured source passage</h3>
      {source ? <>
        <p><strong>{source.source_file || source.chunk_id}</strong> — Page {source.page_number || "not recorded"}</p>
        <p><code>{source.chunk_id}</code></p>
        {text ? <blockquote className="replay-answer">{text}</blockquote> : <p>Full captured text is unavailable.</p>}
        {source.source_file?.toLowerCase().endsWith(".pdf") && <a target="_blank" rel="noreferrer" href={`${API_BASE}/ui/trace/${encodeURIComponent(traceId)}/source/${encodeURIComponent(source.chunk_id)}${Number.isInteger(Number(source.page_number)) && Number(source.page_number) > 0 ? `#page=${source.page_number}` : ""}`}>Open current source PDF{Number.isInteger(Number(source.page_number)) && Number(source.page_number) > 0 ? " at this page" : ""}</a>}
        <p className="text-muted">The captured text is the run's evidence. A current PDF can differ or be unavailable.</p>
      </> : <p>The cited passage was not captured in this trace. Its presence in the answer does not establish support.</p>}
    </section>}
    <details><summary>Ranked candidates excluded from the prompt ({data.excluded_candidates.length})</summary>
      <p>{data.candidate_capture === "unavailable" ? "Candidate snapshots are unavailable for this older trace." : data.candidate_scope}</p>
      <p>These passages were ranked but not sent in the final prompt. Ranking scores alone do not establish relevance or support.</p>
      {data.excluded_candidates.map(c => <article key={c.chunk_id} className="replay-evidence">
        <button type="button" onClick={() => setSelected(c.chunk_id)}>{c.source_file || c.chunk_id} — page {c.page_number}</button>
        <p>Candidate rank: {c.candidate_rank ?? "not recorded"}; reranker score: {c.reranker_score ?? "not recorded"}</p>
        {c.observations && <details><summary>Retrieval observations ({c.observations.length})</summary>{c.observations.map((o, i) => <p key={i}>Query: {o.retrieval_query ?? "not recorded"}; candidate rank: {o.candidate_rank ?? "not recorded"}; reranker: {o.reranker_score ?? "not recorded"}</p>)}</details>}
        <p className="replay-answer">{c.text || c.chunk_text}</p>
      </article>)}
    </details>
  </div>;
}
