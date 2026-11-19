import { useEffect, useRef, useState } from "react";
import AnswerTools from "./AnswerTools";
import {
  API_BASE,
  createSession,
  getConfig,
  getFrameworkPanel,
  getSessionMessages,
  getTrace,
  renameSession,
  saveSessionReport,
  streamChat,
  type ChatEvent,
  type ChatEvaluation,
  type ClaimVerification,
  type Config,
  type FrameworkPanel,
} from "../lib/api";
import type { RuntimeSettings } from "../App";

/* ─── Types ─────────────────────────────────────────────── */
interface Message {
  role: "user" | "assistant";
  content: string;
  meta?: MsgMeta;           // only on assistant messages
  failedQuestion?: string;  // set on error so the turn can be retried
}

interface Chunk {
  chunk_id: string;
  source_file: string;
  page_number: string | number;
  similarity_score: number;
  text: string;
}

interface MemoryRecall {
  memory_id?: string;
  kind?: string;   // "turn" | "fact" | "summary"
  question: string;
  answer: string;
  session_id: string;
  timestamp: string;
  semantic_score: number;
  recency_score: number;
  frequency_score: number;
  importance_score: number;
  final_score: number;
}

interface MsgMeta extends ChatEvaluation {
  arm?: string;
  corpus?: string;
  search_query?: string;
  query_was_condensed?: boolean;
  chunks?: Chunk[];
  retrieval_metadata?: Record<string, unknown>;
  retrieval_time?: number;
  generation_time?: number;
  memory_time?: number;
  total_time?: number;
  recalled_memories?: MemoryRecall[];
  trace_id?: string;
  // Model reasoning (gated arms only -- see REASONING_ARMS in chat_service.py)
  reasoning?: string;
  reasoningDone?: boolean;
  reasoningTimeMs?: number;
  // Claim decomposition + NLI verification (opt-in via "deep analysis" setting)
  claimCount?: number;
  claims?: ClaimVerification[];
  claimError?: string;
}

interface ChatProps {
  activeSessionId: string | null;
  onSessionCreated: (id: string) => void;
  onSessionRenamed: () => void;
  settings: RuntimeSettings;
  onTurnDone: (traceId: string) => void;
}

function buildPipelineTrace(meta: MsgMeta): string[] {
  const lines: string[] = [];
  if (meta.recalled_memories?.length) {
    lines.push(`Recalled ${meta.recalled_memories.length} related past exchange(s) from memory.`);
  }
  if (meta.query_was_condensed && meta.search_query) {
    lines.push(`Condensed this follow-up into a standalone query: "${meta.search_query}".`);
  }
  if (meta.chunks?.length) {
    const desc = meta.arm ? (ARM_DESCRIPTIONS[meta.arm] ?? meta.arm) : "the selected strategy";
    lines.push(`Retrieved ${meta.chunks.length} passages via ${desc}.`);
  }
  return lines;
}

/* ─── Constants ─────────────────────────────────────────── */
const SUGGESTIONS = [
  "What is the punishment for murder?",
  "What are the rights of an arrested person?",
  "Define culpable homicide not amounting to murder",
  "When is secondary evidence admissible?",
];

const ARM_DESCRIPTIONS: Record<string, string> = {
  A_vector: "Pure dense vector (cosine) search",
  A_bm25: "BM25 keyword-only search",
  B_hybrid: "Dense + BM25 with RRF fusion",
  C_hybrid_rerank: "Hybrid + cross-encoder reranker (default)",
  C_fixed_chunking: "Fixed-size chunking baseline",
  D_ircot: "IRCoT — Interleaved chain-of-thought retrieval",
  E_agentic: "Agentic LLM-driven retrieval",
  F_graphrag: "GraphRAG — knowledge graph expansion",
  G_ircot_graph: "IRCoT + knowledge graph",
  H_agentic_graph: "Agentic + knowledge graph",
  I_full: "Full pipeline (all features combined)",
};

const CORPUS_LABELS: Record<string, string> = {
  statutes: "Statutes (BNS / BNSS / BSA)",
  judgments: "Supreme Court Judgments",
  both: "Statutes + Judgments (combined)",
};

function fmt(s?: number) {
  if (s == null) return "—";
  return s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`;
}

function scoreBar(v: number, color = "#10a37f") {
  const pct = Math.min(100, Math.round(v * 100));
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      <span className="score-bar-label">{v.toFixed(3)}</span>
    </div>
  );
}

const CLAIM_STATUS_COLOR: Record<ClaimVerification["status"], string> = {
  SUPPORTED: "#10a37f",
  PARTIALLY_SUPPORTED: "#f59e0b",
  CONTRADICTED: "#ef4444",
  UNSUPPORTED: "#9ca3af",
  NOT_VERIFIABLE: "#9ca3af",
};

const CLAIM_STATUS_LABEL: Record<ClaimVerification["status"], string> = {
  SUPPORTED: "✓ Supported",
  PARTIALLY_SUPPORTED: "◐ Partially supported",
  CONTRADICTED: "✗ Contradicted",
  UNSUPPORTED: "— Unsupported",
  NOT_VERIFIABLE: "? Not verifiable",
};

/* ─── DetailsPanel ───────────────────────────────────────── */
type PanelTab = "retrieval" | "memory" | "trace" | "evaluation" | "claims" | "strategy" | "frameworks";

/* ─── StrategyTab: the multi-step retrieval trace (IRCoT / agentic / graph) ── */

interface IrcotStep {
  iteration: number;
  query: string;
  is_initial: boolean;
  n_new_evidence: number;
  reasoning_summary: string;
  termination_reason?: string | null;
  latency_s?: number;
}

interface AgentStep {
  step: number;
  action: string;
  arg: string;
  why: string;
  n_new_evidence: number;
  graph_hops: number;
  relation_paths?: string[][];
  note?: string;
  latency_s?: number;
}

const AGENT_ACTION_LABEL: Record<string, string> = {
  semantic_search: "🔎 semantic search",
  lexical_search: "🔤 lexical (exact terms)",
  graph_expand: "🕸 graph expansion",
  find_citing_cases: "📚 find citing cases",
  contradiction_search: "⚖️ contradiction search",
  stop: "🛑 stop",
};

function StrategyTab({ meta }: { meta: MsgMeta }) {
  const md = (meta.retrieval_metadata ?? {}) as Record<string, unknown>;
  const ircot = (md.ircot_steps ?? null) as IrcotStep[] | null;
  const agentSteps = (md.agent_step_log ?? null) as AgentStep[] | null;
  const graphAdded = typeof md.graph_added === "number" ? md.graph_added : 0;
  const termination = (md.ircot_termination_reason ?? md.agent_termination_reason) as string | undefined;

  return (
    <div className="details-content">
      <div className="detail-note" style={{ marginBottom: 8 }}>
        {ircot ? "IRCoT — interleaved retrieval/reasoning"
          : agentSteps ? "Agentic controller over retrieval"
          : "Hybrid retrieval + citation-graph expansion"}
        {termination ? ` · stopped: ${termination}` : ""}
        {graphAdded > 0 ? ` · 🕸 graph added ${graphAdded} chunk${graphAdded === 1 ? "" : "s"}` : ""}
      </div>

      {ircot && ircot.map((s) => (
        <div key={s.iteration} className="chunk-card">
          <div className="chunk-header">
            <span className="chunk-rank">{s.is_initial ? "initial query" : `hop ${s.iteration}`}</span>
            {s.query ? <span className="chunk-source">{s.query}</span>
                     : <span className="chunk-source">planner: enough evidence</span>}
            {s.n_new_evidence > 0 && <span className="dpill">+{s.n_new_evidence} new</span>}
            {s.termination_reason && <span className="dpill text-muted">{s.termination_reason}</span>}
          </div>
          {s.reasoning_summary && <div className="detail-note">{s.reasoning_summary}</div>}
        </div>
      ))}

      {agentSteps && agentSteps.map((s) => (
        <div key={s.step} className="chunk-card">
          <div className="chunk-header">
            <span className="chunk-rank">step {s.step}</span>
            <span className="chunk-source">{AGENT_ACTION_LABEL[s.action] ?? s.action}</span>
            {s.n_new_evidence > 0 && <span className="dpill">+{s.n_new_evidence} new</span>}
            {s.graph_hops > 0 && <span className="dpill" style={{ color: "#3b82f6" }}>🕸 {s.graph_hops} hop{s.graph_hops === 1 ? "" : "s"}</span>}
          </div>
          {s.arg && <div className="chunk-id">{s.arg}</div>}
          {s.why && <div className="detail-note">{s.why}</div>}
          {s.note && <div className="detail-note">⚠ {s.note}</div>}
          {(s.relation_paths ?? []).length > 0 && (
            <div className="detail-note">
              {(s.relation_paths ?? []).map((p) => p.join(" → ")).join("   ·   ")}
            </div>
          )}
        </div>
      ))}

      {!ircot && !agentSteps && graphAdded === 0 && (
        <p className="detail-empty">Single-shot retrieval for this arm — nothing multi-step to show.</p>
      )}
    </div>
  );
}

/* ─── FrameworksTab: per-answer RAGAS / RAGChecker / ARES comparison ────── */

// Panels cost real judge calls, so they are cached per trace id across
// mount/unmount of the details panel and the Chat view.
const _panelCache = new Map<string, FrameworkPanel>();

const FAMILY_COLOR: Record<string, string> = {
  RAGAS: "#3b82f6",
  RAGChecker: "#f59e0b",
  ARES: "#ec4899",
  "this project": "#10a37f",
};

function VerdictStrip({ v }: { v: NonNullable<FrameworkPanel["verification"]> }) {
  if (!v.total) return null;
  const shaky = v.unsupported + v.contradicted;
  return (
    <div className="details-pills" style={{ marginBottom: 8 }}>
      <span className="dpill accent">🔎 {v.total} claim{v.total === 1 ? "" : "s"}</span>
      <span className="dpill" style={{ color: "#10a37f" }}>✓ {v.supported} supported</span>
      {v.partially_supported > 0 && <span className="dpill" style={{ color: "#f59e0b" }}>◐ {v.partially_supported} partial</span>}
      {shaky > 0
        ? <span className="dpill" style={{ color: "#ef4444" }}>⚠ {shaky} unsupported/contradicted</span>
        : <span className="dpill" style={{ color: "#10a37f" }}>✓ none contradicted</span>}
      {v.not_verifiable > 0 && <span className="dpill text-muted">? {v.not_verifiable} not verifiable</span>}
      <span className="dpill text-muted">avg entailment {v.average_entailment.toFixed(2)}</span>
    </div>
  );
}

function FrameworksTab({ traceId }: { traceId: string }) {
  const [panel, setPanel] = useState<FrameworkPanel | null>(() => _panelCache.get(traceId) ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = (force = false) => {
    if (!force && _panelCache.has(traceId)) { setPanel(_panelCache.get(traceId)!); return; }
    setLoading(true); setError(null);
    getFrameworkPanel(traceId)
      .then((p) => { _panelCache.set(traceId, p); setPanel(p); })
      .catch((e) => setError(String((e as Error).message ?? e)))
      .finally(() => setLoading(false));
  };

  // Opening the tab IS the ask -- auto-compute on first open, cache after.
  useEffect(() => { if (!panel && !loading && !error) load(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);

  if (loading) {
    return (
      <div className="details-content">
        <div className="msg-thinking"><span /><span /><span /></div>
        <p className="detail-note">
          Judging this answer with the framework panels (claim check + LLM-judged metrics)…
        </p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="details-content">
        <p className="detail-empty">Framework comparison failed: {error}</p>
        <button className="retry-btn" onClick={() => load(true)}>🔄 Retry</button>
      </div>
    );
  }
  if (!panel) return null;

  return (
    <div className="details-content">
      {panel.verification && <VerdictStrip v={panel.verification} />}
      {panel.computed_claims && (
        <div className="detail-note">
          🔎 Deep analysis was run on demand for this answer (claim decomposition + NLI
          verification) and is now saved with the trace.
        </div>
      )}
      {panel.families.map((fam) => (
        <div key={fam.family} style={{ marginBottom: 12 }}>
          <div className="detail-note" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span
              className="tag tag-accent"
              style={{ borderColor: FAMILY_COLOR[fam.family] ?? "#9ca3af", color: FAMILY_COLOR[fam.family] ?? "#9ca3af" }}
            >
              {fam.family === "this project" ? "X-RAG" : fam.family}
            </span>
            <span className="text-muted">
              {fam.family === "this project"
                ? "metrics defined by this framework"
                : `${fam.family}-style metrics, same definitions, local judges`}
            </span>
          </div>
          {fam.metrics.map((m) => (
            <div className="memory-score-row" key={m.key} title={`${m.definition} — judge: ${m.judge}`}>
              <span>{m.label}</span>
              {m.value == null ? (
                <span className="text-muted">— {m.requires ? `needs a ${m.requires}` : "not computed"}</span>
              ) : scoreBar(m.value, FAMILY_COLOR[fam.family] ?? "#10a37f")}
            </div>
          ))}
        </div>
      ))}
      <div className="eval-note">{panel.note}</div>
    </div>
  );
}

function DetailsPanel({ meta, defaultOpen }: { meta: MsgMeta; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(!!defaultOpen);
  const [tab, setTab] = useState<PanelTab>("retrieval");

  const chunkCount = meta.chunks?.length ?? 0;
  const memCount = meta.recalled_memories?.length ?? 0;
  const claimCount = meta.claims?.length ?? meta.claimCount ?? 0;
  const hasClaimData = meta.claims !== undefined || meta.claimError != null;
  const md = (meta.retrieval_metadata ?? {}) as Record<string, unknown>;
  const hasStrategy = Boolean(md.ircot_steps || md.agent_step_log || ((md.graph_added as number) ?? 0) > 0);
  const hasTrace = !!meta.trace_id;

  // Shaky-claim count for the headline verdict pill.
  const shakyClaims = (meta.claims ?? []).filter(
    (c) => c.status === "UNSUPPORTED" || c.status === "CONTRADICTED"
  ).length;
  const partialClaims = (meta.claims ?? []).filter(c => c.status === "PARTIALLY_SUPPORTED").length;
  const unverifiableClaims = (meta.claims ?? []).filter(c => c.status === "NOT_VERIFIABLE").length;

  return (
    <div className="details-root">
      <button className="details-toggle" aria-expanded={open} onClick={() => setOpen(o => !o)}>
        <span className="details-toggle-icon">{open ? "▾" : "▸"}</span>
        <span>Sources & analysis</span>
        <span className="details-pills">
          {chunkCount > 0 && <span className="dpill">📄 {chunkCount}</span>}
          {meta.groundedness != null && (
            <span className="dpill">Word overlap: {(meta.groundedness * 100).toFixed(0)}%</span>
          )}
          {meta.faithfulness != null && (
            <span className="dpill accent">Supported claims: {(meta.faithfulness * 100).toFixed(0)}%</span>
          )}
          {memCount > 0 && <span className="dpill">🧠 {memCount}</span>}
          {hasClaimData && <span className="dpill">🔎 {claimCount} claim{claimCount === 1 ? "" : "s"}</span>}
          {meta.claims !== undefined && claimCount > 0 && <>
            {shakyClaims > 0 && <span className="dpill" style={{ color: "#ef4444" }}>⚠ {shakyClaims} shaky</span>}
            {partialClaims > 0 && <span className="dpill">{partialClaims} partially supported</span>}
            {unverifiableClaims > 0 && <span className="dpill">{unverifiableClaims} unverifiable</span>}
            {meta.claims.every(c => c.status === "SUPPORTED") && (
              <span className="dpill" style={{ color: "#10a37f" }}>All claims supported</span>
            )}
          </>}
          {meta.total_time != null && <span className="dpill">⏱ {fmt(meta.total_time)}</span>}
          {meta.device && <span className="dpill" style={{ color: "#10a37f", fontWeight: 500 }}>⚡ {{ CPU: "CPU", MPS: "MPS" }[meta.device] ?? meta.device}</span>}
        </span>
      </button>

      {open && (
        <div className="details-panel">
          {/* Tab bar */}
          <div className="details-tabs">
            {([
              ["retrieval", `📄 Retrieval (${chunkCount})`],
              ...(hasStrategy ? [["strategy", "🧭 Strategy"] as [PanelTab, string]] : []),
              ["memory",    `🧠 Memories (${memCount})`],
              ["trace",     "🕐 RAG Trace"],
              ["evaluation","📊 Evaluation"],
              ...(hasClaimData ? [["claims", `🔎 Claims (${claimCount})`] as [PanelTab, string]] : []),
              ...(hasTrace    ? [["frameworks", "🏁 Frameworks"] as [PanelTab, string]] : []),
            ] as [PanelTab, string][]).map(([t, label]) => (
              <button
                key={t}
                className={`details-tab ${tab === t ? "active" : ""}`}
                aria-pressed={tab === t}
                onClick={() => setTab(t)}
              >{label}</button>
            ))}
          </div>

          {/* ── Retrieval ───────────────────────────── */}
          {tab === "retrieval" && (
            <div className="details-content">
              {meta.query_was_condensed && (
                <div className="detail-note">
                  ↻ Follow-up condensed to: <em>"{meta.search_query}"</em>
                </div>
              )}
              {(meta.chunks ?? []).map((c, i) => (
                <div key={c.chunk_id} className="chunk-card">
                  <div className="chunk-header">
                    <span className="chunk-rank">#{i + 1}</span>
                    <span className="chunk-source">{c.source_file}</span>
                    {c.page_number != null && (
                      <span className="chunk-page">p{c.page_number}</span>
                    )}
                    <span className="chunk-score-label">score</span>
                    {scoreBar(c.similarity_score)}
                  </div>
                  <div className="chunk-text">{c.text}</div>
                  <div className="chunk-id">ID: {c.chunk_id}</div>
                </div>
              ))}
              {chunkCount === 0 && <p className="detail-empty">No chunks retrieved.</p>}
              {meta.retrieval_metadata && Object.keys(meta.retrieval_metadata).length > 0 && (
                <details className="meta-raw">
                  <summary>Raw retrieval metadata</summary>
                  <pre>{JSON.stringify(meta.retrieval_metadata, null, 2)}</pre>
                </details>
              )}
            </div>
          )}

          {/* ── Memories ────────────────────────────── */}
          {tab === "memory" && (
            <div className="details-content">
              {(meta.recalled_memories ?? []).map((m, i) => (
                <div key={i} className="memory-card">
                  <div className="memory-q">
                    {m.kind && m.kind !== "turn" && (
                      <span className="dpill accent" style={{ marginRight: 6 }}>{m.kind}</span>
                    )}
                    {m.kind === "fact" ? m.answer : `Q: ${m.question}`}
                  </div>
                  <div className="memory-a">{m.kind === "fact" ? `(asked: ${m.question})` : `A: ${m.answer}`}{m.answer.length >= 300 ? "…" : ""}</div>
                  <div className="memory-scores">
                    <div className="memory-score-row">
                      <span>Semantic</span>{scoreBar(m.semantic_score, "#10a37f")}
                    </div>
                    <div className="memory-score-row">
                      <span>Recency</span>{scoreBar(m.recency_score, "#3b82f6")}
                    </div>
                    <div className="memory-score-row">
                      <span>Frequency</span>{scoreBar(m.frequency_score, "#f59e0b")}
                    </div>
                    <div className="memory-score-row">
                      <span>Importance</span>{scoreBar(m.importance_score, "#ec4899")}
                    </div>
                    <div className="memory-score-row final">
                      <span>Final</span>{scoreBar(m.final_score, "#0d0d0d")}
                    </div>
                  </div>
                  <div className="memory-meta">
                    Session: {m.session_id} · {new Date(m.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
              {memCount === 0 && (
                <p className="detail-empty">No past memories recalled for this query.</p>
              )}
            </div>
          )}

          {/* ── RAG Trace ───────────────────────────── */}
          {tab === "trace" && (
            <div className="details-content">
              {meta.trace_id && (
                <div className="detail-note">Trace ID: <code>{meta.trace_id}</code></div>
              )}
              <div className="trace-timeline">
                {[
                  ["Memory retrieval", meta.memory_time, "#ec4899"],
                  ["Document retrieval", meta.retrieval_time, "#3b82f6"],
                  ["LLM generation", meta.generation_time, "#10a37f"],
                ].map(([label, val, color]) => (
                  <div key={label as string} className="trace-row">
                    <div className="trace-label">{label as string}</div>
                    <div className="trace-bar-wrap">
                      <div
                        className="trace-bar"
                        style={{
                          width: meta.total_time
                            ? `${Math.min(100, ((val as number) / meta.total_time!) * 100)}%`
                            : "0%",
                          background: color as string,
                        }}
                      />
                    </div>
                    <div className="trace-time">{fmt(val as number)}</div>
                  </div>
                ))}
                <div className="trace-total">
                  Total: <strong>{fmt(meta.total_time)}</strong>
                </div>
              </div>
              {meta.arm && (
                <div className="detail-note" style={{ marginTop: 8 }}>
                  Arm: <strong>{meta.arm}</strong> — {ARM_DESCRIPTIONS[meta.arm] ?? meta.arm}
                </div>
              )}
            </div>
          )}

          {/* ── Evaluation ──────────────────────────── */}
          {tab === "evaluation" && (
            <div className="details-content">
              <div className="eval-grid">
                {([
                  { label: "Word overlap", value: meta.groundedness,
                    desc: "Lexical overlap with retrieved text. Matching words can include wrong numbers or negated facts." },
                  { label: "Claim support", value: meta.faithfulness,
                    desc: "Share of extracted claims classified as supported by the verifier against the supplied evidence." },
                  { label: "Answer Relevancy", value: meta.answer_relevancy,
                    desc: "Semantic cosine alignment between user question and generated answer." },
                  { label: "Context Precision", value: meta.context_precision,
                    desc: "Requires relevance judgments. See the Frameworks tab for judge-based context precision." },
                  { label: "Citation Precision", value: meta.citation_precision,
                    desc: "Share of the answer's chunk citations that exist in the retrieved evidence." },
                ]).map((card) => (
                  <div className="eval-card" key={card.label}>
                    <div className="eval-label">{card.label}</div>
                    {/* Missing data must read as missing, never as a plausible
                        number -- this card previously fell back to invented
                        constants ("92%", "88%", the top retrieval score),
                        which is metrics-dressing. "—" is the honest display. */}
                    <div
                      className="eval-value"
                      style={card.value == null ? { fontSize: 14 } : card.label !== "Word overlap" ? { color: card.value >= 0.8 ? "#10a37f" : "#f59e0b" } : undefined}
                    >
                      {card.value != null ? `${(card.value * 100).toFixed(0)}%` : "Unavailable"}
                    </div>
                    <div className="eval-desc">{card.desc}</div>
                  </div>
                ))}
                <div className="eval-card">
                  <div className="eval-label">Device</div>
                  <div className="eval-value" style={{ fontSize: 13, color: "#10a37f" }}>
                    {meta.device ?? "Unavailable"}
                  </div>
                  <div className="eval-desc">The machine this answer was computed on (reported by the backend, not assumed).</div>
                </div>
              </div>
              <div className="eval-note">
                ⚡ These scores use captured evidence and measurements from the original answer. Older runs may lack some measurements. For judge-based
                RAGAS / RAGChecker / ARES-style scores on this answer, open the
                <strong> 🏁 Frameworks</strong> tab.
              </div>
            </div>
          )}

          {/* ── Strategy (multi-step retrieval log) ───────── */}
          {tab === "strategy" && <StrategyTab meta={meta} />}

          {/* ── Frameworks (cross-framework metric panel) ─── */}
          {tab === "frameworks" && meta.trace_id && <FrameworksTab traceId={meta.trace_id} />}

          {/* ── Claims (decomposition + NLI verification) ──── */}
          {tab === "claims" && (
            <div className="details-content">
              {meta.claimError && (
                <p className="detail-empty">Claim verification failed: {meta.claimError}</p>
              )}
              {!meta.claimError && claimCount === 0 && (
                <p className="detail-empty">No claims could be decomposed from this answer.</p>
              )}
              {(meta.claims ?? []).map((c) => (
                <div key={c.claim_id} className="chunk-card">
                  <div className="chunk-header">
                    <span
                      className="dpill"
                      style={{ color: CLAIM_STATUS_COLOR[c.status], fontWeight: 600 }}
                    >
                      {CLAIM_STATUS_LABEL[c.status]}
                    </span>
                    {c.verified_by === "llm" && (
                      <span className="dpill" title="NLI was ambiguous on this claim; escalated to an LLM judge">
                        🧭 LLM judge
                      </span>
                    )}
                    <span className="chunk-score-label">confidence</span>
                    {scoreBar(c.confidence, CLAIM_STATUS_COLOR[c.status])}
                  </div>
                  <div className="chunk-text">{c.claim_text}</div>
                  {c.reason && <div className="detail-note">{c.reason}</div>}
                  {c.evidence_text && (
                    <div className="memory-card" style={{ marginTop: 8 }}>
                      <div className="memory-q">
                        Evidence{c.best_chunk_rank != null ? ` (chunk #${c.best_chunk_rank})` : ""}:
                      </div>
                      <div className="memory-a">{c.evidence_text}</div>
                    </div>
                  )}
                  <div className="memory-score-row">
                    <span>Entailment</span>{scoreBar(c.entailment_score, "#10a37f")}
                  </div>
                  <div className="memory-score-row">
                    <span>Contradiction</span>{scoreBar(c.contradiction_score, "#ef4444")}
                  </div>
                  <div className="memory-score-row">
                    <span>Neutral</span>{scoreBar(c.neutral_score, "#9ca3af")}
                  </div>
                  {c.best_chunk_id && (
                    <div className="chunk-id">
                      {c.best_chunk_id.startsWith("memory:")
                        ? "🧠 Grounded in a recalled memory (not a document)"
                        : `Source chunk: ${c.best_chunk_id}`}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── ThinkingBox (DeepSeek-style reasoning trace) ──────────────────────── */
function ThinkingBox({ meta }: { meta: MsgMeta }) {
  const [open, setOpen] = useState(true);
  const hasRealReasoning = meta.reasoning !== undefined;
  const pipelineLines = hasRealReasoning ? [] : buildPipelineTrace(meta);

  // Real model reasoning auto-collapses once the answer starts streaming
  // (matches DeepSeek: open while thinking, collapses to a summary line
  // once done); the synthesized pipeline trace has no "live" phase so it
  // just starts collapsed. Hooks must run unconditionally, so this sits
  // above the "nothing to show" early return below.
  useEffect(() => {
    if (hasRealReasoning && meta.reasoningDone) setOpen(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta.reasoningDone]);

  if (!hasRealReasoning && pipelineLines.length === 0) return null;

  const label = hasRealReasoning
    ? (meta.reasoningDone
        ? `Thought for ${meta.reasoningTimeMs != null ? Math.max(1, Math.round(meta.reasoningTimeMs / 1000)) : "a few"} seconds`
        : "Thinking…")
    : "Pipeline trace";

  return (
    <div className="thinking-box">
      <button className="thinking-box-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="details-toggle-icon">{open ? "▾" : "▸"}</span>
        {hasRealReasoning && !meta.reasoningDone && <span className="thinking-live-dot" />}
        {label}
      </button>
      {open && (
        <div className="thinking-box-body">
          {hasRealReasoning
            ? (meta.reasoning || "…")
            : pipelineLines.map((l, i) => <div key={i}>{l}</div>)}
        </div>
      )}
    </div>
  );
}

/* ─── ActionBar: copy / trace-id / re-run-with-arm ────────────────────────── */
function ActionBar({
  content, traceId, arms, currentArm, busy, onRunAgain,
}: {
  content: string;
  traceId?: string;
  arms: string[];
  currentArm: string;
  busy: boolean;
  onRunAgain: (arm: string) => void;
}) {
  const [copied, setCopied] = useState<"answer" | "trace" | null>(null);

  const copy = (text: string, tag: "answer" | "trace") => {
    void navigator.clipboard?.writeText(text).then(() => {
      setCopied(tag);
      window.setTimeout(() => setCopied((c) => (c === tag ? null : c)), 1500);
    });
  };

  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 4 }}>
      <button className="retry-btn" onClick={() => copy(content, "answer")}>
        {copied === "answer" ? "✓ Copied" : "📋 Copy answer"}
      </button>
      <AnswerTools key={traceId ?? "pending"} traceId={traceId} busy={busy} />
      {traceId && (
        <button
          className="retry-btn"
          title="Copy this turn's RAGTrace id (paste into scripts/generate_diagnostic_report.py)"
          onClick={() => copy(traceId, "trace")}
        >
          {copied === "trace" ? "✓ Copied" : "🪪 Trace ID"}
        </button>
      )}
      {arms.length > 0 && (
        <select
          className="topbar-select"
          value=""
          disabled={busy}
          title="Ask the same question again under a different retrieval arm — the answers land side by side, so you can compare what each strategy did"
          onChange={(e) => {
            const chosen = e.target.value;
            e.target.value = "";
            if (chosen) onRunAgain(chosen);
          }}
        >
          <option value="" disabled>⟲ Re-run with arm…</option>
          {arms.filter((a) => a !== currentArm).map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>
      )}
    </div>
  );
}

/* ─── Chat ───────────────────────────────────────────────── */
function SessionReportAction({ sessionId, disabled }: { sessionId: string; disabled: boolean }) {
  const [saving, setSaving] = useState(false);
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");

  async function save() {
    setSaving(true);
    setError("");
    try {
      setUrl((await saveSessionReport(sessionId)).url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the report.");
    } finally {
      setSaving(false);
    }
  }

  return <>
    <button className="btn btn-secondary" disabled={disabled || saving} onClick={() => void save()}>
      {saving ? "Analyzing turns and saving report…" : url ? "Update session report" : "Save session report"}
    </button>
    {url && <a className="btn btn-secondary" href={`${API_BASE}${url}`} target="_blank" rel="noreferrer">Open saved report</a>}
    {error && <span className="text-muted" role="alert">{error}</span>}
  </>;
}

export default function Chat({ activeSessionId, onSessionCreated, onSessionRenamed, settings, onTurnDone }: ChatProps) {
  const [config, setConfig] = useState<Config | null>(null);
  const [configError, setConfigError] = useState(false);
  const [arm, setArm] = useState("C_hybrid_rerank");
  const [corpus, setCorpus] = useState("statutes");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [stageLabel, setStageLabel] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const busyRef = useRef(false);
  const requestSessionRef = useRef<string | null>(null);
  const reasoningStartRef = useRef<number | null>(null);

  async function loadConfig() {
    try {
      const c = await getConfig();
      setConfig(c);
      setConfigError(false);
      if (c.arms.length) setArm((prev) => (c.arms.includes(prev) ? prev : c.arms[0]));
    } catch {
      setConfigError(true);
    }
  }

  useEffect(() => {
    void Promise.resolve().then(loadConfig);
  }, []);

  // "both" has no graph/IRCoT analogue -- it merges two independently
  // hybrid-reranked corpora, so only the baseline arm applies.
  useEffect(() => {
    const allowed = config?.arms_by_corpus[corpus];
    if (allowed?.length) setArm(current => allowed.includes(current) ? current : allowed[0]);
  }, [corpus, config]);

  useEffect(() => () => { abortRef.current?.abort(); }, []);

  useEffect(() => {
    if (busyRef.current && requestSessionRef.current === activeSessionId) return;
    abortRef.current?.abort();
    abortRef.current = null;
    busyRef.current = false;
    setBusy(false);
    setStageLabel("");
    if (!activeSessionId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    getSessionMessages(activeSessionId)
      .then((msgs) => {
        if (cancelled) return;
        setMessages(msgs.map((m) => ({ role: m.role, content: m.content })));
        // Re-fetching a session (page reload, or navigating to another view
        // and back to Chat -- Chat unmounts, so this whole effect re-runs)
        // gets message text and trace IDs. Hydrate each
        // historical assistant message's Details panel from its saved
        // RAGTrace instead of leaving it permanently blank.
        msgs.forEach((m, i) => {
          if (m.role !== "assistant" || !m.trace_id) return;
          getTrace(m.trace_id).then((trace) => {
            if (cancelled || !trace) return;
            const stats = trace.execution_statistics || {};
            setMessages((cur) => {
              if (cur.length !== msgs.length) return cur; // stale, session changed since
              const next = [...cur];
              next[i] = {
                ...next[i],
                meta: {
                  ...next[i].meta,
                  ...trace.evaluation,
                  trace_id: trace.trace_id,
                  chunks: (trace.retrieved_chunk_references || []).map((c) => ({
                    chunk_id: c.chunk_id,
                    source_file: c.source_file,
                    page_number: c.page_number,
                    similarity_score: c.similarity_score,
                    text: c.text || "",
                  })),
                  retrieval_time: stats.retrieval_time as number,
                  generation_time: stats.generation_time as number,
                  total_time: stats.total_pipeline_time as number,
                  ...(trace.claim_verification && {
                    claimCount: trace.claim_verification.claim_count,
                    claims: trace.claim_verification.results,
                  }),
                  ...(trace.claim_error != null && { claimError: trace.claim_error }),
                },
              };
              return next;
            });
          });
        });
      })
      .catch(() => { if (!cancelled) setMessages([]); });
    return () => { cancelled = true; };
  }, [activeSessionId]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, stageLabel]);

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
  }

  // Helper: patch the last assistant message's meta field
  function patchLastMeta(patch: Partial<MsgMeta> | ((prev: MsgMeta) => Partial<MsgMeta>)) {
    setMessages((m) => {
      if (!m.length) return m;
      const next = [...m];
      const last = { ...next[next.length - 1] };
      const resolved = typeof patch === "function" ? patch(last.meta || {}) : patch;
      last.meta = { ...last.meta, ...resolved };
      next[next.length - 1] = last;
      return next;
    });
  }

  async function send(question?: string, armOverride?: string, baseMessages: Message[] = messages) {
    const q = (question ?? input).trim();
    if (!q || busyRef.current) return;
    busyRef.current = true;
    let sessId = activeSessionId;

    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setBusy(true);
    setStageLabel("Starting…");

    // An arm-override re-run asks the SAME question fresh (no chat history) --
    // otherwise the condenser would resolve it against the previous turn and
    // the two answers would no longer differ only in retrieval arm.
    const armUsed = armOverride ?? arm;
    const history = armOverride ? [] : baseMessages.map((m) => ({ role: m.role, content: m.content }));
    const isFirstQuestion = baseMessages.length === 0;
    const initialMeta: MsgMeta = { arm: armUsed, corpus };

    setMessages([
      ...baseMessages,
      { role: "user", content: q },
      { role: "assistant", content: "", meta: initialMeta },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;
    requestSessionRef.current = sessId;
    reasoningStartRef.current = null;

    try {
      if (!sessId) {
        const newSess = await createSession(q.slice(0, 30));
        if (controller.signal.aborted) return;
        sessId = newSess.session_id;
        requestSessionRef.current = sessId;
        onSessionCreated(sessId);
      }
      for await (const evt of streamChat({
        question: q,
        session_id: sessId,
        arm: armUsed,
        corpus,
        chat_history: history,
        memory_enabled: settings.memoryEnabled,
        deep_analysis: settings.deepAnalysis,
      }, controller.signal)) {
        if (abortRef.current !== controller || controller.signal.aborted) return;
        applyEvent(evt, q);
      }

      if (isFirstQuestion && sessId) {
        const title = q.split(" ").slice(0, 5).join(" ") + (q.split(" ").length > 5 ? "..." : "");
        await renameSession(sessId, title);
        onSessionRenamed();
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError" && abortRef.current === controller) {
        applyEvent({ event: 'error', message: (err as Error).message || 'The chat request failed. Please retry.' }, q);
      }
      // Stopped by the user -- keep whatever text/chunks already streamed in.
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        busyRef.current = false;
        setBusy(false);
        setStageLabel("");
      }
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function applyEvent(evt: ChatEvent, question: string) {
    if (evt.event === "stage") {
      setStageLabel(evt.label as string);
    } else if (evt.event === "memory") {
      patchLastMeta({
        recalled_memories: evt.recalled as MemoryRecall[],
        memory_time: evt.memory_time as number,
      });
    } else if (evt.event === "meta") {
      patchLastMeta({
        search_query: evt.search_query as string,
        query_was_condensed: evt.query_was_condensed as boolean,
        arm: evt.arm as string,
      });
    } else if (evt.event === "chunks") {
      patchLastMeta({
        chunks: evt.chunks as Chunk[],
        retrieval_time: evt.retrieval_time as number,
        retrieval_metadata: evt.retrieval_metadata as Record<string, unknown>,
      });
    } else if (evt.event === "reasoning") {
      if (reasoningStartRef.current === null) reasoningStartRef.current = performance.now();
      const text = evt.text as string;
      patchLastMeta((prev) => ({ reasoning: (prev.reasoning || "") + text }));
    } else if (evt.event === "token") {
      // First answer token after any reasoning marks the thinking phase done.
      if (reasoningStartRef.current !== null) {
        const elapsed = performance.now() - reasoningStartRef.current;
        reasoningStartRef.current = null;
        patchLastMeta({ reasoningDone: true, reasoningTimeMs: elapsed });
      }
      const text = evt.text as string;
      setMessages((m) => {
        const next = [...m];
        const last = { ...next[next.length - 1] };
        last.content = last.content + text;
        next[next.length - 1] = last;
        return next;
      });
    } else if (evt.event === "strategy") {
      const verification = evt.verification as { claim_count: number; results: ClaimVerification[] } | undefined;
      patchLastMeta({
        trace_id: evt.trace_id as string,
        ...(verification && { claimCount: verification.claim_count, claims: verification.results }),
        ...(evt.claim_error != null && { claimError: evt.claim_error as string }),
      });
    } else if (evt.event === "evaluation") {
      patchLastMeta({
        groundedness: evt.groundedness as number,
        faithfulness: evt.faithfulness as number,
        answer_relevancy: evt.answer_relevancy as number,
        context_precision: evt.context_precision as number,
        context_relevancy: evt.context_relevancy as number,
        citation_precision: evt.citation_precision as number,
        citations_found: evt.citations_found as number,
        device: evt.device as string,
      });
    } else if (evt.event === "done") {
      patchLastMeta({
        total_time: evt.total_time as number,
        generation_time: evt.generation_time as number,
        retrieval_time: evt.retrieval_time as number,
        memory_time: evt.memory_time as number,
        trace_id: evt.trace_id as string,
      });
      setStageLabel("");
      if (evt.trace_id) onTurnDone(evt.trace_id as string);
    } else if (evt.event === "error") {
      setMessages((m) => {
        const next = [...m];
        next[next.length - 1] = {
          ...next[next.length - 1],
          content: `⚠️ Error: ${evt.message}`,
          failedQuestion: question,
        };
        return next;
      });
    }
  }

  function retry(question: string) {
    void send(question, undefined, messages.slice(0, -2));
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-page">
      {/* ── Top bar ─────────────────────────────── */}
      <div className="topbar">
        <span className="topbar-title">Chat</span>
        <div className="topbar-controls">
          {activeSessionId && <SessionReportAction key={activeSessionId} sessionId={activeSessionId} disabled={busy || isEmpty} />}
          <label className="chat-setting">
            Retrieval strategy
          <select
            aria-label="Retrieval strategy"
            className="topbar-select"
            value={arm}
            onChange={(e) => setArm(e.target.value)}
            disabled={!config || busy || corpus === "both"}
            title={
              corpus === "both"
                ? "Combined mode always uses hybrid + rerank -- graph/IRCoT arms are built on the judgments-only citation graph"
                : `Retrieval strategy: ${ARM_DESCRIPTIONS[arm] ?? arm}`
            }
          >
            {!config && <option value={arm}>{configError ? "Unavailable" : "Loading strategies…"}</option>}
            {config?.arms_by_corpus[corpus]?.map((a) => (
              <option key={a} value={a} title={a}>{ARM_DESCRIPTIONS[a] ?? a}</option>
            ))}
          </select>
          </label>
          <label className="chat-setting">
            Document collection
          <select
            aria-label="Document collection"
            className="topbar-select"
            value={corpus}
            disabled={!config || busy}
            onChange={(e) => setCorpus(e.target.value)}
            title="Which ingested document corpus to answer from"
          >
            {!config && <option value={corpus}>{configError ? "Unavailable" : "Loading collections…"}</option>}
            {config?.corpora.map((c) => (
              <option key={c} value={c}>{CORPUS_LABELS[c] ?? c}</option>
            ))}
          </select>
          </label>
        </div>
      </div>
      {configError && <div className="connection-notice" role="alert"><span>Cannot load retrieval settings. Check that the API is running.</span><button className="btn" onClick={() => void loadConfig()}>Retry settings</button></div>}

      {/* ── Welcome / Messages ──────────────────── */}
      {isEmpty ? (
        <div className="chat-welcome">
          <h1>What would you like to research?</h1>
          <p className="chat-welcome-note">Choose a collection above, then write a question or edit an example below.</p>
          <div className="suggestions">
            {SUGGESTIONS.map((s) => (
              <button key={s} className="suggestion-chip" onClick={() => { setInput(s); textareaRef.current?.focus(); }}>{s}</button>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-messages" ref={listRef}>
          {messages.map((m, i) => (
            <div key={i} className={`msg-row ${m.role}`}>
              <div className={`msg-content-col ${m.role === "assistant" ? "blueprint" : ""}`}>
                {m.role === "assistant" && (
                  <>
                    <i className="corner tl" /><i className="corner tr" />
                    <i className="corner bl" /><i className="corner br" />
                    <div className="msg-kicker-row">
                      <span className="card-kicker">Answer</span>
                      {m.meta?.arm && <span className="tag tag-neutral">{m.meta.arm}</span>}
                    </div>
                    {m.meta && <ThinkingBox meta={m.meta} />}
                  </>
                )}
                <div className="msg-bubble">
                  {m.content ? (
                    m.content
                  ) : busy && i === messages.length - 1 ? (
                    <div>
                      <div className="stage-label">{stageLabel}</div>
                      <div className="msg-thinking"><span /><span /><span /></div>
                    </div>
                  ) : null}
                </div>
                {m.role === "assistant" && m.failedQuestion && (
                  <button
                    className="retry-btn"
                    onClick={() => retry(m.failedQuestion!)}
                  >🔄 Retry</button>
                )}
                {m.role === "assistant" && m.content && !m.failedQuestion && (
                  <ActionBar
                    content={m.content}
                    traceId={m.meta?.trace_id}
                    arms={config?.arms_by_corpus[corpus] ?? []}
                    currentArm={m.meta?.arm ?? arm}
                    busy={busy}
                    onRunAgain={(a) => {
                      const prevUser = [...messages.slice(0, i)].reverse().find((x) => x.role === "user");
                      if (prevUser && !busy) void send(prevUser.content, a);
                    }}
                  />
                )}
                {/* Details panel — as soon as there's retrieval/memory data to
                    show, not just once the answer has finished streaming. */}
                {m.role === "assistant" && m.meta &&
                  (m.content || m.meta.chunks?.length || m.meta.recalled_memories?.length) && (
                  <DetailsPanel meta={m.meta} defaultOpen={settings.autoExpandDetails} />
                )}
              </div>
            </div>
          ))}
          {/* Stage label while last turn is still streaming */}
          {busy && stageLabel && messages.length > 0 &&
            messages[messages.length - 1].content !== "" && (
            <div className="stage-indicator">
              <span className="stage-dot" />
              {stageLabel}
            </div>
          )}
        </div>
      )}

      {/* ── Input bar ───────────────────────────── */}
      <div className="input-area">
        <div>
          <div className="input-box">
            <textarea
              ref={textareaRef}
              className="input-textarea"
              aria-label="Question"
              aria-describedby="question-hint"
              rows={1}
              value={input}
              onChange={handleInput}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (config) void send(); }
              }}
              placeholder={corpus === "judgments" ? "Ask about Supreme Court judgments…" : corpus === "both" ? "Ask about statutes and judgments…" : "Ask about the statutes…"}
              disabled={busy}
            />
            {busy ? (
              <button className="send-btn stop-btn" aria-label="Stop generating" onClick={stop} title="Stop generating">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              </button>
            ) : (
              <button className="send-btn" aria-label="Send message" onClick={() => send()} disabled={!input.trim() || !config}>
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 4l8 8h-5v8H9v-8H4z" />
                </svg>
              </button>
            )}
          </div>
          <div id="question-hint" className="input-hint">Enter to send · Shift + Enter for a new line. Verify important legal information.</div>
        </div>
      </div>
    </div>
  );
}
