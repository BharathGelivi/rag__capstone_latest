export const API_BASE = "http://127.0.0.1:8010";

export interface Config {
  arms: string[];
  corpora: string[];
  device: string;
  arms_by_corpus: Record<string, string[]>;
}

export async function getConfig(): Promise<Config> {
  const r = await checkedFetch(`${API_BASE}/ui/config`);
  return r.json();
}

export interface ExperimentStatus {
  number: number;
  key: string;
  title: string;
  claim: string;
  status: string;
  n_recorded?: number;
  n_planned?: number;
  mode?: string;
  summary?: Record<string, unknown>;
}

export interface ExperimentRuns {
  run: Record<string, unknown> | null;
  experiments: ExperimentStatus[];
  monitor?: {status: string; elapsed_seconds?: number; next_check_hours?: number};
}

export async function getExperimentRuns(source = "active"): Promise<ExperimentRuns> {
  const r = await fetch(`${API_BASE}/ui/experiments?source=${source}`);
  if (!r.ok) throw new Error(`Experiment status failed (HTTP ${r.status})`);
  return r.json();
}

export interface ChatEvent {
  event: "stage" | "memory" | "meta" | "chunks" | "reasoning" | "token" | "strategy" | "evaluation" | "done" | "error";
  [key: string]: unknown;
}

export interface ChatRequest {
  question: string;
  session_id: string;
  arm: string;
  corpus: string;
  chat_history: { role: string; content: string }[];
  memory_enabled: boolean;
  deep_analysis: boolean;
}

/**
 * Stream a chat turn's SSE events. Reads split across chunk boundaries are
 * common with fetch + ReadableStream, so buffer until the last "\n\n" frame
 * separator rather than assuming one read == one event.
 */
export async function* streamChat(req: ChatRequest, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const res = await checkedFetch(`${API_BASE}/ui/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) throw new Error(`Chat request failed (HTTP ${res.status}). Please retry.`);
  if (!res.body) throw new Error('The chat response was empty. Please retry.');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
      let sep;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const data = frame.split("\n").filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
        if (data) {
          const event = JSON.parse(data) as ChatEvent;
          terminal ||= event.event === "done" || event.event === "error";
          yield event;
          if (terminal) return;
        }
      }
    }
    if (!terminal) throw new Error("The chat connection ended before completion. Please retry.");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export interface GraphData {
  nodes: Array<{
    id: string;
    type: string;
    label: string;
    degree: number;
    detail?: string;
    meta?: string;
  }>;
  edges: Array<{ source: string; target: string; kind: string }>;
  truncated: boolean;
}

export async function getGraph(scope: "rag" | "legal" | "memory", signal?: AbortSignal): Promise<GraphData> {
  const r = await checkedFetch(`${API_BASE}/ui/graph?scope=${scope}`, { signal });
  return r.json();
}

export interface Session {
  session_id: string;
  title: string;
  created_at: string;
  last_activity: string;
  interaction_count: number;
}

export async function getSessions(): Promise<Session[]> {
  const r = await checkedFetch(`${API_BASE}/ui/sessions`);
  return r.json();
}

export async function createSession(title: string = "New Session"): Promise<Session> {
  const r = await checkedFetch(`${API_BASE}/ui/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return r.json();
}

export async function deleteSession(id: string): Promise<void> {
  await checkedFetch(`${API_BASE}/ui/sessions/${id}`, { method: "DELETE" });
}

export async function renameSession(id: string, title: string): Promise<void> {
  await checkedFetch(`${API_BASE}/ui/sessions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export async function getSessionMessages(
  id: string, signal?: AbortSignal
): Promise<{ role: "user" | "assistant"; content: string; trace_id?: string | null }[]> {
  const r = await checkedFetch(`${API_BASE}/ui/sessions/${encodeURIComponent(id)}/messages`, { signal });
  return r.json();
}

export interface MemoryEntry {
  memory_id: string;
  session_id: string;
  question: string;
  answer: string;
  timestamp: string;
  importance_score: number;
  access_count: number;
  trace_id: string | null;
  retrieved_chunk_ids: string[];
  claim_ids: string[];
  tags: string[];
  kind?: string;                 // "turn" | "fact" | "summary"
  scope?: string;                // "chat" | "eval"
  superseded_by?: string | null; // id of the entry that replaced this one
}

export interface ScoredMemory {
  memory: MemoryEntry;
  semantic_score: number;
  recency_score: number;
  frequency_score: number;
  importance_score: number;
  final_score: number;
  retrieval_reason: string;
}

export async function getMemory(
  opts: { search?: string; sessionId?: string; limit?: number } = {}, signal?: AbortSignal
): Promise<{ memories: ScoredMemory[]; searched: boolean }> {
  const params = new URLSearchParams();
  if (opts.search) params.set("search", opts.search);
  if (opts.sessionId) params.set("session_id", opts.sessionId);
  if (opts.limit) params.set("limit", String(opts.limit));
  const r = await checkedFetch(`${API_BASE}/ui/memory?${params}`, { signal });
  return r.json();
}

export interface MemoryHealth {
  status: "healthy" | "degraded";
  memory_count: number;
  session_count: number;
  summary_count: number;
  missing_session_count: number;
  malformed_entry_count: number;
  embedding_model?: string | null;
  collection: string;
}

export async function getMemoryHealth(): Promise<MemoryHealth> {
  const r = await fetch(`${API_BASE}/ui/memory/health`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export interface TraceChunkReference {
  chunk_id: string;
  rank: number;
  similarity_score: number;
  page_number: string | number;
  source_file: string;
  text?: string;
}

export interface ClaimVerification {
  claim_id: string;
  claim_text: string;
  status: "SUPPORTED" | "PARTIALLY_SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED" | "NOT_VERIFIABLE";
  reason: string;
  confidence: number;
  entailment_score: number;
  contradiction_score: number;
  neutral_score: number;
  best_chunk_id: string | null;
  best_chunk_rank: number | null;
  evidence_text: string | null;
  verified_by: "nli" | "llm";
  character_start?: number;
  character_end?: number;
  citation_checks?: Array<{chunk_id: string; status: string; reason: string; entailment_score?: number}>;
}

export interface CaseExpectations {
  expected_evidence: string[]; required_facts: string[]; forbidden_claims: string[];
  refusal: "required" | "allowed" | "forbidden"; fact_match: "text" | "nli";
}
export interface CaseResult {
  case_id: string; name: string; passed: boolean;
  change?: "fixed" | "regressed" | "unchanged" | "unavailable";
  answer?: string; evidence_ids?: string[];
  checks: Array<{kind: string; assertion?: string; status: string; detail?: string}>;
}
export interface SavedCase {case_id: string; name: string; source_trace_id: string; expectations: CaseExpectations}
export interface EvidenceDebugData {
  claims: ClaimVerification[];
  sentences: Array<{sentence_id: string; text: string; citations: string[]; claims: ClaimVerification[]}>;
  unmapped_claim_ids: string[];
  sources: TraceChunkReference[];
  excluded_candidates: Array<TraceChunkReference & {chunk_text?: string; candidate_rank?: number; reranker_score?: number; observations?: Array<{retrieval_query?: string; candidate_rank?: number; reranker_score?: number}>}>;
  candidate_capture: "captured" | "unavailable"; candidate_scope: string;
  signals: Array<{kind: string; message: string}>; note: string;
}
export function saveRegressionCase(traceId: string, name: string, expectations: CaseExpectations): Promise<{case_id: string; path: string; baseline: CaseResult}> {
  return replayResponse(`/ui/trace/${encodeURIComponent(traceId)}/regression-cases`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({name, expectations}),
  });
}
export function getRegressionCases(signal?: AbortSignal): Promise<SavedCase[]> {
  return replayResponse("/ui/regression-cases", {signal});
}
export function runRegressionCases(caseIds: string[], mode: string): Promise<{mode: string; passed: boolean; cases: CaseResult[]}> {
  return replayResponse("/ui/regression-cases/run", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({mode, case_ids: caseIds})});
}
export function getEvidenceDebug(traceId: string, signal?: AbortSignal): Promise<EvidenceDebugData> {
  return replayResponse(`/ui/trace/${encodeURIComponent(traceId)}/evidence`, {signal});
}
export function analyzeEvidence(traceId: string): Promise<EvidenceDebugData> {
  return replayResponse(`/ui/trace/${encodeURIComponent(traceId)}/evidence/analyze`, {method: "POST"});
}

export interface TraceData {
  trace_id: string;
  question: string;
  generated_answer: string;
  timestamp: string;
  prompt_snapshot: string;
  execution_statistics: Record<string, unknown>;
  configuration_snapshot: Record<string, unknown>;
  retrieved_chunk_references: TraceChunkReference[];
  // Only present when the turn ran with deep_analysis=true -- see
  // chat_service.py's post-hoc patch of the saved trace file.
  claim_verification?: { claim_count: number; results: ClaimVerification[] };
  claim_error?: string;
}

export interface ReplaySettings {
  system_prompt: string; model: string; arm: string; corpus: string; top_k: number;
}
export interface ReplayResult {
  original: TraceData;
  replay: TraceData;
  warnings: string[];
  comparison: {
    evidence_added: string[]; evidence_removed: string[];
    evidence_order_changed: boolean; answer_changed: boolean;
    claim_changes?: Array<{claim_text: string; original_status: string | null; replay_status: string | null}> | null;
    scores: Record<"original" | "replay", {claim_count: number; supported_fraction: number | null; statuses: Record<string, number>}>;
  };
}

async function replayResponse<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Replay request failed (HTTP ${response.status})`);
  }
  return response.json();
}
export function getSavedTraces(signal?: AbortSignal): Promise<Array<{trace_id: string; question: string; timestamp: string}>> {
  return replayResponse("/ui/traces", {signal});
}
export function getReplaySettings(traceId: string, signal?: AbortSignal): Promise<ReplaySettings> {
  return replayResponse(`/ui/trace/${encodeURIComponent(traceId)}/replay`, {signal});
}
export function replayTrace(traceId: string, options: {
  mode: "frozen" | "retrieve"; model?: string; system_prompt?: string;
  arm?: string; corpus?: string; top_k?: number; verify_claims: boolean;
}, signal?: AbortSignal): Promise<ReplayResult> {
  return replayResponse(`/ui/trace/${encodeURIComponent(traceId)}/replay`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(options), signal,
  });
}

export async function getLatestTrace(signal?: AbortSignal): Promise<TraceData | null> {
  const r = await fetch(`${API_BASE}/ui/trace/latest`, { signal });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`Unable to load trace (HTTP ${r.status})`);
  return r.json();
}

export async function getTrace(traceId: string, signal?: AbortSignal): Promise<TraceData | null> {
  const r = await fetch(`${API_BASE}/ui/trace/${encodeURIComponent(traceId)}`, { signal });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`Unable to load trace (HTTP ${r.status})`);
  return r.json();
}

/* ─── Per-trace framework comparison (RAGAS / RAGChecker / ARES style) ─── */

export interface FrameworkMetric {
  key: string;
  label: string;
  value: number | null;   // null = not computed (typically: needs a reference answer)
  definition: string;
  judge: string;
  requires?: string | null;
}

export interface FrameworkFamily {
  family: string;         // "RAGAS" | "RAGChecker" | "ARES" | "this project"
  metrics: FrameworkMetric[];
}

export interface FrameworkPanel {
  trace_id: string;
  note: string;           // honesty label: these are X-RAG judges, *style* metrics
  families: FrameworkFamily[];
  verification: {
    total: number; supported: number; partially_supported: number;
    contradicted: number; unsupported: number; not_verifiable: number;
    average_entailment: number;
  } | null;
  computed_claims: boolean;  // deep analysis was run on demand for this call
  cached: boolean;
}

export async function getFrameworkPanel(traceId: string): Promise<FrameworkPanel> {
  const r = await fetch(`${API_BASE}/ui/trace/${traceId}/metrics`);
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* keep status */ }
    throw new Error(msg);
  }
  return r.json();
}

export interface DiagnosticReport {
  name: string;
  trace_id: string;
  session_id?: string | null;
  size_bytes: number;
  modified_at: number;
  url: string;
}

export function saveSessionReport(sessionId: string): Promise<{ url: string }> {
  return replayResponse(`/ui/sessions/${encodeURIComponent(sessionId)}/report`, { method: "POST" });
}

export interface TuningSuggestion {
  title: string;
  experiment: string;
  measure: string;
  source: "diagnostic" | "llm" | "trace_rule" | "fallback";
}

export interface DebugSuggestions {
  current: TuningSuggestion | null;
  trace_count: number;
  ranked: Array<{suggestion: TuningSuggestion; trace_count: number; evidence_count: number}>;
}

export function getDebugSuggestions(traceId: string | undefined, signal: AbortSignal): Promise<DebugSuggestions> {
  return replayResponse("/ui/debug/suggestions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({trace_id: traceId ?? null}),
    signal,
  });
}

export async function getDiagnosticReports(): Promise<DiagnosticReport[]> {
  const r = await fetch(`${API_BASE}/ui/reports`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()).reports;
}


async function checkedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const response = await fetch(input, init);
  if (!response.ok) throw new Error(`Request failed (HTTP ${response.status}). Please retry.`);
  return response;
}

export async function toolsRequest<T>(path: string, body?: unknown, method?: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {method: method ?? (body === undefined ? "GET" : "POST"), headers: {"Content-Type": "application/json"}, ...(body === undefined ? {} : {body: JSON.stringify(body)})});
  if (!response.ok) {const error = await response.json().catch(() => ({})); throw new Error(typeof error.detail === "string" ? error.detail : `Request failed (${response.status})`);}
  return response.json();
}
