# X-RAG audit — 2026-10-02

## 1. Executive Summary

Stopped the older repository copy's Vite (5173), UI API (8010), and diagnostic
API (8001) processes before indexing. Unrelated localhost applications were left
running. Built a lightweight semantic index, established practical baselines,
and applied targeted correctness and error-handling fixes. Existing uncommitted
research, memory, UI, and model changes were preserved. This is a targeted
maintenance pass, not certification of all research claims or deployment security.

## 2. Bugs Found

- Report HTML embedded unescaped answer/evidence text and trace IDs.
- Ablation `max_pool_top3` took independent score maxima while production retained
  the highest-entailment sentence's complete probability distribution.
- The offline integration test selected all 13 studies, including real retrieval
  and remote LLM work, despite asserting exactly five completed experiments.
- Chat transport silently accepted HTTP errors/streams without terminal events;
  the UI rethrew non-abort failures instead of displaying its retry state.
- Unexpected backend chat errors exposed exception types and raw exception text.

## 3. Bugs Fixed

### HTML rendering

Problem/root cause: trace content was interpolated into a `<pre>` and document
title without escaping. Fix: stdlib `html.escape` at the rendering boundary.
Validation: a malicious-markup regression failed before the fix, passes afterward;
existing HTML, Markdown and PDF tests pass. Impact: evidence is preserved as text
without turning model output into browser markup; diagnostic scores are unchanged.

### Ablation scoring

Problem/root cause: duplicated aggregation logic returned invalid distributions
(e.g. 0.3 entailment + 0.9 contradiction + 0.6 neutral). Fix: retain the full
best-sentence triplet, consistent with production; correct the threshold comment.
Validation: conflicting-sentence regression failed before, passes afterward.
Impact: **ablation metrics and diagnoses can change**. Regenerate ablation outputs;
do not compare old and new max-pool results as equivalent methods. Production
verification semantics, evidence order, thresholds and trace fields are unchanged.

### Offline test scope

Problem/root cause: an expanding study catalog unintentionally changed an offline
test into a live benchmark. Fix: select E1–E5 consistently for execution, assertions
and resumption. Validation: full discovery completes; E6–E13 remain available to
the real experiment driver. Impact: avoids unrequested corpus/model/API work in
this integration test without changing experiment or diagnostic implementations.

## 4. Performance Bottlenecks

- Heavy dependency imports dominate API startup. An import trace reached
  Transformers through RAGTrace → Retriever → HuggingFace embeddings.
- Local NLI initialization measured 3777.728 ms; one-pair inference median was
  312.713 ms (three repetitions, including warm-up effects).
- Verification batches sentence pairs within each claim, but `verify_all` still
  visits claims sequentially. Cross-claim batching remains unmeasured.
- Existing runner/verifier/embedding caches already reuse models. `lru_cache`
  permits concurrent cold misses; application lifecycle/concurrency needs a
  separate measured study before changing model ownership.
- Remote generation, planning and judge calls were not timed in this pass.

## 5. Performance Changes

No inference/cache/concurrency optimization is claimed. Measurements do not
justify one yet. Existing rate limits, model selection and GPU locks were retained.
The semantic index reduces navigation work without ML imports or new dependencies.
The offline test fix removes accidental live work; no full before/after suite
speedup is claimed because the original live run was stopped.

## 6. UI/UX Issues

HTTP errors and truncated SSE streams could leave a blank or partial answer and
an unhandled rejected promise. Full browser layout/accessibility and live answer
quality were not verified. Session creation before the busy flag remains a
potential duplicate-send race requiring a dedicated interaction regression.

## 7. UI/UX Fixes

Problem/root cause: transport did not distinguish clean completion from connection
closure. Fix: require a terminal `done` or `error`, check HTTP status/body, release
the stream reader, and route failures into the existing retry UI.
Validation: Node's built-in test covers HTTP 500, truncated streams, UTF-8 split
across individual bytes, terminal completion and early-consumer cancellation;
TypeScript/Vite build passes. Impact: users see retryable failures; no extra
diagnostic requests or polling were added. Cancellation remains user controlled.

## 8. API Issues

Problem/root cause: the outer chat exception handler sent raw exceptions to SSE.
Fix: log the full exception with a UUID and send a generic message/reference ID.
Validation: unit regression confirms secrets stay in developer logs and the
terminal error event contains a reference. Expected missing-corpus hints remain.
Other internal error paths and trace-ID/path validation need a separate boundary
audit before external deployment. Artifact discovery and presenter appendix use
flat names that differ from several save APIs; indexed/documented, not migrated.

## 9. Repository Cleanup

Added precise `.gitignore` exceptions for the index, its guide and this audit;
other ignored documentation retains its existing treatment. Benchmarks/test logs
stay under ignored `scratch/`; no research artifacts, databases or preexisting
changes were deleted. Added no dependencies. No commit or server was left running.

## 10. Semantic Index Design

`scripts/build_semantic_index.py` builds deterministic `docs/semantic_index.json`
using stdlib AST, source hashes and frontend import/export extraction. Includes
paths, purposes/docstrings, symbols and lines, public APIs, imports, config
dependencies, artifact references, curated core I/O, direct/reverse Python
dependencies, test associations and endpoints. `docs/SEMANTIC_INDEX.md` maps the
diagnostic chain and actual artifact naming conventions. Static test references
are not coverage; dynamic imports, computed paths and frontend dependency
resolution remain explicit limitations. Rebuild after source changes.

## 11. Before/After Measurements

Offline timing smoke checks use 20 repetitions and report medians. API import is
one process sample per run; health uses TestClient. Synthetic diagnosis uses
preverified evidence and excludes retrieval, decomposition, NLI and remote LLMs.

| Measurement (ms) | Before | After |
|---|---:|---:|
| API import startup | 11574.855 | 6887.770 |
| Health request median | 0.559 | 0.359 |
| Synthetic state + cause median | 0.012 | 0.009 |
| Report generation median | 0.018 | 0.015 |
| HTML rendering median | 0.011 | 0.012 |

These differences reflect run/cache/environment variation, **not demonstrated
performance gains**. Escaping adds a small expected rendering cost for correctness.
Local model timing is an after-only observation, not a comparison. Live end-to-end
latency is unmeasured. Separate real socket smoke checks observed API readiness at
7156.659 ms and UI backend readiness at 7254.904 ms; each returned its health/config
response and was terminated. First socket-request readings include readiness/
connection effects and are not representative steady-state request latency.

## 12. Remaining Issues

- Browser tests skipped because the old servers were intentionally stopped;
  a prior full run had only the expected browser connection error.
- Frontend lint reports existing effect/dependency/expression warnings and purity
  warnings at event timing calls; lint exits successfully. No warning suppression.
- Python emits deprecated `utcnow` and Chroma embedding-function warnings.
- A complete live RAG/LLM diagnosis, UI visual review, memory profiling and optional
  ARES/RAGChecker environment tests were not run. API tests, synthetic end-to-end
  diagnostics, report serialization/rendering and module self-checks ran through
  unittest discovery.
- Existing README performance/model claims predate this pass and are not newly
  validated; model configuration already differs from some older README prose.

## 13. Recommended Future Work

Regenerate aggregation ablations with the corrected scorer. Run browser/live
diagnostic checks against this copy with recorded corpus/model/config versions.
Then measure cold-miss concurrency, per-claim sentence splitting and cross-claim
batching before modifying model lifecycle or caching. Address artifact naming and
filesystem input validation before external deployment; preserve compatibility
and intentionally version any research-result change.

## Files changed by this pass and validation

- Index: `scripts/build_semantic_index.py`, `docs/semantic_index.json`,
  `docs/SEMANTIC_INDEX.md`, `tests/test_semantic_index.py`, `.gitignore`.
- Reports/verifier: `src/report_presenter.py`, `src/claim_verifier.py`,
  `configs/thresholds.py`, `tests/test_report_presenter.py`, `tests/test_claim_verifier.py`.
- Chat: `frontend/src/lib/api.ts`, `frontend/src/components/Chat.tsx`,
  `frontend/tests/api.test.mjs`, `src/chat_service.py`, `tests/test_api_ui.py`.
- Developer workflow: `tests/test_experiments.py`, `scripts/benchmark_diagnostics.py`,
  `README.md`, this audit. Some of these already had user changes; those were preserved.

Final Python discovery: **501 tests, OK, 4 browser tests skipped**, 74.879 seconds.
Report/verifier focused checks: 13 and 15 tests passed, including PDF generation.
Frontend transport: one grouped regression passed. Production build passed.
Lint completed with warnings. Index determinism/dependency regression passed.
