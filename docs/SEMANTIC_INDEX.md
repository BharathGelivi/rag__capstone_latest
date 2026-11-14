# Repository semantic index

Rebuild with `venv\Scripts\python.exe scripts/build_semantic_index.py`.
`docs/semantic_index.json` is deterministic JSON generated using Python's standard
library AST parser. It reads source without importing modules or loading models.
Search with `rg -n 'RAGTrace|ClaimVerifier' docs/semantic_index.json` or load the JSON
and filter `files` by path, imports, symbols, endpoints or artifact references.
Source hashes make stale entries detectable; rebuild after source changes.

Each file records symbols with line numbers and docstrings, public APIs, imports,
configuration dependencies, artifact path references, direct local dependencies,
reverse dependencies and importing tests. Tests are static associations, not measured
coverage. Dynamic imports and computed artifact paths require manual tracing.
Frontend imports are recorded but are not resolved into the Python dependency graph.
Core artifact inputs/outputs are curated in the generator's `ARTIFACT_IO` table;
empty lists on other files mean unclassified. Review these contracts after changing
save/load APIs. Their outputs describe capabilities, not proof a caller invoked save.

## Diagnostic flow and artifact direction

| Concept | Implementation / consumer | Input | Output |
|---|---|---|---|
| RAGTrace | `src/rag_trace.py`, `src/claim_decomposer.py` | Retrieval and generation results | Builder: `artifacts/rag_traces/<date>/trace_<id>.json`; diagnostic API: `artifacts/rag_traces/<id>.json` |
| ClaimSet | `src/claims.py` | Claim metadata | Claim representation; distinct from decomposer candidates |
| CandidateClaimSet | `src/claim_decomposer.py`, `src/claim_verifier.py` | RAGTrace answer | Candidate claims; debug prompt/response files |
| VerificationResult | `src/claim_verifier.py` | Candidates, retrieved references, `artifacts/chunk_registry.json` | `artifacts/verification/TRACE_<id>.json` via explicit save |
| PipelineStateMatrix | `src/pipeline_state_analyzer.py` | Trace, candidates, verification | `artifacts/pipeline_state_matrix/TRACE_<id>.json` via save |
| RootCauseAnalysis | `src/root_cause_reasoner.py` | State matrix | `artifacts/root_cause_analysis/TRACE_<id>.json` via save |
| CorrectiveActionPlan | `src/corrective_action_engine.py` | Root causes and state matrix | `artifacts/corrective_action_plan/TRACE_<id>.json` via save |
| DiagnosticEvaluationReport | `src/report_builder.py`, `src/report.py` | Trace, state, causes, actions, verification, optional metrics | `artifacts/reports/<id>.json` |
| Report rendering | `src/report_presenter.py`, `scripts/generate_diagnostic_report.py` | Report / trace artifacts | Markdown, HTML, PDF; standalone HTML under `artifacts/diagnostic_reports/` |

`src/runner.py` orchestrates decomposition → verification → state analysis → root
cause reasoning → corrective actions → report building. Save APIs do not imply the
runner persists every intermediate artifact: inspect callers before assuming files exist.

`src/api.py` exposes `/health`, `/version`, `/analyze`, report rendering and artifact
discovery. `src/api_ui.py` serves React chat, memory, graph and metric endpoints;
`frontend/src/lib/api.ts` defines client requests. `ui/` is the separate Streamlit UI.
`configs/` owns model names, batching, prompts, thresholds, pipeline and API settings.
`db/` contains local vector storage; `data/` contains corpus inputs; `experiments/`
contains resumable studies; `scripts/` contains benchmark and reporting utilities.

Reproducibility context: `docs/RESEARCH_LOG.md`, `docs/EXPERIMENTS.md`, and
`docs/COMPONENTS.md`. Runtime research artifacts are intentionally left intact.
