# Overnight repair study with 100 scenarios

The prepared plan has exactly **100 scenarios**, each with baseline, broken and
repaired conditions. It supersedes the earlier 40-trace plan for tonight's run.

There are 20 cases in each group.

- A required document is removed from the isolated test corpus.
- A required passage is demoted below the context cutoff while remaining in the corpus.
- A sentence is split across adjacent chunks and the context window is narrowed.
- A misleading generator instruction demands a wrong numeric answer.
- A healthy control receives no injected fault, allowing false alarms and regressions to be measured.

Each scenario has its own fictional policy, amounts, deadline and waiver code.
These are controlled templates with known source facts. They provide a direct
test of induced failures and repair behavior. Generalization to real legal
documents requires a separate evaluation on real traces.

## Prepare offline

Run from the repository root in PowerShell.

```powershell
.\venv\Scripts\python.exe -m scripts.run_repair_study
```

This writes `artifacts/repair_study/overnight-100/scenarios.json` and a frozen
manifest. It performs no inference, model loading or NVIDIA requests.

## Launch tonight

```powershell
.\venv\Scripts\python.exe -u -m scripts.run_repair_study --run 2>&1 | Tee-Object -FilePath artifacts/repair_study/overnight-100/run.log -Append
```

The script uses the existing project environment and `NVIDIA_API_KEY` in `.env`.
Keep the terminal open and the machine awake. Rerun the same command after an
interruption to reuse successful checkpoints.

The generation and decomposition model comes from the project's model config.
Judges default to DeepSeek V4.1 Flash and Mistral Large 4. Mistral passed an
earlier minimal availability probe. DeepSeek's probe timed out and its access
remains unconfirmed. The first real baseline checks both judges before the
remaining scenarios proceed. Their access errors stop the run and preserve
completed work for resumption. No model is silently substituted.

There are up to **300 generated conditions and 600 independent judge judgments**.
Generation and claim decomposition require additional NVIDIA calls. Each failed
condition has up to three attempts; the existing rate limiter may also retry
rate-limited generation/decomposition requests. Judge judgments have up to three
attempts. These are model workloads, so 100 scenarios can involve more than 100
requests. Model usage depends on the provider and retry behavior.

If no applicable repair is suggested, the broken answer is carried forward and
its identical judge results are reused. This avoids credit for judgment noise.
A healthy case without an applicable recommendation gets another generation
using the same settings as a sham comparison.

## How a suggested fix is tested

For every condition, the runtime uses the existing hybrid retriever, embedding
model, reranker, generator, claim decomposer, local NLI verifier, pipeline state
analyzer, root cause reasoner and corrective-action engine. The index is an
in-memory Chroma test index. Production corpus indexing is outside this study.

Ambiguous NLI verdicts use no remote judge escalation. This makes the diagnostic
configuration explicit and keeps the study's remote workload bounded.

The broken condition is diagnosed first. Its actual corrective-action plan is
saved. The harness applies the first executable, applicable recommendation in
the engine's priority order. The injected fault label is excluded from repair
selection. A wrong recommendation can fail to repair the answer.

Supported adapters perform the following changes.

- **Ingest Missing Domain Knowledge** adds the held-out source document to the
  test corpus and reruns retrieval. The source document is supplied by the
  experiment. Finding the missing document is outside the measured repair.
- **Increase Retrieval Top-K** expands the context limit by four. The faulty
  ranking persists, so the larger window must reach the necessary evidence.
- **Increase Chunk Overlap** joins neighboring fragments of the same source
  into overlapping text and re-embeds them. This tests a context repair without
  rebuilding the production document ingestion pipeline.
- **Lower Sampling Temperature** changes only the sampling temperature to zero.
- **Strict Prompt Grounding** appends instructions to decline unsupported parts.
- **Require Explicit Citations** appends per-sentence citation and numeric-check
  instructions. The earlier misleading instruction remains in the prompt.

Unimplemented or inapplicable suggestions are recorded. Missing recommendations
are visible as unrepaired cases. The study measures the first applicable fix
and does not search over fixes until an answer passes.

## What you can inspect

During execution, open the following local artifact in a browser.

`artifacts/repair_study/overnight-100/REPORT.html`

The report refreshes on disk after each completed scenario. Reload the browser
to see new results. Each row shows the injected condition, X-RAG's diagnosis,
the applied recommendation, before/after quality and whether recovery occurred.
Expanding the row shows the three answers, the actual retrieved passages, the
recorded fix, and both model judgments with quotations and raw response links.

The same pristine reference corpus is shown to both judges in every condition.
The judges are blinded to phase, fault, diagnosis and recommendation. This
measures whether the answer fulfills the question against known source facts.
X-RAG's own NLI results separately assess support in the retrieved context.
An appropriate refusal after evidence removal can therefore have low answer
completion while avoiding unsupported generation.

`summary.json` contains the complete records and per-group totals. Raw generated
traces, states and diagnostics are under `conditions/`. Judge responses, evidence
checks and attempt metadata are under `judgments/`.

Quality is 1 when both judges call an answer healthy and 0 when both identify an
answer issue. Disagreement about healthy versus faulty, inconclusive judgments
and unavailable results are reported as unassessed. Different failure labels
can agree on the binary outcome while remaining visible in the raw judgments.

A recovery requires a healthy baseline, an observed degraded broken answer, an
applied recommendation and a healthy repaired answer. The report also counts
controls that regress and induced faults that fail to degrade a healthy answer.
Cases missing an after judgment stay in the denominator of the conservative
confirmed-recovery rate, with their missing-result count shown explicitly.

There is one observed run per condition. Changes in quality can include sampling
variation, especially in the misleading-prompt group. This study measures an
observed repair effect. Repeat trials would be needed to estimate its stability.

## Resume and configuration

Source code, prompts, schema, scenario data and model settings are fingerprinted.
Changed inputs require a fresh output directory, supplied with `--output`.
The scenario count remains 100. Models can be selected explicitly with
`--models MODEL_ONE MODEL_TWO`. They must belong to different provider families.

Only one process can own a run directory. A hard interruption may leave
`run.lock`, containing the owning PID. Confirm the recorded runner has stopped
before removing that specific lock and resuming. Ctrl+C may wait for a pending
request to finish or reach its timeout.
