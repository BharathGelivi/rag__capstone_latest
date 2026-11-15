# Prepare and run independent NVIDIA judges

For tonight's requested **100 scenarios with induced failures and repair
measurements**, use [the repair study instructions](repair_study.md) and
`python -m scripts.run_repair_study --run`. The commands below remain available
for the earlier historical-trace comparison.

This study asks two model families to independently assess the 40 saved benchmark
answers using the exact document passages captured in their generation prompts.
The judges receive the question, answer and passages. X-RAG's diagnosis, NLI
verdicts, baseline scores and the dataset's speculative failure labels are withheld.

The defaults are `deepseek-ai/deepseek-v4.1-flash` and
`mistralai/mistral-large-4-1t-a52b`. Both appeared in this account's NVIDIA model
catalog on 15 November 2026. A minimal Mistral completion succeeded. DeepSeek's
minimal probe timed out, so its availability requires confirmation during the run.
Kimi K2.6 returned an account access error. The runner records the requested and
returned model names with every successful response.

## Prepare offline

From the repository root in PowerShell, run the following command.

```powershell
.\venv\Scripts\python.exe -m scripts.run_independent_judges
```

This checks all 40 traces and writes the frozen cases, judge prompt, response
schema, model settings and SHA-256 fingerprints under
`artifacts/independent_judges/overnight/`. Preparation makes zero API calls and
requires no key. The original benchmark files and diagnoses remain the source
of the comparison.

## Start overnight

When ready, use the following command from the same directory.

```powershell
.\venv\Scripts\python.exe -u -m scripts.run_independent_judges --run 2>&1 | Tee-Object -FilePath artifacts/independent_judges/overnight/run.log -Append
```

The runner reads the existing `NVIDIA_API_KEY` from `.env` or the environment.
Keep the terminal open and the computer awake. These are remote inference calls
using your NVIDIA account. Their duration and any provider charges depend on the
endpoint. The report records returned token usage. It cannot determine a dollar
cost without provider pricing.

Each model first receives seven fictional controls covering support,
contradiction, unsupported additions, omission, unjustified refusal, justified
refusal and prompt injection. It then judges the 40 benchmark cases.
That is **94 planned judgments**, with at most three requests per judgment.
Two model workers share the project's request limiter. The default request
timeout is 240 seconds and the output-token limit is 8192.

Availability failures with HTTP 400, 401, 403 or 404 stop work for that model.
The first control also checks model availability and response format. Exhausting
its retries stops that model before the remaining cases are sent.
Transient failures, malformed JSON, truncated responses and invalid quotations
receive bounded retries. Model failures appear in the report and produce a
nonzero process exit. The runner uses the two explicitly selected models.

## Resume or inspect

Repeat the same launch command to reuse successful judgments and retry failed
or unattempted cases. Each completed case is saved atomically. Only one process
can own the output directory at a time. Ctrl+C requests a stop after pending
requests finish or time out.

If a hard interruption leaves `run.lock`, check the PID recorded in that file.
After confirming the associated runner has stopped, remove that specific lock
and repeat the command.

```powershell
Get-Content artifacts/independent_judges/overnight/run.lock
# After confirming the recorded runner has stopped:
Remove-Item -LiteralPath artifacts/independent_judges/overnight/run.lock
```

Changed sources, prompt, code, schema or settings require a fresh output directory.
For example, the following prepares a separate short trial offline.

```powershell
.\venv\Scripts\python.exe -m scripts.run_independent_judges --limit 2 --output artifacts/independent_judges/trial
```

Add `--run` when ready to execute that trial. To change model selection, supply
`--models MODEL_ONE MODEL_TWO` and choose a fresh `--output` directory.

Results are written to the following files.

- `REPORT.md` contains completion counts, controls, agreement and cases requiring inspection.
- `summary.json` contains per-model counts, token usage, paired agreement, binary
  Cohen's kappa and per-case comparisons to X-RAG.
- `judgments/` contains each model's raw answer, validated judgment, timing,
  response identity, token usage and failed-attempt metadata.
- `cases.json` contains the exact evidence and answers used in this study.
- `manifest.json` freezes the protocol and fingerprints used for safe resumption.

Quotations must occur in the answer or specified passage, with whitespace
normalization only. A quoted passage establishes provenance. The judge's semantic
interpretation still requires scrutiny. Missing results are reported explicitly.
Inconclusive judgments are excluded from binary failure kappa.

## Interpret the study

Agreement between two models creates a provisional machine-reviewed reference.
Shared mistakes remain possible. The seven controls check basic behavior and
provide a small sanity check on each judge.

The study distinguishes healthy answers, unsupported generation, contradictions,
incomplete answers and inconclusive cases. It evaluates historical answers from
the saved traces. It does not establish whether the corpus lacked a fact, the
retriever missed it, or a chunk boundary caused failure. Those questions require
corpus inspection and controlled repair experiments. X-RAG's `UNKNOWN` cause is
kept explicit because it can represent an unassigned cause or a healthy answer.
All reported comparisons describe agreement with model judgments.

Potential follow-up work with the greatest effect on confidence in the project
includes measuring actual repair success, calibrating diagnostic uncertainty,
recording complete evaluation cost, evaluating held-out traces from another
corpus, and updating the paper to match the code and current results.
