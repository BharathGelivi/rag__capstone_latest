"""Run an isolated 40-example suite, monitor it, and save descriptive statistics."""
from __future__ import annotations
import datetime as dt
import json
import html
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)

def main():
    from experiments.run_all import load_experiments
    from experiments.common import ExperimentContext
    run_id = "run40_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "artifacts" / "experiments" / "runs" / run_id
    directory.mkdir(parents=True)
    ctx = ExperimentContext(mode="live", limit=40, allow_small_sample=True,
                            base_dir=str(directory), extra={"sample_spread": True})
    planned = []
    for experiment in load_experiments():
        print(f"Planning E{experiment.number}: {experiment.key}", flush=True)
        specs = experiment.plan(ctx)
        chosen = [specs[i * len(specs) // 40] for i in range(40)] if len(specs) > 40 else specs
        if len(chosen) != 40:
            raise RuntimeError(f"{experiment.key}: expected 40 unique examples, got {len(chosen)}")
        planned.append({"key": experiment.key, "examples": [{"id": s.example_id, "payload": s.payload} for s in chosen]})
    write(directory / "example_plan.json", planned)
    write(ROOT / "artifacts" / "experiments" / "active_run.json", {"run_id": run_id})
    command = [sys.executable, "-m", "experiments.run_all", "--mode", "live", "--limit", "40",
               "--base-dir", str(directory), "--extra", "sample_spread=true"]
    started = time.time()
    with (directory / "suite.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        checks = []
        thresholds = [4 * 3600, 6 * 3600]
        while child.poll() is None:
            elapsed = time.time() - started
            if thresholds and elapsed >= thresholds[0]:
                checks.append({"elapsed_hours": thresholds.pop(0) / 3600, "status": "running",
                               "checked_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            write(directory / "monitor.json", {"status": "running", "pid": child.pid,
                  "supervisor_pid": os.getpid(), "elapsed_seconds": elapsed, "checks": checks,
                  "next_check_hours": thresholds[0] / 3600 if thresholds else None})
            time.sleep(30)
    statistics = []
    for experiment in load_experiments():
        summary_file = directory / experiment.key / "summary.json"
        state_file = directory / experiment.key / "state.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
        statistics.append({"key": experiment.key, "title": experiment.title,
                           "status": state.get("status", "not_started"),
                           "n_examples": summary.get("n_examples", 0), "headline": summary.get("headline", {})})
    write(directory / "statistics.json", {"run_id": run_id, "sample_size_per_experiment": 40,
          "interpretation": "Descriptive results; 40 is below the suite's 50-example research floor.",
          "experiments": statistics})
    sections = []
    for result in statistics:
        rows = "".join(f"<tr><td>{html.escape(str(key).replace('_', ' '))}</td><td>{html.escape(json.dumps(value))}</td></tr>"
                       for key, value in result["headline"].items())
        sections.append(f"<section><h2>{html.escape(result['title'])}</h2><p>{result['n_examples']} examples · {html.escape(result['status'])}</p><table>{rows}</table></section>")
    (directory / "statistics.html").write_text("<!doctype html><meta charset='utf-8'><title>Experiment statistics</title><style>body{font:15px system-ui;color:#172033;background:#f8f9fb;max-width:960px;margin:40px auto;padding:24px}section{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:24px;margin:20px 0}h2{font-size:20px}td{padding:12px;border-bottom:1px solid #e5e7eb;overflow-wrap:anywhere}table{width:100%}</style><h1>40-example experiment suite</h1><p>Descriptive statistics. Each study uses its own metrics; scores are not averaged across unrelated studies. This sample is below the suite's 50-example research floor.</p>" + "".join(sections), encoding="utf-8")
    write(directory / "monitor.json", {"status": "complete" if child.returncode == 0 else "failed",
          "exit_code": child.returncode, "elapsed_seconds": time.time() - started, "checks": checks})

if __name__ == "__main__":
    main()
