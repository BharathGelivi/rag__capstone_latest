"""Paired strategy comparisons over the same saved questions."""
from copy import deepcopy
from itertools import combinations
import random

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.trace_replay import ReplayOptions, load_trace, replay_trace
from src.strategy_arms import ARMS

router = APIRouter(prefix="/ui/arena", tags=["strategy arena"])


class ArenaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    trace_ids: list[str] = Field(min_length=1, max_length=20)
    arms: list[str] = Field(min_length=2, max_length=6)
    input_per_million: float = Field(default=0, ge=0)
    output_per_million: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def unique_known(self):
        if len(set(self.arms)) != len(self.arms) or any(a not in ARMS for a in self.arms):
            raise ValueError("Select distinct known strategies")
        if len(set(self.trace_ids)) != len(self.trace_ids):
            raise ValueError("Question traces must be distinct")
        return self


def paired(rows, arms):
    reports = []
    for a, b in combinations(arms, 2):
        pairs = []
        for row in rows:
            values = {r["arm"]: r for r in row["results"]}
            left, right = values[a].get("supported_fraction"), values[b].get("supported_fraction")
            if left is not None and right is not None:
                pairs.append({"trace_id": row["source_trace_id"], "difference": right - left})
        deltas = [p["difference"] for p in pairs]
        interval = None
        if len(deltas) >= 2:
            rng = random.Random(42)
            samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
            interval = [samples[49], samples[1949]]
        reports.append({"baseline": a, "challenger": b, "pairs": pairs, "n": len(deltas),
                        "mean_difference": sum(deltas) / len(deltas) if deltas else None,
                        "bootstrap_95_percent": interval,
                        "metric": "verified supported-claim fraction; not factual accuracy"})
    return reports


def run_arena(request, progress=None, checkpoint=None):
    rows = deepcopy(checkpoint or [])
    for trace_id in request.trace_ids:
        source = load_trace(trace_id)
        row = next((r for r in rows if r["source_trace_id"] == trace_id), None)
        if row is None:
            row = {"source_trace_id": trace_id, "question": source.question, "results": []}
            rows.append(row)
        for arm in request.arms:
            if any(r["arm"] == arm for r in row["results"]):
                continue
            if progress:
                progress("running", rows)
            try:
                output = replay_trace(source, ReplayOptions(mode="retrieve", arm=arm),
                                      progress=(lambda stage: progress(stage, rows)) if progress else None)
                trace = output["replay"]
                stats = trace["execution_statistics"]
                # Character estimates cover final generation only, never hidden planner/judge calls.
                inputs = (len(trace["prompt_snapshot"]) + 3) // 4
                outputs = (len(trace["generated_answer"]) + 3) // 4
                row["results"].append({"arm": arm, "trace": trace,
                    "supported_fraction": output["comparison"]["scores"]["replay"]["supported_fraction"],
                    "latency_seconds": stats.get("replay_total_time"),
                    "estimated_input_tokens": inputs, "estimated_output_tokens": outputs,
                    "estimated_generation_cost_usd": (inputs * request.input_per_million + outputs * request.output_per_million) / 1e6,
                    "cost_scope": "Final generation only; character estimates, excludes strategy and judge calls"})
            except (ValueError, RuntimeError) as exc:
                row["results"].append({"arm": arm, "error": str(exc)})
            if progress:
                progress("checkpoint", rows)
    return {"rows": rows, "paired": paired(rows, request.arms),
            "uncertainty_note": "Paired question bootstrap, seed 42, 2000 samples. No interval for fewer than two pairs. Small or selected sets may be unrepresentative."}


@router.get("/strategies")
def strategies():
    return ARMS


@router.post("/run")
def run(request: ArenaRequest):
    try:
        return run_arena(request)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Question trace not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
