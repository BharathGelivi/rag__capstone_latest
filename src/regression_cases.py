"""Portable, reviewable regression cases and deterministic assertion reporting."""
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Expectations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_evidence: list[str] = Field(default_factory=list, max_length=100)
    required_facts: list[str] = Field(default_factory=list, max_length=100)
    forbidden_claims: list[str] = Field(default_factory=list, max_length=100)
    refusal: Literal["required", "allowed", "forbidden"] = "forbidden"
    fact_match: Literal["text", "nli"] = "text"

    @model_validator(mode="after")
    def nonblank(self):
        for values in (self.expected_evidence, self.required_facts, self.forbidden_claims):
            if any(not v.strip() or len(v) > 2000 for v in values):
                raise ValueError("Assertions must be nonblank and at most 2000 characters")
        if self.refusal == "allowed" and not any((self.expected_evidence, self.required_facts, self.forbidden_claims)):
            raise ValueError("Add at least one assertion")
        return self


class RegressionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    case_id: str = Field(default_factory=lambda: str(uuid4()), pattern=r"^[A-Za-z0-9_-]{1,128}$")
    name: str = Field(min_length=1, max_length=160)
    source_trace_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expectations: Expectations
    snapshot: dict

    @model_validator(mode="after")
    def usable_snapshot(self):
        if not self.name.strip() or not isinstance(self.snapshot.get("question"), str) or not self.snapshot["question"].strip():
            raise ValueError("A named case and captured question are required")
        if not isinstance(self.snapshot.get("generated_answer"), str):
            raise ValueError("Captured answer is required")
        chunks = self.snapshot.get("retrieved_chunk_references")
        if not isinstance(chunks, list) or any(not isinstance(c, dict) or not c.get("chunk_id") or not isinstance(c.get("text"), str) for c in chunks):
            raise ValueError("Captured evidence must include chunk IDs and text")
        return self


def load_cases(directory):
    paths = sorted(Path(directory).glob("*.json"))
    cases = [RegressionCase.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Duplicate regression case IDs")
    return cases


def save_case(case, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{case.case_id}.json"
    # Exclusive creation prevents a repeated save from replacing reviewed expectations.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(case.model_dump_json(indent=2) + "\n")
    return path


def normalize(text):
    return " ".join(text.casefold().split())


def evaluate_case(case, candidate, judge=None):
    """Text assertions are literal; optional NLI assesses what the answer asserts."""
    from src.rag_eval import is_refusal
    answer = candidate.get("generated_answer", "")
    refused = is_refusal(answer)
    expectations = case.expectations
    checks = []
    def add(kind, assertion, status, detail=""):
        checks.append({"kind": kind, "assertion": assertion, "status": status, "detail": detail})
    if not answer.strip():
        add("answer", "Nonempty answer", "failed")
    if expectations.refusal != "allowed":
        add("refusal", expectations.refusal, "passed" if refused == (expectations.refusal == "required") else "failed", "Refusal detection is heuristic.")
    ids = {c["chunk_id"] for c in candidate.get("retrieved_chunk_references", [])}
    for cid in expectations.expected_evidence:
        add("evidence", cid, "passed" if cid in ids else "failed", "Must be in the prompt context, not merely cited by the answer.")
    for kind, values in (("required_fact", expectations.required_facts), ("forbidden_claim", expectations.forbidden_claims)):
        for fact in values:
            if kind == "required_fact" and refused and expectations.refusal in ("required", "allowed"):
                add(kind, fact, "not_applicable", "Refusal is permitted for this case.")
            elif expectations.fact_match == "text":
                found = normalize(fact) in normalize(answer)
                add(kind, fact, "passed" if found == (kind == "required_fact") else "failed", "Literal phrase matching, including quoted or negated text.")
            elif judge is None:
                add(kind, fact, "unavailable", "NLI judge was not loaded; this is not a pass.")
            else:
                try:
                    score = judge(answer, fact)["entailment"]
                    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                        raise ValueError("Invalid NLI score")
                    status = ("passed" if score >= .8 else "failed") if kind == "required_fact" else (
                        "failed" if score >= .8 else "passed" if score <= .2 else "uncertain")
                    add(kind, fact, status, f"Local NLI entailment={score:.3f}; support is not world-truth verification.")
                except Exception:
                    add(kind, fact, "unavailable", "NLI judging failed; this is not a pass.")
    return {"case_id": case.case_id, "name": case.name,
            "passed": all(c["status"] in ("passed", "not_applicable") for c in checks), "checks": checks}


def capture_case(trace, name, expectations):
    """Export document evidence, excluding conversation memory and private prompt history."""
    from dataclasses import asdict
    from src.trace_replay import captured_chunks
    snapshot = asdict(trace)
    snapshot["retrieved_chunk_references"] = [{"chunk_id": c.chunk_id, "text": c.chunk_text,
        "rank": c.rank, "source_file": Path(c.source_file).name, "page_number": c.page_number,
        "chunk_index": c.chunk_index, "similarity_score": c.similarity_score} for c in captured_chunks(trace)]
    snapshot["diagnostics"] = {}
    snapshot["prompt_snapshot"] = ""
    snapshot["prompt_length"] = 0
    snapshot["configuration_snapshot"] = {k: v for k, v in snapshot["configuration_snapshot"].items()
        if k in ("corpus", "arm", "llm_model", "llm_temperature", "llm_max_tokens", "retrieval_top_k")}
    return RegressionCase(name=name, source_trace_id=trace.trace_id, expectations=expectations, snapshot=snapshot)


def generate_candidate(case, mode="generate"):
    from dotenv import load_dotenv
    load_dotenv()
    from copy import deepcopy
    from src.generator import PromptBuilder
    from src.rag_trace import RAGTrace
    from src.trace_replay import ReplayOptions, captured_chunks, replay_trace
    from configs.prompts import LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS
    from configs.models import NVIDIA_GENERATION_MODEL
    source = RAGTrace.from_json(deepcopy(case.snapshot))
    messages = PromptBuilder.build_messages(source.question, captured_chunks(source), LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS)
    source.diagnostics = {"generation_metadata": {"messages": [{"role": m.role.value, "content": m.content} for m in messages]}}
    source.prompt_snapshot = "\n".join(f"[{m.role.value.upper()}]: {m.content}" for m in messages)
    return replay_trace(source, ReplayOptions(mode="retrieve" if mode == "retrieve" else "frozen",
                       model=NVIDIA_GENERATION_MODEL, verify_claims=False),
                       regression_case={"case_id": case.case_id, "expected_evidence": case.expectations.expected_evidence,
                                        "scope": mode})["replay"]


def run_case(case, mode="generate"):
    try:
        candidate = case.snapshot if mode == "snapshot" else generate_candidate(case, mode)
        judge = None
        if case.expectations.fact_match == "nli" and (case.expectations.required_facts or case.expectations.forbidden_claims):
            from src import chat_service
            def judge(premise, hypothesis):
                with chat_service.GPU_LOCK:
                    return chat_service.load_verifier().run_nli(premise, hypothesis)
        result = evaluate_case(case, candidate, judge)
        baseline = evaluate_case(case, case.snapshot, judge)
        result["baseline_passed"] = baseline["passed"]
        result["change"] = ("regressed" if baseline["passed"] and not result["passed"] else
                            "fixed" if result["passed"] and not baseline["passed"] else "unchanged")
        result["candidate_trace_id"] = candidate.get("trace_id")
        result["answer"] = candidate.get("generated_answer", "")
        result["evidence_ids"] = [c["chunk_id"] for c in candidate.get("retrieved_chunk_references", [])]
        return result
    except Exception:
        return {"case_id": case.case_id, "name": case.name, "passed": False, "change": "unavailable",
                "checks": [{"kind": "execution", "status": "unavailable",
                            "detail": "Generation or judging failed. Check provider access, corpus availability and model setup."}]}
