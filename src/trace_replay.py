"""Replay saved traces without consulting mutable memory or modifying the source."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Literal

from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import chat_service
from src.generator import Generator, PromptBuilder
from src.rag_trace import RAGTrace, RAGTraceBuilder
from src.retriever import RetrievedChunk, RetrievalResult
from src.strategy_arms import ARMS, execute_arm


class ReplayOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["frozen", "retrieve"] = "frozen"
    model: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_./:-]{1,160}$")
    system_prompt: str | None = Field(default=None, min_length=1, max_length=40000)
    arm: str | None = None
    corpus: Literal["statutes", "judgments", "both"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=30, strict=True)
    verify_claims: bool = True

    @model_validator(mode="after")
    def valid_changes(self):
        if self.system_prompt is not None and not self.system_prompt.strip():
            raise ValueError("System prompt must not be blank")
        if self.arm is not None and self.arm not in ARMS:
            raise ValueError("Unknown retrieval strategy")
        if self.mode == "frozen" and any(v is not None for v in (self.arm, self.corpus, self.top_k)):
            raise ValueError("Retrieval settings require fresh retrieval mode")
        return self


def load_trace(trace_id: str) -> RAGTrace:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id):
        raise ValueError("Invalid trace identifier")
    root = Path(chat_service.PROJECT_ROOT) / "artifacts" / "rag_traces"
    paths = list(root.glob(f"*/trace_{trace_id}.json"))
    paths += [root / f"{trace_id}.json", root / f"trace_{trace_id}.json"]
    for path in paths:
        if path.is_file():
            return RAGTrace.from_json(json.loads(path.read_text(encoding="utf-8")))
    raise FileNotFoundError("Trace not found")


def captured_messages(trace):
    stored = (trace.diagnostics or {}).get("generation_metadata", {}).get("messages")
    if stored:
        return [ChatMessage(role=m["role"], content=m["content"]) for m in stored]
    # Legacy traces flattened roles into prompt_snapshot. Reject absent prompts;
    # a registry lookup would silently turn a frozen replay into new evidence.
    parts = re.split(r"(?m)^\[(SYSTEM|USER|ASSISTANT)\]: ", trace.prompt_snapshot)
    if len(parts) < 3 or parts[0].strip():
        raise ValueError("This trace has no usable captured prompt")
    return [ChatMessage(role=parts[i].lower(), content=parts[i + 1].removesuffix("\n"))
            for i in range(1, len(parts), 2)]


def captured_chunks(trace):
    snapshot = dict(re.findall(
        r"--- Context chunk \d+ \[Chunk-ID: ([^\]]*)\] ---\n(.*?)(?=\n--- Context chunk |\n\nQuestion: |\Z)",
        trace.prompt_snapshot, re.DOTALL))
    chunks = []
    fields = RetrievedChunk.__dataclass_fields__
    for ref in trace.retrieved_chunk_references:
        text = ref.get("text") or ref.get("chunk_text") or snapshot.get(ref["chunk_id"])
        if not text:
            raise ValueError("Captured evidence is incomplete; use fresh retrieval instead")
        values = {k: v for k, v in ref.items() if k in fields}
        for key, value in {"rank": len(chunks) + 1, "similarity_score": 0.0,
                           "page_number": "", "source_file": "", "chunk_index": 0}.items():
            values.setdefault(key, value)
        values["chunk_text"] = text if ref.get("text") or ref.get("chunk_text") else text.strip()
        chunks.append(RetrievedChunk(**values))
    return chunks


def editable_prompt(messages):
    systems = [m.content or "" for m in messages if m.role.value == "system"]
    return systems[-1] if systems else ""


def trace_choices(limit=50):
    root = Path(chat_service.PROJECT_ROOT) / "artifacts" / "rag_traces"
    files = sorted(list(root.glob("*/trace_*.json")) + list(root.glob("*.json")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    choices = []
    for path in files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            choices.append({k: value[k] for k in ("trace_id", "question", "timestamp")})
        except (ValueError, KeyError, OSError):
            continue
        if len(choices) >= limit:
            break
    return choices


def _view(trace):
    value = asdict(trace)
    try:
        evidence = {c.chunk_id: c.chunk_text for c in captured_chunks(trace)}
        for ref in value["retrieved_chunk_references"]:
            ref["text"] = evidence[ref["chunk_id"]]
    except ValueError:
        pass
    value.update({k: v for k, v in (trace.diagnostics or {}).items()
                  if k in ("claim_verification", "claim_error")})
    return value


def _scores(trace):
    results = (trace.diagnostics or {}).get("claim_verification", {}).get("results", [])
    return {"claim_count": len(results),
            "supported_fraction": sum(r["status"] == "SUPPORTED" for r in results) / len(results)
            if results else None,
            "statuses": {s: sum(r["status"] == s for r in results) for s in
                         ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "NOT_VERIFIABLE")}}


def _claim_changes(original, replay):
    panels = [(t.diagnostics or {}).get("claim_verification") for t in (original, replay)]
    if any(p is None for p in panels):
        return None
    before, after = [{" ".join(r["claim_text"].lower().split()): r for r in p.get("results", [])}
                     for p in panels]
    return [{"claim_text": (after.get(key) or before[key])["claim_text"],
             "original_status": before[key]["status"] if key in before else None,
             "replay_status": after[key]["status"] if key in after else None}
            for key in sorted(before.keys() | after.keys())
            if before.get(key, {}).get("status") != after.get(key, {}).get("status")]


def replay_trace(source: RAGTrace, options: ReplayOptions, *, regression_case=None, progress=None):
    started = time.perf_counter()
    source = deepcopy(source)
    messages = captured_messages(source)
    if not messages or messages[-1].role.value != "user":
        raise ValueError("Captured prompt must end with a user question")
    old_config = source.configuration_snapshot
    corpus = options.corpus or old_config.get("corpus")
    arm = options.arm or old_config.get("arm") or "C_hybrid_rerank"
    top_k = options.top_k or len(source.retrieved_chunk_references) or 6
    retrieval_started = time.perf_counter()
    metadata = deepcopy((source.diagnostics or {}).get("retrieval_metadata") or {})
    search_query = metadata.get("search_query") or source.question
    if options.mode == "frozen":
        chunks = captured_chunks(source)
        retrieval_time = 0.0
    else:
        if corpus not in chat_service.CORPUS_OPTIONS:
            raise ValueError("Select a corpus for fresh retrieval of this legacy trace")
        if arm not in ARMS or (corpus == "both" and arm != "C_hybrid_rerank"):
            raise ValueError("Strategy is incompatible with this corpus")
        if corpus == "statutes" and (ARMS[arm].get("graph") or ARMS[arm].get("graph_expand")
                                     or arm == "C_fixed_chunking"):
            raise ValueError("This strategy requires the judgments corpus")
        pipeline_corpus = "judgments_fixed" if arm == "C_fixed_chunking" else corpus
        if progress:
            progress("loading models")
        with chat_service.GPU_LOCK:
            if corpus == "both":
                chunks = []
                candidates = []
                for name in ("statutes", "judgments"):
                    retriever, _, _, error = chat_service.load_pipeline(name)
                    if error:
                        raise ValueError(error)
                    result = retriever.retrieve(search_query, top_n=top_k)
                    chunks.extend(result.retrieved_chunks)
                    candidates.extend(result.retrieval_metadata.get("candidate_chunks", []))
                chunks = sorted(chunks, key=lambda c: c.reranker_score, reverse=True)[:top_k]
                metadata = {"combined_corpora": ["statutes", "judgments"], "candidate_chunks": candidates,
                            "candidate_capture_scope": "Both corpus retrieval calls, before final merging and truncation."}
            else:
                retriever, planner, registry, error = chat_service.load_pipeline(pipeline_corpus)
                if error:
                    raise ValueError(error)
                if progress:
                    progress("retrieving")
                chunks, metadata, _ = execute_arm(
                    arm, search_query, registry, retriever,
                    chat_service.load_knowledge_graph() if corpus == "judgments" else None,
                    planner.llm, top_n=top_k)
        messages[-1] = PromptBuilder.build_messages(source.question, chunks)[-1]
        retrieval_time = time.perf_counter() - retrieval_started
        metadata = {**metadata, "arm": arm, "corpus": corpus, "retrieval_mode": ARMS[arm]["mode"],
                    "rerank_enabled": ARMS[arm]["rerank"]}
    if options.system_prompt is not None:
        system_indices = [i for i, m in enumerate(messages) if m.role.value == "system"]
        if system_indices:
            messages[system_indices[-1]] = ChatMessage(role="system", content=options.system_prompt)
        else:
            messages.insert(0, ChatMessage(role="system", content=options.system_prompt))
    retrieval = RetrievalResult(source.question, 0, chunks, [c.chunk_id for c in chunks],
                                [c.similarity_score for c in chunks], retrieval_time, len(chunks), metadata)
    if progress:
        progress("waiting for inference")
    generator = Generator(model_name=options.model or old_config.get("llm_model"),
                          temperature=old_config.get("llm_temperature", 0.0),
                          max_tokens=old_config.get("llm_max_tokens", 2048))
    generation_metadata = (source.diagnostics or {}).get("generation_metadata") or {}
    if generation_metadata.get("streamed"):
        reasoning = generation_metadata.get("reasoning_mode", "<thinking>" in editable_prompt(messages))
        for _ in generator.generate_stream(retrieval, reasoning=reasoning, messages_override=messages):
            pass
        generation = generator.last_stream_result
    else:
        generation = generator.generate(retrieval, messages_override=messages)
    if not generation.ok:
        raise RuntimeError("Replay generation failed; please retry or choose an accessible model")
    trace = RAGTraceBuilder.build(retrieval, generation, retrieval_time + generation.generation_time)
    if options.mode == "frozen":
        for key, value in old_config.items():
            if key not in ("llm_model", "llm_temperature", "temperature", "llm_max_tokens"):
                trace.configuration_snapshot[key] = deepcopy(value)
    warnings = ["Remote inference can vary even with unchanged settings; provider model revisions are not pinned."]
    if not (source.diagnostics or {}).get("generation_metadata", {}).get("messages"):
        warnings.append("Legacy prompt roles were recovered from text; new traces capture structured messages.")
    if generation.generation_metadata.get("finish_reason") == "length":
        warnings.append("The replay answer reached its token limit.")
    memory_evidence = (source.diagnostics or {}).get("memory_evidence", [])
    if (source.diagnostics or {}).get("memory_ids") and not memory_evidence:
        warnings.append("Original memory remains in the captured prompt, but legacy memory evidence is unavailable for verification.")
    trace.diagnostics["memory_evidence"] = memory_evidence
    trace.diagnostics["memory_ids"] = (source.diagnostics or {}).get("memory_ids", [])
    if regression_case is not None:
        trace.diagnostics["regression_case"] = deepcopy(regression_case)
    trace.diagnostics["replay"] = {
        "source_trace_id": source.trace_id, "mode": options.mode,
        "source_sha256": hashlib.sha256(source.to_json().encode()).hexdigest(),
        "requested_top_k": top_k if options.mode == "retrieve" else None,
        "warnings": warnings,
    }
    if options.verify_claims:
        if progress:
            progress("verifying claims")
        try:
            verification_evidence = chunks + chat_service.captured_memory_chunks(trace)
        except ValueError as exc:
            outcome = {"error": str(exc)}
        else:
            outcome = chat_service.deep_analyze(trace, verification_evidence)
        trace.diagnostics["claim_error" if "error" in outcome else "claim_verification"] = (
            outcome.get("error") if "error" in outcome else outcome["verification"])
    trace.execution_statistics["replay_total_time"] = time.perf_counter() - started
    RAGTraceBuilder.save_to_json(trace, str(Path(chat_service.PROJECT_ROOT) / "artifacts" / "rag_traces"))
    old_ids = [c["chunk_id"] for c in source.retrieved_chunk_references]
    new_ids = [c.chunk_id for c in chunks]
    return {"original": _view(source), "replay": _view(trace), "warnings": warnings,
            "comparison": {"evidence_added": [c for c in new_ids if c not in old_ids],
                           "claim_changes": _claim_changes(source, trace),
                           "evidence_removed": [c for c in old_ids if c not in new_ids],
                           "evidence_order_changed": old_ids != new_ids,
                           "answer_changed": source.generated_answer != trace.generated_answer,
                           "scores": {"original": _scores(source), "replay": _scores(trace)}}}
