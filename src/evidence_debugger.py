"""Read-only evidence provenance. Missing observations stay unknown."""
import re

CITATION = re.compile(r"\[(?:Chunk-ID|Chunk):\s*([^\]]+)\]", re.I)


def citations_in(text, known_ids=()):
    known = set(known_ids)
    ids = CITATION.findall(text)
    for value in re.findall(r"\[([^\]\n]+)\]", text):
        value = value.strip()
        if value in known or re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
            ids.append(value)
    return list(dict.fromkeys(ids))


def answer_sentences(answer, known_ids=()):
    """Approximate display boundaries, preserving offsets and trailing citations."""
    rows = []
    for match in re.finditer(r"\S[\s\S]*?(?:[.!?](?:\s*\[[^\]\n]+\])*(?=\s|$)|\n+|$)", answer):
        raw = match.group().rstrip()
        if raw:
            rows.append({"sentence_id": f"S{len(rows)+1}", "text": raw,
                         "start": match.start(), "end": match.start() + len(raw),
                         "citations": citations_in(raw, known_ids)})
    return rows


def sentence_for_claim(answer, claim, known_ids=()):
    start, end = claim.get("character_start", -1), claim.get("character_end", -1)
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(answer)):
        text = claim.get("claim_text", "")
        start = answer.find(text) if text else -1
        if start < 0 or answer.find(text, start + 1) >= 0:
            return []
        end = start + len(text)
    return [s for s in answer_sentences(answer, known_ids) if start < s["end"] and end > s["start"]]


def build_evidence_debugger(trace):
    diagnostics = trace.get("diagnostics") or {}
    payload = trace.get("claim_verification") or diagnostics.get("claim_verification") or {}
    claims = payload.get("results", [])
    known_ids = {r["chunk_id"] for r in trace.get("retrieved_chunk_references", [])}
    sentences = answer_sentences(trace.get("generated_answer", ""), known_ids)
    linked = {s["sentence_id"]: [] for s in sentences}
    unmapped = []
    for claim in claims:
        matches = sentence_for_claim(trace.get("generated_answer", ""), claim, known_ids)
        if not matches:
            unmapped.append(claim["claim_id"])
        for sentence in matches:
            linked[sentence["sentence_id"]].append(claim)
    for sentence in sentences:
        sentence["claims"] = linked[sentence["sentence_id"]]
    selected = {r["chunk_id"] for r in trace.get("retrieved_chunk_references", [])}
    metadata = diagnostics.get("retrieval_metadata") or {}
    captured = metadata.get("candidate_chunks")
    excluded_by_id = {}
    for candidate in captured or []:
        if candidate["chunk_id"] in selected:
            continue
        item = excluded_by_id.setdefault(candidate["chunk_id"], {**candidate, "observations": []})
        item["observations"].append({k: candidate.get(k) for k in ("retrieval_query", "candidate_rank", "reranker_score")})
    excluded = list(excluded_by_id.values())
    checks = [check for claim in claims for check in claim.get("citation_checks", [])]
    signals = []
    if any(c["status"] in ("UNSUPPORTED", "CONTRADICTED") for c in claims):
        signals.append({"kind": "generation", "message": "One or more claims lack support in the supplied evidence. This does not by itself prove a retrieval failure."})
    if any(c["status"] in ("PARTIALLY_SUPPORTED", "NOT_VERIFIABLE") for c in claims) or not claims:
        signals.append({"kind": "uncertain", "message": "Some judgments are ambiguous or unavailable. Unmapped sentences are not automatically supported."})
    if any(c["status"] in ("UNSUPPORTED", "CONTRADICTED", "MISSING_EVIDENCE") for c in checks):
        signals.append({"kind": "citation", "message": "A cited passage was missing or did not support its associated claim under the local NLI check."})
    case = diagnostics.get("regression_case") or {}
    expected = case.get("expected_evidence") or []
    missing = [cid for cid in expected if cid not in selected]
    if expected and case.get("scope") == "retrieve":
        signals.append({"kind": "retrieval", "message": ("Expected prompt evidence is missing: " + ", ".join(missing) + ". Inspect excluded candidates to distinguish ranking/truncation from search misses.")
                        if missing else "All engineer-labeled expected passage IDs reached the prompt. This does not establish complete retrieval recall."})
    elif expected:
        signals.append({"kind": "retrieval", "message": "This case reused captured evidence; current retrieval was not measured."})
    else:
        signals.append({"kind": "retrieval", "message": "Retrieval failure is unconfirmed without expected evidence labels. Use a regression case to specify those labels."})
    return {"sentences": sentences, "claims": claims, "unmapped_claim_ids": unmapped, "excluded_candidates": excluded,
            "candidate_capture": "captured" if captured is not None else "unavailable",
            "candidate_scope": metadata.get("candidate_capture_scope", "legacy trace"),
            "signals": signals,
            "note": "Sentence boundaries are approximate. Claim extraction can omit facts; model support scores are not proof of truth."}
