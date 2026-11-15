"""Trace-specific reviewed references, separate from question-level expectations."""
import json
from pathlib import Path


REFERENCE_SOURCES = {"human_review", "model_consensus", "controlled_intervention"}
REFERENCE_LABELS = {"HEALTHY", "MISSING_CORPUS", "RETRIEVAL_MISS", "CHUNK_BOUNDARY",
                    "UNSUPPORTED_GENERATION", "CONTRADICTORY_GENERATION", "GROUNDING_FAILURE"}


def reviewed_reference(row):
    reference = row.get("diagnostic_reference")
    if not isinstance(reference, dict):
        return None
    if (not row.get("trace_id") or reference.get("trace_id") != row["trace_id"]
            or reference.get("source") not in REFERENCE_SOURCES
            or reference.get("label") not in REFERENCE_LABELS):
        return None
    return {"cause": "UNKNOWN" if reference["label"] == "HEALTHY" else reference["label"],
            "source": reference["source"]}


def load_reviewed_references(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Reviewed references must be a JSON array")
    references = {}
    for reference in rows:
        if not isinstance(reference, dict) or reviewed_reference({
                "trace_id": reference.get("trace_id"), "diagnostic_reference": reference}) is None:
            raise ValueError("Each reference needs a trace_id, explicit label and review source")
        if reference["trace_id"] in references:
            raise ValueError("Duplicate reviewed trace")
        references[reference["trace_id"]] = reference
    return references
