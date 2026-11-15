"""Read-only corpus checks used before rebuilding an index."""
from collections import defaultdict
import hashlib
import re


def inspect_records(records):
    """Return duplicate, extraction, and provenance findings for registry records."""
    duplicates = defaultdict(list)
    findings = []
    for record in records:
        text = record.text or ""
        normalized = " ".join(text.casefold().split())
        if normalized:
            duplicates[hashlib.sha256(normalized.encode()).hexdigest()].append(record.chunk_id)
        if not normalized or text.count("\ufffd") > max(2, len(text) // 100):
            findings.append({"kind": "broken_extraction", "chunk_id": record.chunk_id,
                             "source": record.source_file, "page": record.page_number})
        if not str(record.page_number or "").strip() or str(record.page_number).lower() in {"unknown", "none"}:
            findings.append({"kind": "missing_page", "chunk_id": record.chunk_id,
                             "source": record.source_file, "page": record.page_number})
    for ids in duplicates.values():
        if len(ids) > 1:
            findings.append({"kind": "duplicate_chunks", "chunk_ids": ids})
    return findings


def legal_metadata(text, metadata):
    """Keep supplied legal provenance; text-derived section mentions are hints."""
    return {"metadata": dict(metadata or {}),
            "section_mentions": list(dict.fromkeys(re.findall(r"\b(?:section|sections|s\.)\s+\d+[A-Za-z]*(?:\(\d+\))?", text or "", re.I)))[:30]}
