from types import SimpleNamespace
from src.corpus_inspection import inspect_records, legal_metadata


def test_duplicates_and_missing_page_are_reported_without_mutating_records():
    records = [SimpleNamespace(chunk_id="a", text="Section 12 text", source_file="act.pdf", page_number="1"),
               SimpleNamespace(chunk_id="b", text=" section 12  text ", source_file="act.pdf", page_number="unknown")]
    findings = inspect_records(records)
    assert {f["kind"] for f in findings} == {"duplicate_chunks", "missing_page"}
    assert next(f for f in findings if f["kind"] == "duplicate_chunks")["chunk_ids"] == ["a", "b"]
    assert records[1].text == " section 12  text "


def test_legal_metadata_preserves_versions_and_marks_section_mentions():
    result = legal_metadata("Section 103 applies", {"amendment_date": "2024-01-01", "version": "v2"})
    assert result["metadata"]["version"] == "v2"
    assert result["section_mentions"] == ["Section 103"]
