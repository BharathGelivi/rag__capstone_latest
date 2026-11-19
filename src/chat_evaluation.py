"""Fast evaluation scores that can be recovered from captured chat evidence."""

import re

from src.evidence_debugger import citations_in

STOPWORDS = {
    "that", "this", "with", "from", "which", "under", "shall", "have",
    "been", "about", "there", "their",
}


def evaluate_answer(answer, references, verification):
    texts = [ref.get("text") or ref.get("chunk_text") for ref in references]
    sentences = [s for s in re.split(r"[.!?\n]+", answer.lower()) if len(s.strip()) > 15]
    groundedness = None
    if sentences and (not references or all(texts)):
        context = " ".join(texts).lower()
        grounded = 0
        for sentence in sentences:
            words = [word for word in re.findall(r"\b\w{4,}\b", sentence) if word not in STOPWORDS]
            grounded += bool(words and sum(word in context for word in words) / len(words) >= 0.45)
        groundedness = round(grounded / len(sentences), 4)

    ids = {ref["chunk_id"] for ref in references if ref.get("chunk_id")}
    citations = set(citations_in(answer, ids))
    results = (verification or {}).get("results") or []
    return {
        "groundedness": groundedness,
        "faithfulness": round(sum(r.get("status") == "SUPPORTED" for r in results) / len(results), 4) if results else None,
        "citation_precision": round(len(citations & ids) / len(citations), 4) if citations else None,
        "citations_found": len(citations),
    }
