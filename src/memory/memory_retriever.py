"""
Memory Retriever Module.

Responsible for searching memory using semantic similarity and
applying multi-factor ranking (semantic, recency, frequency, importance).
Memory retrieval augments — never replaces — document retrieval.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    MemorySearchResult,
)
from src.memory.memory_store import MemoryStore
from src.memory.memory_utils import (
    compute_frequency_score,
    compute_recency_score,
    setup_memory_logger,
)

logger = logging.getLogger(__name__)
mem_logger = setup_memory_logger()


class MemoryRetriever:
    """Retrieves relevant memories based on semantic similarity and multi-factor ranking.

    Ranking formula:
        final_score = (semantic_weight × semantic_similarity)
                    + (recency_weight  × recency_score)
                    + (frequency_weight × frequency_score)
                    + (importance_weight × importance_score)

    Default weights: 0.5 / 0.2 / 0.2 / 0.1
    """

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[MemoryConfig] = None,
        embed_model: Optional[HuggingFaceEmbedding] = None,
    ) -> None:
        self.store = store
        self.config = config or MemoryConfig()
        self._embed_model = embed_model

    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        """Lazily load the embedding model, reusing the shared instance.

        Memory and document retrieval must use the SAME model (their vectors
        share one comparison space), so the name is read from
        ``configs.models.EMBEDDING_MODEL_NAME`` unless explicitly overridden.
        Constructing a second instance previously loaded an identical
        ~0.5 GB model onto the GPU for no benefit, so this goes through the
        process-wide cache instead.
        """
        if self._embed_model is None:
            from src.embedding_engine import get_shared_embed_model

            model_name = self.config.embedding_model
            if not model_name:
                from configs.models import EMBEDDING_MODEL_NAME as model_name
            self._embed_model = get_shared_embed_model(model_name)
        return self._embed_model

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[MemorySearchResult]:
        """Search memories by semantic similarity with multi-factor ranking.

        Args:
            query:      The search query text.
            top_k:      Maximum number of results to return.
            session_id: Optional filter to search within a single session.
            min_score:  Minimum final score threshold (defaults to config).

        Returns:
            Ranked list of MemorySearchResult objects.
        """
        if min_score is None:
            min_score = self.config.similarity_threshold

        # Embed the query
        query_embedding = self.embed_query(query)

        # Search ChromaDB
        raw_results = self.store.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-fetch for re-ranking
            session_id=session_id,
        )

        # Parse hits directly from the query payload (documents+metadata are
        # already in it -- re-fetching each id was an N+1 round-trip).
        hits = self.store.entries_from_query(raw_results)
        if not hits:
            return []

        # Compute max access count for frequency normalization
        max_access = max((int(h[0].access_count) for h in hits), default=1)

        # Build scored results
        scored_results: List[MemorySearchResult] = []
        for entry, dist in hits:
            # Compute individual scores
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance / 2)
            semantic = max(0.0, 1.0 - (dist / 2.0))
            recency = compute_recency_score(
                entry.timestamp, decay_hours=self.config.recency_half_life_hours
            )
            frequency = compute_frequency_score(entry.access_count, max_access)
            importance = min(1.0, entry.importance_score)

            # The threshold gates *semantic* similarity alone. Gating on the
            # blended score (the old behavior) let fresh/frequent but
            # topically unrelated memories in purely on recency+frequency.
            if semantic < min_score:
                continue

            # Multi-factor ranking
            final_score = (
                self.config.semantic_weight * semantic
                + self.config.recency_weight * recency
                + self.config.frequency_weight * frequency
                + self.config.importance_weight * importance
            )

            # A superseded/consolidated entry is kept for auditability but
            # demoted past anything still current.
            if entry.superseded_by:
                final_score *= 0.35

            # Generate retrieval reason
            reason_parts = []
            if semantic > 0.7:
                reason_parts.append(f"High semantic match ({semantic:.2f})")
            elif semantic > 0.5:
                reason_parts.append(f"Moderate semantic match ({semantic:.2f})")
            if recency > 0.8:
                reason_parts.append("Recent conversation")
            if frequency > 0.5:
                reason_parts.append("Frequently referenced")
            if importance > 0.8:
                reason_parts.append("High importance")
            if entry.superseded_by:
                reason_parts.append("Superseded by a newer entry")
            reason = "; ".join(reason_parts) if reason_parts else "Relevant memory"

            scored_results.append(
                MemorySearchResult(
                    memory=entry,
                    semantic_score=semantic,
                    recency_score=recency,
                    frequency_score=frequency,
                    importance_score=importance,
                    final_score=final_score,
                    retrieval_reason=reason,
                )
            )

        # Sort by final score and truncate
        scored_results.sort(key=lambda r: r.final_score, reverse=True)
        results = scored_results[:top_k]

        # One batched metadata-only write for access counts, not one upsert
        # per hit (which additionally clobbered embeddings -- see
        # MemoryStore.bump_access).
        now_iso = datetime.now(timezone.utc).isoformat() + "Z"
        bumped = []
        for result in results:
            result.memory.access_count += 1
            result.memory.last_accessed = now_iso
            bumped.append(result.memory)
        self.store.bump_access(bumped)

        mem_logger.info(
            "Memory search for '%s': %d results (top score: %.3f)",
            query[:50],
            len(results),
            results[0].final_score if results else 0.0,
        )

        return results

    def get_relevant_context(
        self,
        query: str,
        top_k: int = 3,
        session_id: Optional[str] = None,
    ) -> str:
        """Get a formatted context string from relevant memories for prompt injection.

        This is the main integration point with the RAG generator — the
        returned string is prepended to the retrieval context.

        Prefer :meth:`format_context` when you already hold search results:
        calling this in addition to ``search_memory`` runs the whole search
        twice (two embeddings, two vector queries) *and* double-increments
        every hit's ``access_count``, which skews the frequency term of the
        ranking against reality.
        """
        return self.format_context(
            self.search_memory(query, top_k=top_k, session_id=session_id),
            session_id=session_id,
        )

    def format_context(
        self,
        results: List[MemorySearchResult],
        session_id: Optional[str] = None,
    ) -> str:
        """Render already-retrieved memories into prompt-injectable text."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            mem = result.memory
            if mem.kind == "summary":
                header_type = "Earlier consolidation (summary)"
                body = f"Summary: {mem.answer}"
            elif mem.kind == "fact":
                is_current = session_id and mem.session_id == session_id
                src = "current session" if is_current else "an earlier session"
                verified = mem.metadata.get("verified")
                tag = "verified" if verified else ""
                header_type = f"Remembered fact ({src}{', ' + tag if tag else ''})"
                body = f"{mem.answer}\n(originally asked: {mem.question})"
            else:
                is_current = session_id and mem.session_id == session_id
                header_type = "Current Session Interaction" if is_current else "Previous conversation"
                body = f"Q: {mem.question}\nA: {mem.answer}"
            context_parts.append(
                f"--- {header_type} {i} (Relevance: {result.final_score:.2f}) ---\n{body}\n"
            )
        return "\n".join(context_parts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query (with the BGE query instruction prefix, which
        BGE is trained to see on the QUERY side only)."""
        from configs.models import BGE_QUERY_INSTRUCTION
        return self.embed_model.get_text_embedding(BGE_QUERY_INSTRUCTION + text)

    def embed_text(self, text: str) -> List[float]:
        """Embed text being STORED. Document-side: no query prefix -- the
        previous shared path prefixed stored memories too, depositing them
        slightly off the document manifold they are searched against."""
        return self.embed_model.get_text_embedding(text)
