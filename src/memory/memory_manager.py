"""
Memory Manager Module.

Top-level orchestrator for the memory system. Coordinates between
MemoryStore, MemoryRetriever, MemorySummarizer, and SessionManager.
Provides a unified API for the UI and pipeline integration.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    MemorySearchResult,
    MemorySummary,
    MemoryType,
    SessionInfo,
)
from src.memory.memory_store import MemoryStore
from src.memory.memory_retriever import MemoryRetriever
from src.memory.memory_summarizer import MemorySummarizer
from src.memory.session_manager import SessionManager
from configs.thresholds import CONTRADICTION_THRESHOLD as _CONTRADICTION_THRESHOLD
from src.memory.memory_utils import (
    compute_text_hash,
    generate_memory_id,
    get_timestamp,
    save_json,
    setup_memory_logger,
)

logger = logging.getLogger(__name__)
mem_logger = setup_memory_logger()


class MemoryManager:
    """Unified interface for the X-RAG memory system.

    Coordinates all memory components and provides:
    - Memory save/search/delete operations
    - Session management
    - Automatic summarization
    - Integration with the RAG pipeline

    Usage:
        manager = MemoryManager()
        manager.initialize()
        session_id = manager.ensure_session()
        manager.save_interaction(question, answer, session_id=session_id)
        relevant_memories = manager.search_memory(new_question)
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or self._load_config()
        self.store = MemoryStore(self.config)
        self.retriever = MemoryRetriever(self.store, self.config)
        self.summarizer = MemorySummarizer(self.config)
        self.session_manager = SessionManager(self.store, self.config)
        self._initialized = False

    @staticmethod
    def _load_config() -> MemoryConfig:
        """Load configuration from YAML file if available.

        Resolved against the project root, not the process's current working
        directory: a relative ``"configs/memory.yaml"`` silently misses (and
        ``from_yaml`` silently falls back to defaults on any exception,
        including FileNotFoundError) whenever the server is launched from a
        directory other than the repo root -- which in turn falls back to
        the *default* ``persistence_directory``, itself relative, so the
        memory database silently moves depending on how the process was
        started. Same failure mode either way; anchoring the yaml lookup to
        the project root closes it here, and MemoryStore anchors
        ``persistence_directory`` the same way as a second line of defense.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(project_root, "configs", "memory.yaml")
        if os.path.exists(config_path):
            return MemoryConfig.from_yaml(config_path)
        return MemoryConfig()

    def initialize(self) -> None:
        """Initialize all memory components."""
        if self._initialized:
            return
        self.store.initialize()
        self._initialized = True
        mem_logger.info("MemoryManager fully initialized.")

    def ensure_session(self) -> str:
        """Ensure a session exists, returning the current session ID."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.ensure_session()

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def save_interaction(
        self,
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        retrieved_chunk_ids: Optional[List[str]] = None,
        claim_ids: Optional[List[str]] = None,
        importance_score: float = 1.0,
        scope: str = "chat",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Save a Q&A interaction to memory.

        This is the primary method called after each RAG pipeline run.
        Creates a MemoryEntry, computes its embedding, saves to the store,
        and triggers summarization if needed.

        Args:
            question:           The user's question.
            answer:             The generated answer.
            session_id:         Session to save to (uses current if None).
            trace_id:           Associated RAG trace ID.
            retrieved_chunk_ids: IDs of chunks used in retrieval.
            claim_ids:          IDs of extracted claims.
            importance_score:   User or system importance rating (0-1+).
            tags:               Optional tags for categorization.
            metadata:           Additional metadata.

        Returns:
            The saved MemoryEntry.
        """
        if not self._initialized:
            self.initialize()

        if not self.config.enable_memory:
            # Return a dummy entry if memory is disabled
            return MemoryEntry(
                memory_id=generate_memory_id(),
                session_id=session_id or "",
                question=question,
                answer=answer,
            )

        sid = session_id or self.session_manager.ensure_session()

        # --- Dedup on write -------------------------------------------------
        # Exact duplicate (same Q+A text already stored): refresh timestamps
        # and bump access instead of inserting a second copy.
        content_hash = compute_text_hash(f"{question}\n{answer}")
        existing = self.store.find_by_content_hash(content_hash)
        if existing is not None:
            existing.access_count += 1
            existing.timestamp = get_timestamp()
            self.store.bump_access([existing])
            return existing

        # Compute embedding for the Q&A pair (document space: no BGE query
        # prefix -- queries are prefixed, stored documents are not).
        try:
            combined_text = f"{question} {answer}"
            embedding = self.retriever.embed_text(combined_text)
        except Exception as e:
            logger.warning("Failed to compute memory embedding: %s", e)
            embedding = None

        # Near-duplicate: same session, very similar content -> merge into the
        # existing entry (bump + refresh) rather than growing the store with
        # near-identical turns that later split the recall budget.
        if embedding is not None:
            dup = self._find_near_duplicate(embedding, session_id=sid, kind="turn")
            if dup is not None:
                dup.access_count += 1
                dup.timestamp = get_timestamp()
                dup.importance_score = max(dup.importance_score, importance_score)
                self.store.save_memory(dup, embedding)
                return dup

        # Create memory entry
        entry = MemoryEntry(
            memory_id=generate_memory_id(),
            session_id=sid,
            question=question,
            answer=answer,
            timestamp=get_timestamp(),
            memory_type=MemoryType.LONG_TERM,
            kind="turn",
            scope=scope,
            content_hash=content_hash,
            retrieved_chunk_ids=retrieved_chunk_ids or [],
            claim_ids=claim_ids or [],
            trace_id=trace_id,
            importance_score=importance_score,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Save to store
        self.store.save_memory(entry, embedding)

        # Update session metadata
        self.session_manager.update_session_activity(sid)
        self.session_manager.increment_memory_count(sid)
        if trace_id:
            self.session_manager.increment_trace_count(sid)

        # Check if summarization is needed
        self._check_summarization(sid)

        mem_logger.info(
            "Saved interaction: %s (session: %s, trace: %s)",
            entry.memory_id, sid, trace_id,
        )

        return entry

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        """Search memories by semantic similarity.

        Args:
            query:      Search query text.
            top_k:      Maximum results to return.
            session_id: Optional session filter.

        Returns:
            Ranked list of MemorySearchResult objects.
        """
        if not self._initialized:
            self.initialize()

        if not self.config.enable_memory:
            return []

        return self.retriever.search_memory(
            query=query, top_k=top_k, session_id=session_id
        )

    def get_memory_context(
        self,
        query: str,
        top_k: int = 3,
        session_id: Optional[str] = None,
    ) -> str:
        """Get formatted memory context for prompt injection.

        Returns a string ready to be prepended to the RAG context.
        """
        if not self._initialized:
            self.initialize()

        if not self.config.enable_memory:
            return ""

        return self.retriever.get_relevant_context(
            query=query, top_k=top_k, session_id=session_id
        )

    def format_memory_context(
        self,
        results: List[MemorySearchResult],
        session_id: Optional[str] = None,
    ) -> str:
        """Format memories you already searched for, without re-running the search.

        Use this together with :meth:`search_memory` instead of calling
        :meth:`get_memory_context` as a second step — see
        ``MemoryRetriever.get_relevant_context`` for why the double call is
        both slower and wrong.
        """
        if not self.config.enable_memory:
            return ""
        return self.retriever.format_context(results, session_id=session_id)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory entry."""
        if not self._initialized:
            self.initialize()
        return self.store.delete_memory(memory_id)

    def clear_session_memory(self, session_id: Optional[str] = None) -> int:
        """Clear all memories for a session."""
        if not self._initialized:
            self.initialize()
        sid = session_id or self.session_manager.current_session_id
        if sid:
            return self.store.clear_session_memory(sid)
        return 0

    def clear_all_memory(self) -> int:
        """Clear all memories across all sessions."""
        if not self._initialized:
            self.initialize()
        return self.store.clear_all_memory()

    def health(self) -> Dict[str, Any]:
        """Read-only diagnostics used by the API and memory UI."""
        if not self._initialized:
            self.initialize()
        return self.store.health()

    # ------------------------------------------------------------------
    # Session operations (delegated to SessionManager)
    # ------------------------------------------------------------------

    def create_session(self, title: str = "New Session") -> SessionInfo:
        """Create a new session."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.create_session(title)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.delete_session(session_id)

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """Rename a session."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.rename_session(session_id, new_title)

    def switch_session(self, session_id: str) -> Optional[SessionInfo]:
        """Switch to a different session."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.switch_session(session_id)

    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.list_sessions()

    def get_current_session(self) -> Optional[SessionInfo]:
        """Get the current active session."""
        return self.session_manager.get_current_session()

    def get_session_memories(
        self, session_id: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get all memories for a session."""
        if not self._initialized:
            self.initialize()
        sid = session_id or self.session_manager.current_session_id
        if sid:
            return self.store.get_session_memories(sid)
        return []

    def export_session(
        self, session_id: str, format: str = "json"
    ) -> Dict[str, Any]:
        """Export a session with its data."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.export_session(session_id, format)

    def import_session(self, data: Dict[str, Any]) -> SessionInfo:
        """Import a session from exported data."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.import_session(data)

    # ------------------------------------------------------------------
    # Statistics and info
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self._initialized:
            self.initialize()

        sessions = self.store.list_sessions()
        entries = self.store.get_all_memories(limit=2000)
        fact_count = sum(1 for e in entries if e.kind == "fact")
        summary_entries = sum(1 for e in entries if e.kind == "summary")
        superseded = sum(1 for e in entries if e.superseded_by)
        return {
            "total_memories": self.store.memory_count(),
            "turn_entries": len(entries) - fact_count - summary_entries,
            "fact_count": fact_count,
            "superseded_count": superseded,
            "total_sessions": len(sessions),
            "summaries_count": len(self.store.get_summaries()),
            "memory_enabled": self.config.enable_memory,
            "current_session": self.session_manager.current_session_id,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_near_duplicate(
        self,
        embedding: List[float],
        session_id: Optional[str],
        kind: Optional[str] = None,
        similarity_floor: Optional[float] = None,
    ) -> Optional[MemoryEntry]:
        """Return the nearest same-kind entry if it is at/above the dedup
        similarity floor. Small local query -- one extra Chroma call per save
        is cheap insurance against a store full of near-copies splitting the
        recall budget later."""
        floor = similarity_floor if similarity_floor is not None else self.config.dedup_similarity
        try:
            raw = self.store.search_by_embedding(
                query_embedding=embedding, top_k=3, session_id=session_id
            )
        except Exception:
            return None
        for entry, dist in self.store.entries_from_query(raw):
            if entry.superseded_by:
                continue
            if kind and entry.kind != kind:
                continue
            sim = max(0.0, 1.0 - (dist / 2.0))
            if sim >= floor:
                return entry
        return None

    def chat_recall(
        self,
        query: str,
        session_id: str,
        top_k: int = 3,
    ) -> List[MemorySearchResult]:
        """The chat paths' recall entry point.

        Applies the ``cross_session`` policy in one place: with it on (the
        default, matching the README's promise), recall spans all sessions;
        the *session_id* is only used afterwards for "current session"
        labelling in format_context. With it off, recall is restricted to the
        active session. (Previously both chat UIs hard-passed the session
        filter, so cross-session recall was silently off despite the docs.)
        """
        if not self.config.enable_memory:
            return []
        session_filter = None if self.config.cross_session else session_id
        return self.search_memory(query, top_k=top_k, session_id=session_filter)

    def get_memories(self, memory_ids: List[str]) -> List[MemoryEntry]:
        """Batch fetch by ids (used to rehydrate a trace's memory evidence)."""
        out = []
        for mid in memory_ids:
            entry = self.store.get_memory(mid)
            if entry is not None:
                out.append(entry)
        return out

    def save_verified_facts(
        self,
        question: str,
        facts: List[str],
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        claim_ids: Optional[List[str]] = None,
        verifier=None,
    ) -> List[MemoryEntry]:
        """Persist compact *fact units* extracted from a turn.

        Facts come from claims the NLI verifier judged SUPPORTED/
        PARTIALLY_SUPPORTED against retrieved evidence -- memory inherits the
        project's diagnostic standard, so a recalled fact is itself a verified
        statement with trace/claim provenance, not a blur of a whole answer.

        High-similarity facts (>= config.supersede_similarity) get an NLI
        contradiction check: if the new fact contradicts the old one, the old
        is marked superseded (demoted in recall, kept for audit). Consistent
        near-dups merge into the old entry (bump), never duplicated.
        """
        if not self._initialized:
            self.initialize()
        if not self.config.enable_memory or not self.config.fact_memory:
            return []
        sid = session_id or self.session_manager.ensure_session()

        saved: List[MemoryEntry] = []
        for fact_text, claim_id in zip(facts, claim_ids or [None] * len(facts)):
            text = fact_text.strip()
            if len(text) < 12:  # not a durable statement
                continue
            try:
                embedding = self.retriever.embed_text(text)
            except Exception as e:
                logger.warning("Failed to embed fact: %s", e)
                embedding = None

            dup = None
            if embedding is not None:
                dup = self._find_near_duplicate(
                    embedding, session_id=None, kind="fact",
                    similarity_floor=self.config.supersede_similarity,
                )

            if dup is not None:
                if verifier is not None:
                    try:
                        scores = verifier.run_nli(premise=text, hypothesis=dup.answer)
                        if scores.get("contradiction", 0.0) >= _CONTRADICTION_THRESHOLD:
                            # The new fact displaces the old; keep the old one
                            # on record but out of default recall.
                            new_entry = self._insert_fact(
                                text, question, embedding, sid, trace_id, claim_id)
                            dup.superseded_by = new_entry.memory_id
                            self.store.bump_access([dup])
                            saved.append(new_entry)
                            continue
                    except Exception as e:
                        logger.warning("Supersede check failed, merging instead: %s", e)
                dup.access_count += 1
                dup.timestamp = get_timestamp()
                self.store.save_memory(dup, embedding)
                continue

            saved.append(self._insert_fact(
                text, question, embedding, sid, trace_id, claim_id))
        return saved

    def _insert_fact(
        self,
        text: str,
        question: str,
        embedding: Optional[List[float]],
        session_id: str,
        trace_id: Optional[str],
        claim_id: Optional[str],
    ) -> MemoryEntry:
        entry = MemoryEntry(
            memory_id=generate_memory_id(),
            session_id=session_id,
            question=question[:500],
            answer=text,
            timestamp=get_timestamp(),
            memory_type=MemoryType.LONG_TERM,
            kind="fact",
            trace_id=trace_id,
            claim_ids=[claim_id] if claim_id else [],
            importance_score=1.0,
            content_hash=compute_text_hash(text),
            metadata={"verified": True},
        )
        self.store.save_memory(entry, embedding)
        return entry

    def _check_summarization(self, session_id: str) -> None:
        """Check if summarization is needed and trigger if so.

        The summary is also persisted INTO the vector collection as a
        ``kind="summary"`` entry, with its source entries marked superseded
        (demoted, not deleted) -- previously summaries were written to a JSON
        file that retrieval never read, so consolidation compressed nothing.
        """
        if not self.config.auto_summary:
            return

        memories = self.store.get_session_memories(session_id)
        # Only consolidated *turn* records count toward the trigger; facts and
        # summaries are the outputs, not the input material.
        active_turns = [m for m in memories
                        if m.kind == "turn" and not m.superseded_by]
        if self.summarizer.should_summarize(len(active_turns)):
            # Summarize the oldest half of active turns
            half = len(active_turns) // 2
            to_summarize = active_turns[:half]

            summary = self.summarizer.summarize(to_summarize, session_id)
            if summary:
                self.store.save_summary(summary)

                # Persist as a searchable entry so retrieval can reach it.
                try:
                    summary_entry = MemoryEntry(
                        memory_id=summary.summary_id,
                        session_id=session_id,
                        question=f"Consolidated summary of {summary.question_count} older interactions",
                        answer=summary.summary_text,
                        timestamp=summary.created_at,
                        memory_type=MemoryType.SUMMARY,
                        kind="summary",
                        importance_score=1.0,
                        metadata={"source_memory_ids": summary.source_memory_ids,
                                  "key_topics": summary.key_topics},
                    )
                    try:
                        embedding = self.retriever.embed_text(
                            summary.summary_text + " " + " ".join(summary.key_topics))
                    except Exception:
                        embedding = None
                    self.store.save_memory(summary_entry, embedding)
                except Exception as e:
                    logger.warning("Summary indexing failed (summary still kept as JSON): %s", e)

                # Demote the consolidated sources: they remain on disk and in
                # the graph, but stop competing with the summary in recall.
                for entry in to_summarize:
                    entry.superseded_by = summary.summary_id
                try:
                    self.store.bump_access(to_summarize)
                except Exception:
                    pass

                mem_logger.info(
                    "Auto-summary triggered for session %s: %d memories summarized",
                    session_id, len(to_summarize),
                )
