"""
Unit tests for the Memory Store.

Tests long-term persistence, session management, kind-aware metadata,
dedup anchors, and metadata-only updates.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    MemorySummary,
    MemoryType,
    SessionInfo,
)
from src.memory.memory_store import MemoryStore
from src.memory.memory_utils import compute_text_hash


class TestMemoryStoreLongTerm(unittest.TestCase):
    """Tests for the long-term (ChromaDB) persistence layer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = MemoryConfig(
            persistence_directory=self.tmpdir,
            collection_name="test_memory",
        )
        self.store = MemoryStore(self.config)
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_get_memory(self):
        entry = MemoryEntry(
            memory_id="mem_test",
            session_id="s1",
            question="What is Python?",
            answer="A programming language.",
        )
        self.store.save_memory(entry)
        retrieved = self.store.get_memory("mem_test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.question, "What is Python?")

    def test_delete_memory(self):
        entry = MemoryEntry(
            memory_id="mem_del",
            session_id="s1",
            question="Q",
            answer="A",
        )
        self.store.save_memory(entry)
        self.assertTrue(self.store.delete_memory("mem_del"))
        self.assertIsNone(self.store.get_memory("mem_del"))

    def test_memory_count(self):
        self.assertEqual(self.store.memory_count(), 0)
        self.store.save_memory(
            MemoryEntry(memory_id="m1", session_id="s1", question="Q1", answer="A1")
        )
        self.assertEqual(self.store.memory_count(), 1)

    def test_health_reports_counts_and_clean_status(self):
        self.store.save_memory(MemoryEntry(memory_id="m_health", session_id="s1", question="Q", answer="A"))
        health = self.store.health()
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["memory_count"], 1)
        self.assertEqual(health["missing_session_count"], 0)

    def test_clear_all_memory(self):
        for i in range(3):
            self.store.save_memory(
                MemoryEntry(
                    memory_id=f"m{i}", session_id="s1", question=f"Q{i}", answer=f"A{i}"
                )
            )
        self.assertEqual(self.store.memory_count(), 3)
        cleared = self.store.clear_all_memory()
        self.assertEqual(cleared, 3)
        self.assertEqual(self.store.memory_count(), 0)

    def test_get_session_memories(self):
        for i in range(3):
            self.store.save_memory(
                MemoryEntry(
                    memory_id=f"m_s1_{i}", session_id="s1", question=f"Q{i}", answer=f"A{i}"
                )
            )
        self.store.save_memory(
            MemoryEntry(memory_id="m_s2_0", session_id="s2", question="Q", answer="A")
        )
        s1_memories = self.store.get_session_memories("s1")
        self.assertEqual(len(s1_memories), 3)

    def test_clear_session_memory(self):
        self.store.save_memory(
            MemoryEntry(memory_id="m_a", session_id="s1", question="Q1", answer="A1")
        )
        self.store.save_memory(
            MemoryEntry(memory_id="m_b", session_id="s2", question="Q2", answer="A2")
        )
        cleared = self.store.clear_session_memory("s1")
        self.assertEqual(cleared, 1)
        self.assertEqual(self.store.memory_count(), 1)

    def test_question_answer_stored_as_fields_not_blob(self):
        """An answer containing the legacy 'A: ' separator must round-trip intact."""
        tricky = "Line one.\nA: this is part of the answer itself."
        self.store.save_memory(MemoryEntry(
            memory_id="m_qa", session_id="s1", question="What gives?", answer=tricky,
        ))
        got = self.store.get_memory("m_qa")
        self.assertEqual(got.answer, tricky)
        self.assertEqual(got.question, "What gives?")

    def test_kind_scope_superseded_round_trip(self):
        self.store.save_memory(MemoryEntry(
            memory_id="m_f", session_id="s1", question="Q", answer="A",
            kind="fact", scope="chat", superseded_by=None,
            content_hash=compute_text_hash("Q\nA"),
        ))
        got = self.store.get_memory("m_f")
        self.assertEqual(got.kind, "fact")
        self.assertIsNone(got.superseded_by)
        self.assertEqual(got.content_hash, compute_text_hash("Q\nA"))

    def test_find_by_content_hash(self):
        h = compute_text_hash("hello world")
        self.store.save_memory(MemoryEntry(
            memory_id="m_hash", session_id="s1", question="q", answer="a",
            content_hash=h,
        ))
        self.assertIsNotNone(self.store.find_by_content_hash(h))
        self.assertIsNone(self.store.find_by_content_hash("nope"))

    def test_bump_access_does_not_clobber_embedding(self):
        """Metadata-only update: the stored vector must survive a bump."""
        vec = [1.0] + [0.0] * 767
        entry = MemoryEntry(memory_id="m_vec", session_id="s1", question="Q", answer="A")
        self.store.save_memory(entry, vec)
        entry.access_count = 7
        self.store.bump_access([entry])

        # If bump_access had gone through upsert-without-embeddings, Chroma
        # would have recomputed the vector with the null embedding function
        # and this query would match anything/nothing -- not m_vec best.
        hits = self.store.search_by_embedding(vec, top_k=1)
        self.assertEqual(hits["ids"][0][0], "m_vec")
        got = self.store.get_memory("m_vec")
        self.assertEqual(got.access_count, 7)

    def test_entries_from_query_is_not_n_plus_one(self):
        """Query results must convert to entries without get_memory calls."""
        entry = MemoryEntry(memory_id="m_np1", session_id="s1", question="Q", answer="A")
        self.store.save_memory(entry, [1.0] + [0.0] * 767)
        raw = self.store.search_by_embedding([1.0] + [0.0] * 767, top_k=1)
        with patch.object(self.store, "get_memory", side_effect=AssertionError("N+1!")):
            hits = self.store.entries_from_query(raw)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0].memory_id, "m_np1")


class TestMemoryStoreSession(unittest.TestCase):
    """Tests for session persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = MemoryConfig(
            persistence_directory=self.tmpdir,
            collection_name="test_sessions",
        )
        self.store = MemoryStore(self.config)
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_get_session(self):
        session = SessionInfo(session_id="s1", title="Test Session")
        self.store.save_session(session)
        retrieved = self.store.get_session("s1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Test Session")

    def test_list_sessions(self):
        self.store.save_session(SessionInfo(session_id="s1", title="Session 1"))
        self.store.save_session(SessionInfo(session_id="s2", title="Session 2"))
        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_delete_session(self):
        self.store.save_session(SessionInfo(session_id="s1", title="To Delete"))
        self.assertTrue(self.store.delete_session("s1"))
        self.assertIsNone(self.store.get_session("s1"))

    def test_failed_memory_deletion_keeps_session_visible_and_retryable(self):
        self.store.save_session(SessionInfo(session_id="s1", title="Keep until deleted"))
        entry = MemoryEntry(memory_id="retained", session_id="s1", question="Question", answer="Answer")
        self.store.save_memory(entry)
        with patch.object(self.store._collection, "delete", side_effect=OSError("Database unavailable")):
            with self.assertRaises(OSError):
                self.store.delete_session("s1")
        self.assertIsNotNone(self.store.get_session("s1"))
        self.assertIsNotNone(self.store.get_memory("retained"))
        self.assertTrue(self.store.delete_session("s1"))
        self.assertIsNone(self.store.get_memory("retained"))

    def test_failed_metadata_write_restores_session_for_retry(self):
        self.store.save_session(SessionInfo(session_id="s1", title="Retry"))
        with patch.object(self.store, "_persist_sessions", side_effect=OSError("Disk unavailable")):
            with self.assertRaises(OSError):
                self.store.delete_session("s1")
        self.assertIsNotNone(self.store.get_session("s1"))
        self.assertTrue(self.store.delete_session("s1"))

    def test_rename_session(self):
        self.store.save_session(SessionInfo(session_id="s1", title="Old Name"))
        self.assertTrue(self.store.rename_session("s1", "New Name"))
        self.assertEqual(self.store.get_session("s1").title, "New Name")

    def test_session_persistence(self):
        """Sessions should survive store re-initialization."""
        self.store.save_session(SessionInfo(session_id="s1", title="Persistent"))

        # Create a new store pointing to the same directory
        store2 = MemoryStore(self.config)
        store2.initialize()
        retrieved = store2.get_session("s1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Persistent")


class TestMemoryStoreSummary(unittest.TestCase):
    """Tests for summary persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = MemoryConfig(persistence_directory=self.tmpdir, collection_name="test_sum")
        self.store = MemoryStore(self.config)
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_get_summary(self):
        summary = MemorySummary(
            summary_id="sum_1",
            session_id="s1",
            summary_text="A summary",
            key_topics=["topic1", "topic2"],
        )
        self.store.save_summary(summary)
        summaries = self.store.get_summaries("s1")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].summary_text, "A summary")


if __name__ == "__main__":
    unittest.main()
