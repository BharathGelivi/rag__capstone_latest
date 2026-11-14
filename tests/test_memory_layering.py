"""
Unit tests for the memory layer's 2026 redesign: dedup-on-write, semantic
gating, cross-session recall policy, fact units with supersession, and
searchable summaries.

All tests use a deterministic hash-based embedder and a stub NLI verifier --
no models, no network, no GPU.
"""

import hashlib
import shutil
import tempfile
import unittest

from src.memory.memory_manager import MemoryManager
from src.memory.memory_models import MemoryConfig


class FakeEmbedder:
    """Deterministic bag-of-words embedding (32 dims) -- similar texts share
    most slots, so cosine similarity is stable inside one process."""

    DIM = 32

    def get_text_embedding(self, text: str):
        v = [0.0] * self.DIM
        for tok in text.lower().split():
            slot = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % self.DIM
            v[slot] += 1.0
        return v


class StubVerifier:
    """Always-contradicting NLI verdict for supersession tests."""

    def __init__(self, contradiction: float):
        self._scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": contradiction}

    def run_nli(self, premise: str, hypothesis: str):
        return dict(self._scores)


def make_manager(**overrides) -> MemoryManager:
    tmpdir = tempfile.mkdtemp()
    config = MemoryConfig(
        persistence_directory=tmpdir,
        collection_name="test_layering",
        similarity_threshold=0.4,
        **overrides,
    )
    mm = MemoryManager(config)
    mm.initialize()
    mm.retriever._embed_model = FakeEmbedder()
    return mm


class MemoryLayeringTestCase(unittest.TestCase):
    def setUp(self):
        self.mm = make_manager()
        self.addCleanup(
            lambda: shutil.rmtree(self.mm.config.persistence_directory, ignore_errors=True)
        )
        self.mm.ensure_session()


class TestDedupOnWrite(MemoryLayeringTestCase):
    def test_exact_duplicate_is_not_inserted_twice(self):
        sid = self.mm.ensure_session()
        first = self.mm.save_interaction("What is theft?", "Theft is a cognizable offence.", session_id=sid)
        second = self.mm.save_interaction("What is theft?", "Theft is a cognizable offence.", session_id=sid)
        self.assertEqual(first.memory_id, second.memory_id)
        self.assertEqual(self.mm.store.memory_count(), 1)
        self.assertEqual(self.mm.store.get_memory(first.memory_id).access_count, 1)

    def test_normalized_duplicate_collapses(self):
        """Case/whitespace differences share one content hash."""
        sid = self.mm.ensure_session()
        self.mm.save_interaction("Q", "The  Fine  Is  500.", session_id=sid)
        dup = self.mm.save_interaction("Q", "the fine is 500.", session_id=sid)
        self.assertEqual(self.mm.store.memory_count(), 1)
        self.assertGreaterEqual(dup.access_count, 1)


class TestSemanticFloor(MemoryLayeringTestCase):
    def test_irrelevant_but_fresh_memory_is_not_recalled(self):
        """A recent, important, frequently-accessed memory must still clear
        the semantic floor -- the blended score must never let an irrelevant
        memory into the prompt."""
        self.mm.config = MemoryConfig(
            persistence_directory=self.mm.config.persistence_directory,
            collection_name=self.mm.config.collection_name,
            similarity_threshold=0.9,
        )
        self.mm.retriever.config = self.mm.config
        sid = self.mm.ensure_session()
        self.mm.save_interaction(
            "What is the punishment for murder?",
            "Under BNS section 103, murder is punishable with death or life imprisonment.",
            session_id=sid,
            importance_score=1.0,
        )
        # Semantically unrelated query in a 32-dim hash embedding still shares
        # slots by chance; assert only the weak gate applies (nothing passes).
        results = self.mm.search_memory("quantum chromodynamics lattice gauge", top_k=3)
        for r in results:
            self.assertGreaterEqual(r.semantic_score, 0.9)

    def test_semantically_close_memory_passes(self):
        sid = self.mm.ensure_session()
        self.mm.save_interaction(
            "What is the punishment for murder?",
            "Death or life imprisonment under BNS 103.",
            session_id=sid,
        )
        results = self.mm.search_memory("punishment for murder section 103", top_k=3)
        self.assertTrue(results, "a clearly related memory must be recalled")


class TestCrossSessionRecall(MemoryLayeringTestCase):
    def test_cross_session_default_true_recalls_across_sessions(self):
        self.mm.config = MemoryConfig(
            persistence_directory=self.mm.config.persistence_directory,
            collection_name=self.mm.config.collection_name,
            cross_session=True,
        )
        self.mm.retriever.config = self.mm.config
        self.mm.manager_config_note = None
        s1 = self.mm.ensure_session()
        self.mm.save_interaction("murder punishment", "Death or life imprisonment.", session_id=s1)
        # New session; recall from session s2 must still see s1's memory.
        s2 = self.mm.create_session("other session")
        results = self.mm.chat_recall("murder punishment section 103", session_id=s2)
        self.assertTrue(any(r.memory.session_id == s1 for r in results))

    def test_cross_session_false_scopes_to_session(self):
        self.mm.config = MemoryConfig(
            persistence_directory=self.mm.config.persistence_directory,
            collection_name=self.mm.config.collection_name,
            cross_session=False,
        )
        self.mm.retriever.config = self.mm.config
        s1 = self.mm.ensure_session()
        self.mm.save_interaction("murder punishment", "Death or life imprisonment.", session_id=s1)
        s2 = self.mm.create_session("other session")
        results = self.mm.chat_recall("murder punishment section 103", session_id=s2)
        self.assertTrue(all(r.memory.session_id == s2 for r in results))


class TestFactUnits(MemoryLayeringTestCase):
    def test_verified_facts_are_stored_and_recallable(self):
        sid = self.mm.ensure_session()
        saved = self.mm.save_verified_facts(
            question="punishment for murder?",
            facts=["BNS 103 punishes murder with death or life imprisonment."],
            session_id=sid,
            trace_id="TRACE_1",
            claim_ids=["c1"],
        )
        self.assertEqual(len(saved), 1)
        entry = saved[0]
        self.assertEqual(entry.kind, "fact")
        self.assertEqual(entry.trace_id, "TRACE_1")
        self.assertTrue(self.mm.store.get_memory(entry.memory_id).metadata.get("verified"))

    def test_contradicting_fact_supersedes_the_old_one(self):
        self.mm.config = MemoryConfig(
            persistence_directory=self.mm.config.persistence_directory,
            collection_name=self.mm.config.collection_name,
            supersede_similarity=0.5,  # low, so the near-dup check fires in tests
        )
        self.mm.retriever.config = self.mm.config
        sid = self.mm.ensure_session()
        self.mm.save_verified_facts(
            question="fine?", facts=["The fine is 500 rupees."], session_id=sid)
        saved = self.mm.save_verified_facts(
            question="fine?", facts=["The fine is 5000 rupees per offence."],
            session_id=sid, verifier=StubVerifier(contradiction=0.99))
        self.assertEqual(len(saved), 1)
        entries = self.mm.store.get_all_memories(limit=50)
        facts = [e for e in entries if e.kind == "fact"]
        self.assertEqual(len(facts), 2)
        old = next(f for f in facts if f.memory_id != saved[0].memory_id)
        self.assertEqual(old.superseded_by, saved[0].memory_id)

    def test_consistent_near_duplicate_fact_merges(self):
        self.mm.config = MemoryConfig(
            persistence_directory=self.mm.config.persistence_directory,
            collection_name=self.mm.config.collection_name,
            supersede_similarity=0.5,
        )
        self.mm.retriever.config = self.mm.config
        sid = self.mm.ensure_session()
        text = "The fine set out in section 103 is five hundred rupees."
        self.mm.save_verified_facts(question="fine?", facts=[text], session_id=sid)
        again = self.mm.save_verified_facts(
            question="fine?", facts=["The fine set out in section 103 is five hundred rupees. "],
            session_id=sid, verifier=StubVerifier(contradiction=0.0))
        self.assertEqual(again, [])  # merged, not duplicated
        facts = [e for e in self.mm.store.get_all_memories(limit=50) if e.kind == "fact"]
        self.assertEqual(len(facts), 1)


class TestSummaryIsSearchable(MemoryLayeringTestCase):
    def test_consolidation_produces_a_recallable_summary(self):
        self.mm.config = MemoryConfig(
            persistence_directory=self.mm.config.persistence_directory,
            collection_name=self.mm.config.collection_name,
            auto_summary=True,
            summarization_threshold=2,
        )
        # Should_summarize reads the manager's config passed at construction;
        # update the summarizer's view too (it holds its own reference).
        self.mm.summarizer = type(self.mm.summarizer)(self.mm.config)
        sid = self.mm.ensure_session()
        self.mm.save_interaction("murder defined", "Murder is culpable homicide with intent.", session_id=sid)
        self.mm.save_interaction("murder punishment", "Death or life imprisonment for murder.", session_id=sid)

        all_entries = self.mm.store.get_all_memories(limit=50)
        summaries = [e for e in all_entries if e.kind == "summary"]
        self.assertTrue(summaries, "trigger must persist a searchable summary entry")
        superseded_turns = [e for e in all_entries if e.kind == "turn" and e.superseded_by]
        self.assertTrue(superseded_turns, "consolidated turns must be marked superseded")


if __name__ == "__main__":
    unittest.main()
