"""Validate persisted memory and optionally remove entries for deleted sessions.

The default mode is read-only. Use ``--prune-orphans`` only when the report has
been reviewed; it is deliberately explicit because deletion is irreversible.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.memory.memory_store import MemoryStore

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune-orphans", action="store_true")
    args = parser.parse_args()
    store = MemoryStore()
    sessions = {s.session_id for s in store.list_sessions()}
    memories = store.get_all_memories(limit=100000)
    orphaned = [m.memory_id for m in memories if m.session_id and sessions and m.session_id not in sessions]
    result = {"health": store.health(), "orphaned_memory_ids": orphaned,
              "orphan_check_skipped": not bool(sessions), "pruned": 0}
    if args.prune_orphans:
        result["pruned"] = sum(store.delete_memory(mid) for mid in orphaned)
    print(json.dumps(result, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
