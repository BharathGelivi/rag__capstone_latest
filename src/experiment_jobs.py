"""Durable experiment job state, independent of browser sessions."""
import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4


class JobStore:
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL, payload TEXT NOT NULL, checkpoint TEXT NOT NULL, updated TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def create(self, payload):
        job_id = str(uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES (?, 'queued', 'queued', ?, '{}', ?)",
                       (job_id, json.dumps(payload, allow_nan=False), self.now()))
        return self.get(job_id)

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT id,status,stage,payload,checkpoint,updated FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        value = dict(zip(('id', 'status', 'stage', 'payload', 'checkpoint', 'updated'), row))
        for key in ('payload', 'checkpoint'):
            value[key] = json.loads(value[key])
        return value

    def list(self, limit=100):
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("Limit must be between 1 and 1000")
        with self.connect() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM jobs ORDER BY updated DESC LIMIT ?", (limit,))]
        return [self.get(job_id) for job_id in ids]

    def claim_next(self, concurrency=1):
        if not isinstance(concurrency, int) or not 1 <= concurrency <= 8:
            raise ValueError("Concurrency must be between 1 and 8")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT count(*) FROM jobs WHERE status='running'").fetchone()[0]
            if active >= concurrency:
                return None
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY updated LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',stage='starting',updated=? WHERE id=?", (self.now(), row[0]))
        return self.get(row[0])

    def recover_interrupted(self):
        """Call only when the sole worker starts, after acquiring its process lock."""
        with self.connect() as db:
            return db.execute("UPDATE jobs SET status='interrupted',stage='worker stopped',updated=? WHERE status='running'", (self.now(),)).rowcount

    def transition(self, job_id, expected, status, stage, checkpoint=None):
        """Compare-and-swap prevents cancellation from being overwritten by workers."""
        with self.connect() as db:
            if checkpoint is None:
                result = db.execute("UPDATE jobs SET status=?,stage=?,updated=? WHERE id=? AND status=?",
                                    (status, stage, self.now(), job_id, expected))
            else:
                result = db.execute("UPDATE jobs SET status=?,stage=?,checkpoint=?,updated=? WHERE id=? AND status=?",
                                    (status, stage, json.dumps(checkpoint, allow_nan=False), self.now(), job_id, expected))
            return result.rowcount == 1
