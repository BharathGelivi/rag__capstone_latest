from src.experiment_jobs import JobStore


def test_jobs_survive_reopening_and_cancellation_wins(tmp_path):
    path = tmp_path / 'jobs.sqlite'
    store = JobStore(path)
    job = store.create({'experiment': 'example'})
    assert store.transition(job['id'], 'queued', 'running', 'retrieving', {'done': ['a']})
    reopened = JobStore(path)
    assert reopened.get(job['id'])['checkpoint'] == {'done': ['a']}
    assert reopened.transition(job['id'], 'running', 'cancelled', 'cancelled')
    assert not store.transition(job['id'], 'running', 'completed', 'completed')
    assert reopened.get(job['id'])['status'] == 'cancelled'


def test_claim_enforces_concurrency_and_restart_preserves_checkpoint(tmp_path):
    store = JobStore(tmp_path / 'jobs.sqlite')
    first = store.create({'experiment': 'a'})
    store.create({'experiment': 'b'})
    assert store.claim_next()['id'] == first['id']
    assert store.claim_next() is None
    assert store.transition(first['id'], 'running', 'running', 'verifying', {'completed': 1})
    assert store.recover_interrupted() == 1
    assert store.get(first['id'])['checkpoint'] == {'completed': 1}
    assert store.transition(first['id'], 'interrupted', 'queued', 'queued')
    assert len(store.list()) == 2
