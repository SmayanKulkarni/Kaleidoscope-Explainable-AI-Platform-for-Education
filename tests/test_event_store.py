"""
Tests for EventStore — in-memory SQLite, no Postgres needed.
"""
import pytest
from backend.app.tracker.event_store import EventStore


@pytest.fixture
def store():
    return EventStore(db_url="sqlite:///:memory:")


def test_record_batch_empty(store):
    assert store.record_batch([]) == 0


def test_record_batch_single(store):
    n = store.record_batch([{
        "learner_id":   "L001",
        "session_id":   "S001",
        "event_type":   "page_view",
        "event_target": None,
        "event_value":  3500.0,
        "page":         "dashboard",
    }])
    assert n == 1


def test_record_batch_multiple(store):
    events = [
        {"learner_id": "L001", "event_type": "click",         "event_target": "action_card"},
        {"learner_id": "L001", "event_type": "whatif_slider", "event_target": "quiz_avg_score", "event_value": 75.0},
        {"learner_id": "L002", "event_type": "page_view",     "page": "explain"},
    ]
    assert store.record_batch(events) == 3


def test_get_learner_events(store):
    store.record_batch([
        {"learner_id": "L001", "event_type": "page_view", "page": "dashboard"},
        {"learner_id": "L001", "event_type": "click",     "event_target": "shap_bar"},
        {"learner_id": "L002", "event_type": "page_view", "page": "explain"},
    ])
    events = store.get_learner_events("L001")
    assert len(events) == 2
    assert all(e["learner_id"] == "L001" for e in events)


def test_get_all_learner_ids(store):
    store.record_batch([
        {"learner_id": "L001", "event_type": "page_view"},
        {"learner_id": "L002", "event_type": "page_view"},
        {"learner_id": "L001", "event_type": "click"},
    ])
    ids = store.get_all_learner_ids()
    assert set(ids) == {"L001", "L002"}


def test_count(store):
    assert store.count() == 0
    store.record_batch([
        {"learner_id": "L001", "event_type": "page_view"},
        {"learner_id": "L001", "event_type": "click"},
    ])
    assert store.count() == 2
