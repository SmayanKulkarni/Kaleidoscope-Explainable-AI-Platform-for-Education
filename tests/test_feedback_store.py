"""
Tests for FeedbackStore — in-memory SQLite, no Postgres needed.
"""
import pytest
from backend.app.tracker.feedback_store import FeedbackInput, FeedbackStore


@pytest.fixture
def store():
    return FeedbackStore(db_url="sqlite:///:memory:")


def _make_input(**kwargs) -> FeedbackInput:
    defaults = {"learner_id": "L001"}
    defaults.update(kwargs)
    return FeedbackInput(**defaults)


def test_record_rating(store):
    fid = store.record(_make_input(rating=4))
    assert isinstance(fid, int)
    assert fid > 0


def test_record_invalid_rating(store):
    with pytest.raises(ValueError):
        store.record(_make_input(rating=6))


def test_get_learner_feedback(store):
    store.record(_make_input(rating=5, learner_id="L001"))
    store.record(_make_input(rating=3, learner_id="L001"))
    store.record(_make_input(rating=4, learner_id="L002"))

    records = store.get_learner_feedback("L001")
    assert len(records) == 2
    assert all("id" in r for r in records)     # learner_id is the filter, not in returned dict


def test_record_follow_and_correction(store):
    fid = store.record(_make_input(
        followed_recommendation=True,
        correction_feature="quiz_avg_score",
        correction_comment="My quiz scores are higher now",
    ))
    records = store.get_learner_feedback("L001")
    assert len(records) == 1
    assert records[0]["followed_recommendation"] is True
    assert records[0]["correction_feature"] == "quiz_avg_score"


def test_get_stats_no_data(store):
    stats = store.get_stats()
    assert stats.total_feedback == 0
    assert stats.avg_rating is None


def test_get_stats_with_data(store):
    store.record(_make_input(rating=5, followed_recommendation=True))
    store.record(_make_input(rating=3, followed_recommendation=False))
    store.record(_make_input(rating=4, followed_recommendation=True))

    stats = store.get_stats()
    assert stats.total_feedback == 3
    assert abs(stats.avg_rating - 4.0) < 0.01
    assert abs(stats.recommendation_follow_rate - 2/3) < 0.01
