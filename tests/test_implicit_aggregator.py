"""
Tests for ImplicitAggregator — uses in-memory stores, no Postgres needed.
"""
import pytest
import numpy as np

from backend.app.tracker.event_store import EventStore
from backend.app.tracker.feedback_store import FeedbackInput, FeedbackStore
from backend.app.model.implicit_aggregator import (
    ImplicitAggregator,
    IMPLICIT_FEATURE_NAMES,
    EXPLICIT_FEATURE_NAMES,
    ALL_ENGAGEMENT_FEATURE_NAMES,
    N_TOTAL,
)


@pytest.fixture
def stores():
    ev = EventStore(db_url="sqlite:///:memory:")
    fb = FeedbackStore(db_url="sqlite:///:memory:")
    return ev, fb


def test_feature_name_counts():
    assert len(IMPLICIT_FEATURE_NAMES) == 15
    assert len(EXPLICIT_FEATURE_NAMES) == 5
    assert len(ALL_ENGAGEMENT_FEATURE_NAMES) == N_TOTAL == 20


def test_cold_start_returns_zeros(stores):
    ev, fb = stores
    agg = ImplicitAggregator(ev, fb)
    sig = agg.aggregate_learner("UNKNOWN_LEARNER")

    assert sig.is_cold_start()
    assert sig.to_vector() == [0.0] * 20


def test_vector_length(stores):
    ev, fb = stores
    agg = ImplicitAggregator(ev, fb)
    sig = agg.aggregate_learner("L001")
    assert len(sig.to_vector()) == 20


def test_session_count_computed(stores):
    ev, fb = stores
    ev.record_batch([
        {"learner_id": "L001", "session_id": "S1", "event_type": "page_view", "page": "dashboard"},
        {"learner_id": "L001", "session_id": "S1", "event_type": "click"},
        {"learner_id": "L001", "session_id": "S2", "event_type": "page_view", "page": "explain"},
    ])
    agg = ImplicitAggregator(ev, fb)
    sig = agg.aggregate_learner("L001")

    assert sig.session_count == 2.0
    assert not sig.is_cold_start()


def test_whatif_count_and_unique_features(stores):
    ev, fb = stores
    ev.record_batch([
        {"learner_id": "L001", "event_type": "whatif_slider", "event_target": "quiz_avg_score"},
        {"learner_id": "L001", "event_type": "whatif_slider", "event_target": "quiz_avg_score"},
        {"learner_id": "L001", "event_type": "whatif_slider", "event_target": "login_frequency_weekly"},
    ])
    agg = ImplicitAggregator(ev, fb)
    sig = agg.aggregate_learner("L001")

    assert sig.whatif_interaction_count == 3.0
    assert sig.unique_features_explored == 2.0   # 2 distinct features


def test_explicit_signals_from_feedback(stores):
    ev, fb = stores
    for rating in [4, 5, 3]:
        fb.record(FeedbackInput(learner_id="L001", rating=rating, followed_recommendation=True))
    fb.record(FeedbackInput(learner_id="L001", followed_recommendation=False))

    agg = ImplicitAggregator(ev, fb)
    sig = agg.aggregate_learner("L001")

    assert abs(sig.avg_explanation_rating - 4.0) < 0.01
    assert sig.correction_count == 0.0


def test_to_matrix(stores):
    ev, fb = stores
    ev.record_batch([
        {"learner_id": "L001", "event_type": "page_view", "session_id": "S1"},
        {"learner_id": "L002", "event_type": "page_view", "session_id": "S2"},
    ])
    agg = ImplicitAggregator(ev, fb)
    signals = agg.aggregate_all()
    matrix, feature_names, learner_ids = agg.to_matrix(signals)

    assert matrix.shape == (len(learner_ids), 20)
    assert feature_names == ALL_ENGAGEMENT_FEATURE_NAMES
    assert set(learner_ids) == {"L001", "L002"}
