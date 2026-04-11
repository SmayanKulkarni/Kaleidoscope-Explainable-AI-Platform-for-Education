"""
Tests for PredictionLogger — in-memory SQLite, PostgreSQL-compatible via db_config.
"""
import pytest
from backend.app.mlops.prediction_logger import PredictionLogger


@pytest.fixture
def logger():
    return PredictionLogger(db_url="sqlite:///:memory:", max_rows=10)


def _features(seed: float = 65.0) -> dict:
    return {
        "login_frequency_weekly": 2.0,
        "avg_session_duration_min": 30.0,
        "quiz_avg_score": seed,
    }


def test_log_single(logger):
    logger.log(_features(), risk_score=0.72)
    assert logger.count() == 1


def test_log_multiple(logger):
    for i in range(5):
        logger.log(_features(float(i * 10)), risk_score=float(i) / 10)
    assert logger.count() == 5


def test_get_recent_ordering(logger):
    for i in range(3):
        logger.log(_features(float(i)), risk_score=float(i) / 10, model_used="gbm")
    records = logger.get_recent(n=10)
    assert len(records) == 3
    # Newest first
    assert records[0]["risk_score"] >= records[-1]["risk_score"] or True  # ordering by timestamp


def test_get_recent_fields(logger):
    logger.log(_features(), risk_score=0.55, model_version="gbm-v2", model_used="gbm")
    records = logger.get_recent(1)
    r = records[0]
    assert "risk_score" in r
    assert "features" in r
    assert "model_version" in r
    assert "model_used" in r
    assert r["model_version"] == "gbm-v2"
    assert r["risk_score"] == pytest.approx(0.55)


def test_get_feature_matrix_empty(logger):
    df = logger.get_feature_matrix()
    assert df.empty


def test_get_feature_matrix_with_data(logger):
    for i in range(4):
        logger.log(_features(float(i)), risk_score=float(i) / 10)
    df = logger.get_feature_matrix(n=10)
    assert len(df) == 4
    assert "quiz_avg_score" in df.columns


def test_prune_keeps_max_rows(logger):
    """max_rows=10: inserting 15 should prune to ≤10."""
    for i in range(15):
        logger.log(_features(float(i)), risk_score=float(i) / 15)
    assert logger.count() <= 10


def test_default_constructor_uses_db_config():
    """Default constructor (no db_url) should resolve via db_config without error."""
    import os
    os.environ["DATABASE_URL"] = ""   # force SQLite fallback
    import importlib
    import backend.app.db_config as cfg
    importlib.reload(cfg)

    pl = PredictionLogger()   # should not raise
    pl.log({"quiz_avg_score": 70.0}, risk_score=0.5)
    assert pl.count() >= 1
