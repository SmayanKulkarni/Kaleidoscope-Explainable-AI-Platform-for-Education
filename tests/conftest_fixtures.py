"""
Shared heavy fixtures — trained GBM/RF, X_train, feature_names.
Imported by other test modules via:  from tests.conftest_fixtures import gbm_bundle
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from backend.app.model.data_loader import FEATURE_COLUMNS

N_TRAIN = 200
N_TEST  = 50
N_FEATS = len(FEATURE_COLUMNS)   # 12 original features


def _make_dataset(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, N_FEATS)).astype(np.float32)
    # Scale a few to realistic ranges
    X[:, FEATURE_COLUMNS.index("quiz_avg_score")] *= 100
    X[:, FEATURE_COLUMNS.index("current_week_in_course")] = (
        rng.integers(1, 13, size=n).astype(np.float32)
    )
    y = (X[:, FEATURE_COLUMNS.index("quiz_avg_score")] < 50).astype(int)
    return X, y


@pytest.fixture(scope="session")
def dataset():
    X_train, y_train = _make_dataset(N_TRAIN, seed=1)
    X_test,  y_test  = _make_dataset(N_TEST,  seed=2)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "feature_names": FEATURE_COLUMNS,
    }


@pytest.fixture(scope="session")
def gbm_bundle(dataset):
    gbm = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=0)
    gbm.fit(dataset["X_train"], dataset["y_train"])
    return {"model": gbm, "raw": gbm, "feature_names": FEATURE_COLUMNS}


@pytest.fixture(scope="session")
def rf_bundle(dataset):
    rf = RandomForestClassifier(n_estimators=20, random_state=0)
    rf.fit(dataset["X_train"], dataset["y_train"])
    return {"model": rf, "feature_names": FEATURE_COLUMNS}


@pytest.fixture(scope="session")
def sample_features():
    """A realistic learner feature dict."""
    return {
        "login_frequency_weekly":     3.0,
        "avg_session_duration_min":   45.0,
        "forum_posts_count":          2,
        "video_completion_rate":      0.7,
        "quiz_avg_score":             62.0,
        "quiz_completion_rate":       0.75,
        "assignment_submission_rate": 0.8,
        "days_since_last_activity":   3,
        "prior_course_completions":   1,
        "current_week_in_course":     6,
        "missed_deadlines_count":     1,
        "help_requests_count":        2,
        "engagement_latent_1":        0.0,
        "engagement_latent_2":        0.0,
        "engagement_latent_3":        0.0,
    }


@pytest.fixture(scope="session")
def high_risk_features():
    return {
        "login_frequency_weekly":     0.5,
        "avg_session_duration_min":   5.0,
        "forum_posts_count":          0,
        "video_completion_rate":      0.1,
        "quiz_avg_score":             20.0,
        "quiz_completion_rate":       0.1,
        "assignment_submission_rate": 0.1,
        "days_since_last_activity":   20,
        "prior_course_completions":   0,
        "current_week_in_course":     8,
        "missed_deadlines_count":     5,
        "help_requests_count":        0,
        "engagement_latent_1":        0.0,
        "engagement_latent_2":        0.0,
        "engagement_latent_3":        0.0,
    }


@pytest.fixture(scope="session")
def low_risk_features():
    return {
        "login_frequency_weekly":     6.0,
        "avg_session_duration_min":   90.0,
        "forum_posts_count":          10,
        "video_completion_rate":      0.95,
        "quiz_avg_score":             88.0,
        "quiz_completion_rate":       0.95,
        "assignment_submission_rate": 0.98,
        "days_since_last_activity":   0,
        "prior_course_completions":   3,
        "current_week_in_course":     4,
        "missed_deadlines_count":     0,
        "help_requests_count":        1,
        "engagement_latent_1":        0.1,
        "engagement_latent_2":        0.2,
        "engagement_latent_3":        0.1,
    }
