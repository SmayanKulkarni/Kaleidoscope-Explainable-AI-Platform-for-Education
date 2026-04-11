"""
Tests for MonteCarloSimulator — no transitions.pkl file needed (uses synthetic data).
"""
import numpy as np
import pytest

from backend.app.model.temporal_builder import MonteCarloSimulator
from backend.app.model.data_loader import FEATURE_COLUMNS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_transitions(n_rows: int = 200, seed: int = 0) -> dict:
    """Build a minimal synthetic transitions dict matching the real format."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    week_pairs = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
    rows = []
    for (fw, tw) in week_pairs:
        for _ in range(n_rows // len(week_pairs)):
            row = {feat: rng.normal(0, 0.1) for feat in FEATURE_COLUMNS}
            row["from_week"] = fw
            row["to_week"] = tw
            row["final_result"] = rng.integers(0, 2)
            rows.append(row)

    deltas = pd.DataFrame(rows)

    summary = {}
    for (fw, tw) in week_pairs:
        summary[(fw, tw)] = {
            feat: {"mean": 0.0, "std": 0.05, "q25": -0.05, "q75": 0.05}
            for feat in FEATURE_COLUMNS
        }

    return {"deltas": deltas, "summary": summary, "cutoff_weeks": [w for w, _ in week_pairs]}


def _current_features() -> dict:
    return {
        "login_frequency_weekly": 2.0,
        "avg_session_duration_min": 35.0,
        "forum_posts_count": 1,
        "video_completion_rate": 0.6,
        "quiz_avg_score": 65.0,
        "quiz_completion_rate": 0.7,
        "assignment_submission_rate": 0.8,
        "days_since_last_activity": 3,
        "prior_course_completions": 1,
        "current_week_in_course": 6,
        "missed_deadlines_count": 1,
        "help_requests_count": 2,
    }


@pytest.fixture
def sim():
    return MonteCarloSimulator(_make_transitions(), seed=42)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_simulate_returns_required_keys(sim):
    result = sim.simulate(_current_features(), current_week=6, target_week=12, n_simulations=100)
    assert "feature_distributions" in result
    assert "current_week" in result
    assert "target_week" in result
    assert "n_simulations" in result


def test_feature_distributions_has_all_features(sim):
    result = sim.simulate(_current_features(), current_week=6, target_week=12, n_simulations=100)
    fd = result["feature_distributions"]
    for feat in FEATURE_COLUMNS:
        assert feat in fd
        assert set(fd[feat].keys()) == {"mean", "std", "q10", "q50", "q90"}


def test_simulate_n_simulations_respected(sim):
    for n in [10, 50, 200]:
        result = sim.simulate(_current_features(), current_week=6, target_week=12, n_simulations=n)
        assert result["n_simulations"] == n


def test_simulate_current_target_week_in_result(sim):
    result = sim.simulate(_current_features(), current_week=4, target_week=10, n_simulations=50)
    assert result["current_week"] == 4
    assert result["target_week"] == 10


def test_simulate_with_model_adds_outcome_distribution(sim):
    """With a mock model, outcome_distribution should be present."""
    class MockModel:
        def predict_proba(self, X):
            n = X.shape[0]
            probs = np.full((n, 2), 0.5)
            probs[:, 1] = 0.4
            return probs

    result = sim.simulate(
        _current_features(), current_week=6, target_week=12,
        n_simulations=100, model=MockModel(),
    )
    od = result["outcome_distribution"]
    assert "dropout_prob_mean" in od
    assert "dropout_prob_std" in od
    assert "dropout_prob_q10" in od
    assert "dropout_prob_q50" in od
    assert "dropout_prob_q90" in od
    assert "dropout_rate" in od
    assert abs(od["dropout_prob_mean"] - 0.4) < 0.01


def test_simulate_without_model_no_outcome_distribution(sim):
    result = sim.simulate(_current_features(), current_week=6, target_week=12, n_simulations=50)
    assert "outcome_distribution" not in result


def test_simulate_invalid_weeks(sim):
    with pytest.raises(ValueError):
        sim.simulate(_current_features(), current_week=10, target_week=6, n_simulations=50)


def test_feature_clipping_respects_bounds(sim):
    """All simulated features should stay within physical bounds."""
    result = sim.simulate(_current_features(), current_week=2, target_week=12, n_simulations=200)
    fd = result["feature_distributions"]
    # Rates must stay 0-1
    for feat in ("video_completion_rate", "quiz_completion_rate", "assignment_submission_rate"):
        assert fd[feat]["q10"] >= 0.0
        assert fd[feat]["q90"] <= 1.0
    # Score must stay 0-100
    assert fd["quiz_avg_score"]["q10"] >= 0.0
    assert fd["quiz_avg_score"]["q90"] <= 100.0


def test_reproducibility(sim):
    """Same seed → same output."""
    sim_a = MonteCarloSimulator(_make_transitions(), seed=7)
    sim_b = MonteCarloSimulator(_make_transitions(), seed=7)
    r_a = sim_a.simulate(_current_features(), current_week=6, target_week=12, n_simulations=100)
    r_b = sim_b.simulate(_current_features(), current_week=6, target_week=12, n_simulations=100)
    assert r_a["feature_distributions"]["quiz_avg_score"]["mean"] == \
           r_b["feature_distributions"]["quiz_avg_score"]["mean"]
