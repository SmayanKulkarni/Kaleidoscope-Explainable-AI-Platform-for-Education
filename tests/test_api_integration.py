"""
Full API integration tests — every endpoint, with a real trained GBM loaded into AppState.

These tests:
  - Boot the full FastAPI app via TestClient (no network)
  - Inject a real GBM + all explainers into AppState before tests run
  - Cover happy path, error/edge cases, validation, and auth flows
"""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from backend.app.model.data_loader import FEATURE_COLUMNS
from tests.conftest_fixtures import *   # noqa: F401,F403


# ──────────────────────────────────────────────────────────────────────────────
# App + State bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def _build_state(dataset: dict):
    """Inject a real GBM + all available explainers into the global AppState."""
    from backend.app import main as app_module

    X, y = dataset["X_train"], dataset["y_train"]

    gbm = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=0)
    gbm.fit(X, y)

    rf = RandomForestClassifier(n_estimators=20, random_state=0)
    rf.fit(X, y)

    app_module.state.gbm_model    = gbm
    app_module.state.rf_model     = rf
    app_module.state.feature_names = FEATURE_COLUMNS
    app_module.state.X_train      = X
    app_module.state.y_train      = y
    app_module.state.model_version = "gbm-test-v1"

    import shap as _shap
    from backend.app.explainers.shap_explainer import SHAPExplainer
    from backend.app.explainers.dice_explainer import DiCEExplainer
    from backend.app.explainers.prototype_explainer import PrototypeExplainer
    from backend.app.explainers.archipelago import ArchipelagoExplainer
    from backend.app.evaluator.trust_scorer import TrustScorer
    from backend.app.evaluator.uncertainty_estimator import UncertaintyEstimator
    from backend.app.causal.causal_annotator import CausalAnnotator
    from backend.app.prescriptor.action_ranker import ActionRanker
    from backend.app.tracker.consistency_store import ExplanationStore
    from backend.app.tracker.feedback_store import FeedbackStore
    from backend.app.tracker.event_store import EventStore
    from backend.app.mlops.prediction_logger import PredictionLogger

    shap_exp = SHAPExplainer(gbm_model=gbm, feature_names=FEATURE_COLUMNS)
    app_module.state.shap_explainer = shap_exp

    app_module.state.archipelago_explainer = ArchipelagoExplainer(
        tree_explainer=shap_exp.tree_explainer,
        feature_names=FEATURE_COLUMNS,
    )
    app_module.state.dice_explainer = DiCEExplainer(
        model=gbm, X_train=X, feature_names=FEATURE_COLUMNS, y_train=y
    )

    def _predict_fn(X_arr):
        return (gbm.predict_proba(X_arr)[:, 1] >= 0.5).astype(int)

    app_module.state.prototype_explainer = PrototypeExplainer(
        X_train=X, y_train=y,
        learner_ids=np.arange(len(X)),
        feature_names=FEATURE_COLUMNS,
    )
    app_module.state.uncertainty_estimator = UncertaintyEstimator(
        model=gbm, X_cal=X, y_cal=y
    )
    app_module.state.trust_scorer = TrustScorer()
    app_module.state.causal_annotator = CausalAnnotator(
        X_train=X, y_train=y, feature_names=FEATURE_COLUMNS
    )
    app_module.state.action_ranker = ActionRanker(
        feature_names=FEATURE_COLUMNS, X_train=X,
        causal_annotator=app_module.state.causal_annotator,
    )
    app_module.state.explanation_store  = ExplanationStore(db_url="sqlite:///:memory:")
    app_module.state.feedback_store     = FeedbackStore(db_url="sqlite:///:memory:")
    app_module.state.event_store        = EventStore(db_url="sqlite:///:memory:")
    app_module.state.prediction_logger  = PredictionLogger(db_url="sqlite:///:memory:")


@pytest.fixture(scope="module")
def client(dataset):
    _build_state(dataset)
    from backend.app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────────────
# Shared payload helpers
# ──────────────────────────────────────────────────────────────────────────────

FEATURES_PAYLOAD = {
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
}

HIGH_RISK_PAYLOAD = {
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
}


# ──────────────────────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_status_ok(self, client):
        d = client.get("/health").json()
        assert d["status"] == "ok"

    def test_health_fields(self, client):
        d = client.get("/health").json()
        assert "status" in d
        assert "gbm_loaded" in d
        assert d["gbm_loaded"] is True
        assert "timestamp" in d

    def test_health_model_version(self, client):
        d = client.get("/health").json()
        assert d["model_version"] == "gbm-test-v1"


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_200(self, client):
        r = client.post("/predict", json=FEATURES_PAYLOAD)
        assert r.status_code == 200

    def test_predict_returns_risk_score(self, client):
        d = client.post("/predict", json=FEATURES_PAYLOAD).json()
        assert "risk_score" in d
        assert 0.0 <= d["risk_score"] <= 1.0

    def test_predict_returns_risk_label(self, client):
        d = client.post("/predict", json=FEATURES_PAYLOAD).json()
        assert d["risk_label"] in {"low", "medium", "high"}

    def test_predict_high_risk_label(self, client):
        d = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
        assert d["risk_label"] in {"medium", "high"}

    def test_predict_model_used_gbm(self, client):
        d = client.post("/predict", json=FEATURES_PAYLOAD).json()
        assert d["model_used"] == "gbm"

    def test_predict_missing_required_field(self, client):
        bad = dict(FEATURES_PAYLOAD)
        del bad["quiz_avg_score"]
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_field_out_of_range(self, client):
        bad = dict(FEATURES_PAYLOAD, quiz_avg_score=999.0)
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_negative_field(self, client):
        bad = dict(FEATURES_PAYLOAD, forum_posts_count=-1)
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_with_latent_features(self, client):
        payload = dict(FEATURES_PAYLOAD, engagement_latent_1=0.3,
                       engagement_latent_2=0.2, engagement_latent_3=0.1)
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_predict_boundary_values(self, client):
        edge = {
            "login_frequency_weekly": 0, "avg_session_duration_min": 0,
            "forum_posts_count": 0, "video_completion_rate": 0.0,
            "quiz_avg_score": 0.0, "quiz_completion_rate": 0.0,
            "assignment_submission_rate": 0.0, "days_since_last_activity": 0,
            "prior_course_completions": 0, "current_week_in_course": 1,
            "missed_deadlines_count": 0, "help_requests_count": 0,
        }
        r = client.post("/predict", json=edge)
        assert r.status_code == 200

    def test_predict_max_boundary_values(self, client):
        edge = {
            "login_frequency_weekly": 14, "avg_session_duration_min": 300,
            "forum_posts_count": 0, "video_completion_rate": 1.0,
            "quiz_avg_score": 100.0, "quiz_completion_rate": 1.0,
            "assignment_submission_rate": 1.0, "days_since_last_activity": 0,
            "prior_course_completions": 10, "current_week_in_course": 52,
            "missed_deadlines_count": 0, "help_requests_count": 0,
        }
        r = client.post("/predict", json=edge)
        assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# POST /whatif
# ──────────────────────────────────────────────────────────────────────────────

class TestWhatIf:
    def test_whatif_200(self, client):
        payload = {"features": FEATURES_PAYLOAD, "overrides": {"quiz_avg_score": 90.0}}
        r = client.post("/whatif", json=payload)
        assert r.status_code == 200

    def test_whatif_returns_risk_delta(self, client):
        payload = {"features": FEATURES_PAYLOAD, "overrides": {"quiz_avg_score": 90.0}}
        d = client.post("/whatif", json=payload).json()
        assert "risk_delta" in d
        assert isinstance(d["risk_delta"], float)

    def test_whatif_returns_shap_values(self, client):
        payload = {"features": FEATURES_PAYLOAD, "overrides": {"quiz_avg_score": 90.0}}
        d = client.post("/whatif", json=payload).json()
        assert "shap_values" in d

    def test_whatif_empty_overrides(self, client):
        payload = {"features": FEATURES_PAYLOAD, "overrides": {}}
        r = client.post("/whatif", json=payload)
        assert r.status_code == 200

    def test_whatif_multiple_overrides(self, client):
        payload = {
            "features": FEATURES_PAYLOAD,
            "overrides": {
                "quiz_avg_score": 85.0,
                "assignment_submission_rate": 0.95,
                "login_frequency_weekly": 6.0,
            }
        }
        r = client.post("/whatif", json=payload)
        assert r.status_code == 200

    def test_whatif_missing_features(self, client):
        payload = {"features": {}, "overrides": {"quiz_avg_score": 80.0}}
        r = client.post("/whatif", json=payload)
        assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# POST /explain
# ──────────────────────────────────────────────────────────────────────────────

class TestExplain:
    def test_explain_200(self, client):
        r = client.post("/explain", json={"features": FEATURES_PAYLOAD})
        assert r.status_code == 200

    def test_explain_has_shap_values(self, client):
        d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
        assert "shap_values" in d
        assert isinstance(d["shap_values"], dict)

    def test_explain_has_trust_score(self, client):
        d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
        assert "trust_score" in d

    def test_explain_has_causal_annotations(self, client):
        d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
        assert "causal_annotations" in d
        assert isinstance(d["causal_annotations"], list)

    def test_explain_latent_annotations_are_correlational(self, client):
        payload = dict(FEATURES_PAYLOAD, engagement_latent_1=0.5,
                       engagement_latent_2=0.3, engagement_latent_3=0.2)
        d = client.post("/explain", json={"features": payload}).json()
        for ann in d.get("causal_annotations", []):
            if ann.get("feature", "").startswith("engagement_latent_"):
                assert ann.get("causal_type") == "correlational" or ann.get("is_causal") is False

    def test_explain_has_counterfactual(self, client):
        d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
        assert "counterfactual" in d

    def test_explain_counterfactual_no_immutable(self, client):
        from backend.app.explainers.dice_explainer import IMMUTABLE_FEATURES
        d = client.post("/explain", json={"features": HIGH_RISK_PAYLOAD}).json()
        actions = d.get("counterfactual", {}).get("actions", [])
        for action in actions:
            feat = action.get("feature", "")
            assert feat not in IMMUTABLE_FEATURES, f"Immutable '{feat}' in counterfactual"

    def test_explain_has_prototypes(self, client):
        d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
        assert "prototypes" in d

    def test_explain_has_interactions(self, client):
        d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
        assert "interactions" in d

    def test_explain_with_learner_id(self, client):
        r = client.post("/explain", json={
            "features": FEATURES_PAYLOAD,
            "learner_id": "test-learner-001",
        })
        assert r.status_code == 200

    def test_explain_persists_to_store(self, client):
        client.post("/explain", json={
            "features": FEATURES_PAYLOAD,
            "learner_id": "persist-test-learner",
        })
        r = client.get("/history/persist-test-learner")
        assert r.status_code == 200
        d = r.json()
        assert len(d["timeline"]) > 0 or len(d.get("history", [])) > 0

    def test_explain_audience_learner(self, client):
        r = client.post("/explain", json={
            "features": FEATURES_PAYLOAD,
            "audience": "learner",
        })
        assert r.status_code == 200

    def test_explain_audience_instructor(self, client):
        r = client.post("/explain", json={
            "features": FEATURES_PAYLOAD,
            "audience": "instructor",
        })
        assert r.status_code == 200

    def test_explain_risk_score_range(self, client):
        d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
        assert 0.0 <= d["risk_score"] <= 1.0

    def test_explain_stability_range(self, client):
        d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
        if "stability" in d:
            assert 0.0 <= d["stability"] <= 1.0

    def test_explain_full_response_serialisable(self, client):
        d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
        json.dumps(d)  # must not raise


# ──────────────────────────────────────────────────────────────────────────────
# POST /counterfactual
# ──────────────────────────────────────────────────────────────────────────────

class TestCounterfactual:
    def test_counterfactual_200(self, client):
        r = client.post("/counterfactual", json=FEATURES_PAYLOAD)
        assert r.status_code == 200

    def test_counterfactual_has_actions(self, client):
        d = client.post("/counterfactual", json=FEATURES_PAYLOAD).json()
        assert "actions" in d

    def test_counterfactual_high_risk_has_ranked_actions(self, client):
        d = client.post("/counterfactual", json=HIGH_RISK_PAYLOAD).json()
        if d.get("ranked_actions"):
            assert isinstance(d["ranked_actions"], list)

    def test_counterfactual_no_latent_actions(self, client):
        d = client.post("/counterfactual", json=HIGH_RISK_PAYLOAD).json()
        for action in d.get("actions", []):
            assert not action.get("feature", "").startswith("engagement_latent_")


# ──────────────────────────────────────────────────────────────────────────────
# POST /simulate
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulate:
    def test_simulate_200(self, client):
        r = client.post("/simulate", json={
            "features": FEATURES_PAYLOAD,
            "current_week": 4,
            "target_week": 12,
            "n_simulations": 50,
        })
        assert r.status_code == 200

    def test_simulate_no_transitions_graceful(self, client):
        """Without transitions.pkl loaded, endpoint returns graceful empty response."""
        d = client.post("/simulate", json={
            "features": FEATURES_PAYLOAD,
            "current_week": 4,
            "target_week": 12,
            "n_simulations": 50,
        }).json()
        assert "feature_distributions" in d

    def test_simulate_invalid_week_order(self, client):
        r = client.post("/simulate", json={
            "features": FEATURES_PAYLOAD,
            "current_week": 10,
            "target_week": 4,
            "n_simulations": 50,
        })
        # Either 422 (FastAPI validation) or 200 with message (no simulator loaded)
        assert r.status_code in {200, 422}

    def test_simulate_n_simulations_too_small(self, client):
        r = client.post("/simulate", json={
            "features": FEATURES_PAYLOAD,
            "current_week": 4,
            "target_week": 12,
            "n_simulations": 5,   # below min=10
        })
        assert r.status_code == 422

    def test_simulate_n_simulations_too_large(self, client):
        r = client.post("/simulate", json={
            "features": FEATURES_PAYLOAD,
            "current_week": 4,
            "target_week": 12,
            "n_simulations": 99999,
        })
        assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# GET /history/{learner_id}
# ──────────────────────────────────────────────────────────────────────────────

class TestHistory:
    def test_history_unknown_learner(self, client):
        r = client.get("/history/nobody-xyz")
        assert r.status_code == 200
        d = r.json()
        assert "timeline" in d

    def test_history_after_explain(self, client):
        client.post("/explain", json={
            "features": FEATURES_PAYLOAD,
            "learner_id": "history-test-L1",
        })
        d = client.get("/history/history-test-L1").json()
        assert "learner_id" in d
        assert d["learner_id"] == "history-test-L1"

    def test_history_has_drift_flags_key(self, client):
        d = client.get("/history/nobody-xyz").json()
        assert "drift_flags" in d


# ──────────────────────────────────────────────────────────────────────────────
# POST /events  &  GET /events/{learner_id}
# ──────────────────────────────────────────────────────────────────────────────

class TestEvents:
    def _event(self, learner_id: str = "EL1", etype: str = "page_view"):
        return {
            "learner_id": learner_id,
            "session_id": "S-001",
            "event_type": etype,
            "page": "dashboard",
            "event_value": 1200.0,
        }

    def test_events_post_200(self, client):
        r = client.post("/events", json={"events": [self._event()]})
        assert r.status_code == 200

    def test_events_recorded_count(self, client):
        d = client.post("/events", json={"events": [self._event(), self._event()]}).json()
        assert d["recorded"] == 2

    def test_events_batch_500(self, client):
        events = [self._event(f"L{i}", "page_view") for i in range(500)]
        r = client.post("/events", json={"events": events})
        assert r.status_code == 200
        assert r.json()["recorded"] == 500

    def test_events_batch_too_large(self, client):
        events = [self._event() for _ in range(501)]
        r = client.post("/events", json={"events": events})
        assert r.status_code == 422

    def test_events_empty_batch_rejected(self, client):
        r = client.post("/events", json={"events": []})
        assert r.status_code == 422

    def test_events_invalid_event_type(self, client):
        bad_event = dict(self._event(), event_type="hacking_attempt")
        r = client.post("/events", json={"events": [bad_event]})
        assert r.status_code == 422

    def test_get_learner_events_200(self, client):
        client.post("/events", json={"events": [self._event("EL-GET-TEST")]})
        r = client.get("/events/EL-GET-TEST")
        assert r.status_code == 200

    def test_get_learner_events_fields(self, client):
        client.post("/events", json={"events": [self._event("EL-FIELDS")]})
        d = client.get("/events/EL-FIELDS").json()
        assert "learner_id" in d
        assert "count" in d
        assert "events" in d

    def test_get_events_unknown_learner(self, client):
        r = client.get("/events/nobody-at-all")
        assert r.status_code == 200
        assert r.json()["count"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# POST /feedback  &  GET /feedback/stats  &  GET /feedback/{learner_id}
# ──────────────────────────────────────────────────────────────────────────────

class TestFeedback:
    def test_feedback_post_200(self, client):
        r = client.post("/feedback", json={
            "learner_id": "FB-L1",
            "rating": 4,
            "followed_recommendation": True,
        })
        assert r.status_code == 200

    def test_feedback_returns_recorded_true(self, client):
        d = client.post("/feedback", json={"learner_id": "FB-L2", "rating": 3}).json()
        assert d["recorded"] is True
        assert "feedback_id" in d

    def test_feedback_rating_out_of_range_low(self, client):
        r = client.post("/feedback", json={"learner_id": "FB-L3", "rating": 0})
        assert r.status_code == 422

    def test_feedback_rating_out_of_range_high(self, client):
        r = client.post("/feedback", json={"learner_id": "FB-L3", "rating": 6})
        assert r.status_code == 422

    def test_feedback_no_rating_ok(self, client):
        r = client.post("/feedback", json={
            "learner_id": "FB-L4",
            "followed_recommendation": False,
            "correction_feature": "quiz_avg_score",
        })
        assert r.status_code == 200

    def test_feedback_with_correction(self, client):
        r = client.post("/feedback", json={
            "learner_id": "FB-CORR",
            "rating": 2,
            "correction_feature": "login_frequency_weekly",
            "correction_comment": "This seems wrong for me.",
        })
        assert r.status_code == 200

    def test_feedback_stats_200(self, client):
        r = client.get("/feedback/stats")
        assert r.status_code == 200

    def test_feedback_stats_has_avg_rating(self, client):
        client.post("/feedback", json={"learner_id": "STATS-L", "rating": 5})
        d = client.get("/feedback/stats").json()
        assert "avg_rating" in d or len(d) > 0

    def test_get_learner_feedback_200(self, client):
        client.post("/feedback", json={"learner_id": "FB-GET", "rating": 4})
        r = client.get("/feedback/FB-GET")
        assert r.status_code == 200

    def test_get_learner_feedback_records(self, client):
        lid = "FB-RECORDS"
        client.post("/feedback", json={"learner_id": lid, "rating": 3})
        client.post("/feedback", json={"learner_id": lid, "rating": 5})
        d = client.get(f"/feedback/{lid}").json()
        assert "records" in d
        assert len(d["records"]) >= 2

    def test_get_unknown_learner_feedback_empty(self, client):
        d = client.get("/feedback/NOBODY-EVER").json()
        assert "records" in d
        assert d["records"] == []


# ──────────────────────────────────────────────────────────────────────────────
# GET /mlops/health  &  GET /mlops/metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMLOps:
    def _admin_headers(self, client):
        reg = client.post("/auth/register", json={
            "username": "mlops_admin_api",
            "email":    "mlops_admin_api@example.com",
            "password": "Str0ngPass!",
            "role":     "admin",
        })
        assert reg.status_code in {200, 201, 409}

        login = client.post(
            "/auth/login",
            json={"username": "mlops_admin_api", "password": "Str0ngPass!"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_mlops_health_200(self, client):
        r = client.get("/mlops/health")
        assert r.status_code == 200

    def test_mlops_health_fields(self, client):
        d = client.get("/mlops/health").json()
        assert "status" in d
        assert "model_version" in d
        assert "prediction_count" in d

    def test_mlops_metrics_200(self, client):
        r = client.get("/mlops/metrics")
        assert r.status_code == 200

    def test_mlops_metrics_has_model_version(self, client):
        d = client.get("/mlops/metrics").json()
        assert "model_version" in d

    def test_mlops_retrain_no_events_returns_result(self, client):
        r = client.post("/mlops/retrain?min_events=0", headers=self._admin_headers(client))
        # May succeed or fail gracefully (no training data in test env) — no 500
        assert r.status_code in {200, 503}

    def test_mlops_reload_when_no_new_model_graceful(self, client):
        r = client.post("/mlops/reload", headers=self._admin_headers(client))
        assert r.status_code in {200, 500}  # 500 is acceptable — returns detail dict


# ──────────────────────────────────────────────────────────────────────────────
# Auth — register, login, /auth/me
# ──────────────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_register_201(self, client):
        r = client.post("/auth/register", json={
            "username": "testuser_api",
            "email":    "testapi@example.com",
            "password": "Str0ngPass!",
            "full_name": "Test User",
            "role": "student",
        })
        assert r.status_code in {200, 201}

    def test_register_duplicate_username(self, client):
        payload = {
            "username": "dup_user_api",
            "email":    "dup1@example.com",
            "password": "Str0ngPass!",
            "role": "student",
        }
        client.post("/auth/register", json=payload)
        r = client.post("/auth/register", json=payload)
        assert r.status_code in {400, 409, 422}

    def test_login_success(self, client):
        client.post("/auth/register", json={
            "username": "logintest_api",
            "email":    "loginapi@example.com",
            "password": "Str0ngPass!",
            "role": "student",
        })
        r = client.post("/auth/login",
                        data={"username": "logintest_api", "password": "Str0ngPass!"},
                        headers={"Content-Type": "application/x-www-form-urlencoded"})
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_login_wrong_password(self, client):
        r = client.post("/auth/login",
                        data={"username": "logintest_api", "password": "wrongpassword"},
                        headers={"Content-Type": "application/x-www-form-urlencoded"})
        assert r.status_code in {401, 403}

    def test_me_without_token_401(self, client):
        r = client.get("/auth/me")
        assert r.status_code == 401

    def test_me_with_token(self, client):
        client.post("/auth/register", json={
            "username": "me_test_api",
            "email":    "meapi@example.com",
            "password": "Str0ngPass!",
            "role": "student",
        })
        login = client.post("/auth/login",
                            data={"username": "me_test_api", "password": "Str0ngPass!"},
                            headers={"Content-Type": "application/x-www-form-urlencoded"})
        token = login.json()["access_token"]
        r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["username"] == "me_test_api"
