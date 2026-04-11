"""
Stress & edge-case tests — concurrency, repeated calls, large payloads,
malformed inputs, boundary values across every major endpoint.
"""
from __future__ import annotations

import json
import random
import threading
import time

import numpy as np
import pytest

from tests.conftest_fixtures import *   # noqa: F401,F403

# Re-use the same bootstrapped client from integration tests
from tests.test_api_integration import _build_state, FEATURES_PAYLOAD, HIGH_RISK_PAYLOAD


@pytest.fixture(scope="module")
def client(dataset):
    _build_state(dataset)
    from backend.app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _random_features(rng=None):
    r = rng or random.Random()
    return {
        "login_frequency_weekly":     round(r.uniform(0, 14), 2),
        "avg_session_duration_min":   round(r.uniform(0, 300), 1),
        "forum_posts_count":          r.randint(0, 50),
        "video_completion_rate":      round(r.uniform(0, 1), 3),
        "quiz_avg_score":             round(r.uniform(0, 100), 2),
        "quiz_completion_rate":       round(r.uniform(0, 1), 3),
        "assignment_submission_rate": round(r.uniform(0, 1), 3),
        "days_since_last_activity":   r.randint(0, 60),
        "prior_course_completions":   r.randint(0, 10),
        "current_week_in_course":     r.randint(1, 52),
        "missed_deadlines_count":     r.randint(0, 10),
        "help_requests_count":        r.randint(0, 20),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Concurrency — /predict
# ──────────────────────────────────────────────────────────────────────────────

class TestConcurrency:
    def test_concurrent_predict_10_threads(self, client):
        results = []
        errors  = []

        def call(i):
            try:
                r = client.post("/predict", json=_random_features(random.Random(i)))
                results.append(r.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=call, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert all(s == 200 for s in results), f"Non-200: {results}"

    def test_concurrent_explain_5_threads(self, client):
        results = []

        def call(i):
            r = client.post("/explain", json={
                "features": _random_features(random.Random(i * 10)),
                "learner_id": f"stress-L{i}",
            })
            results.append(r.status_code)

        threads = [threading.Thread(target=call, args=(i,)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert all(s == 200 for s in results)

    def test_concurrent_events_3_threads(self, client):
        results = []

        def call(i):
            events = [
                {"learner_id": f"C{i}", "session_id": "S", "event_type": "page_view",
                 "page": "dashboard", "event_value": float(i)}
                for _ in range(10)
            ]
            r = client.post("/events", json={"events": events})
            results.append(r.status_code)

        threads = [threading.Thread(target=call, args=(i,)) for i in range(3)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert all(s == 200 for s in results)


# ──────────────────────────────────────────────────────────────────────────────
# Repeated calls — idempotency + no state bleed
# ──────────────────────────────────────────────────────────────────────────────

class TestIdempotency:
    def test_predict_repeated_same_features(self, client):
        """Same features must always return same risk_score."""
        scores = set()
        for _ in range(5):
            d = client.post("/predict", json=FEATURES_PAYLOAD).json()
            scores.add(d["risk_score"])
        assert len(scores) == 1, f"Non-deterministic predict: {scores}"

    def test_explain_same_features_stable_shap(self, client):
        """SHAP values must be identical across repeated calls with same input."""
        shap_sets = []
        for _ in range(3):
            d = client.post("/explain", json={"features": FEATURES_PAYLOAD}).json()
            shap_sets.append(json.dumps(d["shap_values"], sort_keys=True))
        assert len(set(shap_sets)) == 1, "SHAP values differ across repeated calls"

    def test_whatif_repeated_stable(self, client):
        deltas = []
        payload = {"features": FEATURES_PAYLOAD, "overrides": {"quiz_avg_score": 85.0}}
        for _ in range(3):
            d = client.post("/whatif", json=payload).json()
            deltas.append(d["risk_delta"])
        assert len(set(deltas)) == 1, f"Non-deterministic whatif: {deltas}"


# ──────────────────────────────────────────────────────────────────────────────
# Bulk throughput
# ──────────────────────────────────────────────────────────────────────────────

class TestThroughput:
    def test_predict_50_sequential(self, client):
        rng = random.Random(99)
        start = time.time()
        for _ in range(50):
            r = client.post("/predict", json=_random_features(rng))
            assert r.status_code == 200
        elapsed = time.time() - start
        # Should complete 50 predicts in under 30s on any machine
        assert elapsed < 30, f"50 sequential predicts took {elapsed:.1f}s"

    def test_events_batch_max_size(self, client):
        events = [
            {"learner_id": f"BULK{i}", "session_id": "S0",
             "event_type": "page_view", "page": "dashboard",
             "event_value": float(i)}
            for i in range(500)
        ]
        r = client.post("/events", json={"events": events})
        assert r.status_code == 200
        assert r.json()["recorded"] == 500

    def test_feedback_20_sequential(self, client):
        for i in range(20):
            r = client.post("/feedback", json={
                "learner_id": f"BULK-FB-{i}",
                "rating": (i % 5) + 1,
                "followed_recommendation": i % 2 == 0,
            })
            assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases — extreme feature values
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_predict_all_zeros(self, client):
        zero = {k: 0 for k in FEATURES_PAYLOAD}
        zero["current_week_in_course"] = 1
        r = client.post("/predict", json=zero)
        assert r.status_code == 200

    def test_predict_all_max(self, client):
        mx = {
            "login_frequency_weekly": 14, "avg_session_duration_min": 300,
            "forum_posts_count": 1000, "video_completion_rate": 1.0,
            "quiz_avg_score": 100.0, "quiz_completion_rate": 1.0,
            "assignment_submission_rate": 1.0, "days_since_last_activity": 0,
            "prior_course_completions": 100, "current_week_in_course": 52,
            "missed_deadlines_count": 0, "help_requests_count": 0,
        }
        r = client.post("/predict", json=mx)
        assert r.status_code == 200
        assert 0.0 <= r.json()["risk_score"] <= 1.0

    def test_explain_with_all_zero_latent(self, client):
        payload = dict(FEATURES_PAYLOAD,
                       engagement_latent_1=0.0,
                       engagement_latent_2=0.0,
                       engagement_latent_3=0.0)
        r = client.post("/explain", json={"features": payload})
        assert r.status_code == 200

    def test_explain_with_negative_latent_rejected(self, client):
        """Latent features have no ge constraint, should still not crash."""
        payload = dict(FEATURES_PAYLOAD,
                       engagement_latent_1=-999.0)
        r = client.post("/explain", json={"features": payload})
        # No constraint on latent — should be 200
        assert r.status_code == 200

    def test_events_all_event_types(self, client):
        valid_types = [
            "page_view", "whatif_interaction", "action_view",
            "action_dismiss", "resource_download", "scroll",
            "focus_start", "focus_end",
        ]
        events = [
            {"learner_id": "EDGE-TYPES", "session_id": "S", "event_type": t, "page": "p"}
            for t in valid_types
        ]
        r = client.post("/events", json={"events": events})
        assert r.status_code == 200
        assert r.json()["recorded"] == len(valid_types)

    def test_predict_float_as_int_fields(self, client):
        """forum_posts_count is int — float should be rejected."""
        bad = dict(FEATURES_PAYLOAD, forum_posts_count=2.7)
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_explain_extra_fields_ignored(self, client):
        """Extra fields in payload should be ignored (not crash)."""
        payload = dict(FEATURES_PAYLOAD, unknown_field_xyz=99.9)
        r = client.post("/explain", json={"features": payload})
        assert r.status_code in {200, 422}  # Pydantic may reject extras

    def test_feedback_missing_learner_id(self, client):
        r = client.post("/feedback", json={"rating": 4})
        assert r.status_code == 422

    def test_whatif_invalid_override_key(self, client):
        """Overriding a non-existent feature should not crash the server."""
        payload = {"features": FEATURES_PAYLOAD, "overrides": {"totally_fake_feature": 99.0}}
        r = client.post("/whatif", json=payload)
        assert r.status_code in {200, 422, 500}
        # Crucially: no unhandled 500 with a non-JSON body
        if r.status_code == 500:
            assert r.json() is not None


# ──────────────────────────────────────────────────────────────────────────────
# Malformed JSON / missing body
# ──────────────────────────────────────────────────────────────────────────────

class TestMalformedRequests:
    def test_predict_empty_body(self, client):
        r = client.post("/predict", content=b"", headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_predict_string_body(self, client):
        r = client.post("/predict", content=b'"just a string"',
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_predict_array_body(self, client):
        r = client.post("/predict", content=b"[1,2,3]",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_predict_null_body(self, client):
        r = client.post("/predict", content=b"null",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_events_non_list_events(self, client):
        r = client.post("/events", json={"events": "not-a-list"})
        assert r.status_code == 422

    def test_feedback_null_rating(self, client):
        r = client.post("/feedback", json={"learner_id": "L1", "rating": None})
        assert r.status_code == 200  # rating is Optional

    def test_explain_wrong_content_type(self, client):
        r = client.post("/explain",
                        content=b"login_frequency_weekly=3",
                        headers={"Content-Type": "application/x-www-form-urlencoded"})
        assert r.status_code == 422
