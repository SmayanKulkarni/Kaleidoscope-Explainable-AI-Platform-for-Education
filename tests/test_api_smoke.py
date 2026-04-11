"""
API smoke tests — spin up the FastAPI app with TestClient (no real models needed).
Tests only the endpoints that work without loaded models (health, auth, events, feedback).
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


def test_mlops_health(client):
    r = client.get("/mlops/health")
    assert r.status_code == 200


# ── Auth endpoints ─────────────────────────────────────────────────────────────

def test_register_and_login(client):
    payload = {
        "username":  "test_student_ci",
        "password":  "Test1234!",
        "email":     "ci@test.com",
        "role":      "student",
        "full_name": "CI Test Student",
    }
    r = client.post("/auth/register", json=payload)
    # 200 = new registration, 400 = already exists (idempotent in CI)
    assert r.status_code in (200, 400)

    if r.status_code == 200:
        r2 = client.post("/auth/login", json={
            "username": "test_student_ci",
            "password": "Test1234!",
        })
        assert r2.status_code == 200
        assert "access_token" in r2.json()


# ── Events endpoint ────────────────────────────────────────────────────────────

def test_post_events_valid(client):
    payload = {"events": [
        {
            "learner_id":   "L_CI_001",
            "session_id":   "S_CI_001",
            "event_type":   "page_view",
            "page":         "dashboard",
            "event_value":  5000.0,
        },
        {
            "learner_id":   "L_CI_001",
            "session_id":   "S_CI_001",
            "event_type":   "whatif_slider",
            "event_target": "quiz_avg_score",
            "event_value":  75.0,
        },
    ]}
    r = client.post("/events", json=payload)
    assert r.status_code == 200
    assert r.json()["recorded"] == 2


def test_post_events_empty_batch(client):
    r = client.post("/events", json={"events": []})
    assert r.status_code == 422   # min_length=1 validation


def test_post_events_invalid_type(client):
    r = client.post("/events", json={"events": [{
        "learner_id": "L001",
        "event_type": "not_a_real_event_type",
    }]})
    assert r.status_code == 422


def test_get_learner_events(client):
    r = client.get("/events/L_CI_001")
    assert r.status_code == 200
    data = r.json()
    assert "learner_id" in data
    assert "events" in data


# ── Feedback endpoint ──────────────────────────────────────────────────────────

def test_post_feedback_valid(client):
    r = client.post("/feedback", json={
        "learner_id": "L_CI_001",
        "rating":      4,
        "followed_recommendation": True,
    })
    assert r.status_code == 200
    assert r.json()["recorded"] is True


def test_post_feedback_invalid_rating(client):
    r = client.post("/feedback", json={
        "learner_id": "L_CI_001",
        "rating":      10,
    })
    assert r.status_code == 422


def test_feedback_stats(client):
    r = client.get("/feedback/stats")
    assert r.status_code == 200


# ── Predict/Explain return 503 gracefully without models ──────────────────────

def test_predict_no_model_returns_503(client):
    r = client.post("/predict", json={"features": {
        "login_frequency_weekly": 2.0,
        "avg_session_duration_min": 30.0,
        "forum_posts_count": 1,
        "video_completion_rate": 0.6,
        "quiz_avg_score": 65.0,
        "quiz_completion_rate": 0.7,
        "assignment_submission_rate": 0.8,
        "days_since_last_activity": 3,
        "prior_course_completions": 1,
        "current_week_in_course": 4,
        "missed_deadlines_count": 1,
        "help_requests_count": 2,
    }})
    # Without model loaded, expect 503 (or 200 if model happens to be present in CI)
    assert r.status_code in (200, 503)
