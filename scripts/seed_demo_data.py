"""
Seed script for the XAI Learning Recommendation System.

Creates demo accounts (student / instructor / admin), enrolls the student,
generates 5 synthetic students via Groq LLM, then populates explanation
history, feedback records, and interaction events for all students.

Usage:
    python scripts/seed_demo_data.py --base-url http://localhost:8000

Env:
    GROQ_API_KEY  — optional; falls back to hardcoded synthetic profiles
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Seed demo data for XAI system")
parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
args = parser.parse_args()
BASE = args.base_url.rstrip("/")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Demo accounts
# ---------------------------------------------------------------------------
DEMO_ACCOUNTS = [
    {
        "username": "demo_student",
        "email": "demo_student@learnlens.com",
        "full_name": "Alex Rivera",
        "password": "DemoStudent2026!",
        "role": "student",
        "learner_id": "demo-learner-001",
        "course_id": "AAA",
        "module_presentation": "2014J",
    },
    {
        "username": "demo_instructor",
        "email": "demo_instructor@learnlens.com",
        "full_name": "Dr. Sarah Chen",
        "password": "DemoInstructor2026!",
        "role": "instructor",
        "department": "Computer Science",
    },
    {
        "username": "demo_admin",
        "email": "demo_admin@learnlens.com",
        "full_name": "System Administrator",
        "password": "DemoAdmin2026!",
        "role": "admin",
    },
]

# ---------------------------------------------------------------------------
# Hardcoded fallback synthetic profiles (used if GROQ_API_KEY is absent)
# ---------------------------------------------------------------------------
FALLBACK_STUDENTS = [
    {
        "username": "synth_student_1",
        "email": "s1@learnlens.com",
        "full_name": "Jordan Kim",
        "password": "Synth1234!",
        "role": "student",
        "learner_id": "synth-001",
        "course_id": "AAA",
        "features": {
            "login_frequency_weekly": 1.0,
            "avg_session_duration_min": 15.0,
            "forum_posts_count": 0,
            "video_completion_rate": 0.3,
            "quiz_avg_score": 38.0,
            "quiz_completion_rate": 0.4,
            "assignment_submission_rate": 0.35,
            "days_since_last_activity": 12,
            "prior_course_completions": 0,
            "current_week_in_course": 6,
            "missed_deadlines_count": 4,
            "help_requests_count": 0,
            "engagement_latent_1": 0.0,
            "engagement_latent_2": 0.0,
            "engagement_latent_3": 0.0,
        },
    },
    {
        "username": "synth_student_2",
        "email": "s2@learnlens.com",
        "full_name": "Morgan Patel",
        "password": "Synth1234!",
        "role": "student",
        "learner_id": "synth-002",
        "course_id": "AAA",
        "features": {
            "login_frequency_weekly": 7.0,
            "avg_session_duration_min": 80.0,
            "forum_posts_count": 5,
            "video_completion_rate": 0.9,
            "quiz_avg_score": 82.0,
            "quiz_completion_rate": 0.95,
            "assignment_submission_rate": 0.9,
            "days_since_last_activity": 1,
            "prior_course_completions": 3,
            "current_week_in_course": 8,
            "missed_deadlines_count": 0,
            "help_requests_count": 2,
            "engagement_latent_1": 0.0,
            "engagement_latent_2": 0.0,
            "engagement_latent_3": 0.0,
        },
    },
    {
        "username": "synth_student_3",
        "email": "s3@learnlens.com",
        "full_name": "Casey Nguyen",
        "password": "Synth1234!",
        "role": "student",
        "learner_id": "synth-003",
        "course_id": "BBB",
        "features": {
            "login_frequency_weekly": 3.5,
            "avg_session_duration_min": 45.0,
            "forum_posts_count": 2,
            "video_completion_rate": 0.6,
            "quiz_avg_score": 58.0,
            "quiz_completion_rate": 0.65,
            "assignment_submission_rate": 0.7,
            "days_since_last_activity": 4,
            "prior_course_completions": 1,
            "current_week_in_course": 5,
            "missed_deadlines_count": 2,
            "help_requests_count": 1,
            "engagement_latent_1": 0.0,
            "engagement_latent_2": 0.0,
            "engagement_latent_3": 0.0,
        },
    },
    {
        "username": "synth_student_4",
        "email": "s4@learnlens.com",
        "full_name": "Taylor Okafor",
        "password": "Synth1234!",
        "role": "student",
        "learner_id": "synth-004",
        "course_id": "AAA",
        "features": {
            "login_frequency_weekly": 0.5,
            "avg_session_duration_min": 8.0,
            "forum_posts_count": 0,
            "video_completion_rate": 0.1,
            "quiz_avg_score": 25.0,
            "quiz_completion_rate": 0.2,
            "assignment_submission_rate": 0.1,
            "days_since_last_activity": 21,
            "prior_course_completions": 0,
            "current_week_in_course": 4,
            "missed_deadlines_count": 6,
            "help_requests_count": 0,
            "engagement_latent_1": 0.0,
            "engagement_latent_2": 0.0,
            "engagement_latent_3": 0.0,
        },
    },
    {
        "username": "synth_student_5",
        "email": "s5@learnlens.com",
        "full_name": "Riley Santos",
        "password": "Synth1234!",
        "role": "student",
        "learner_id": "synth-005",
        "course_id": "BBB",
        "features": {
            "login_frequency_weekly": 5.0,
            "avg_session_duration_min": 60.0,
            "forum_posts_count": 8,
            "video_completion_rate": 0.85,
            "quiz_avg_score": 74.0,
            "quiz_completion_rate": 0.8,
            "assignment_submission_rate": 0.85,
            "days_since_last_activity": 2,
            "prior_course_completions": 2,
            "current_week_in_course": 10,
            "missed_deadlines_count": 1,
            "help_requests_count": 3,
            "engagement_latent_1": 0.0,
            "engagement_latent_2": 0.0,
            "engagement_latent_3": 0.0,
        },
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post(path: str, body=None, token: Optional[str] = None, params=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(f"{BASE}{path}", json=body, headers=headers, params=params)
    return r


def get(path: str, token: Optional[str] = None, params=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{BASE}{path}", headers=headers, params=params)
    return r


def register_user(user_data: dict) -> bool:
    r = post("/auth/register", user_data)
    if r.status_code == 201:
        print(f"  ✓ Created  {user_data['username']}  ({user_data['role']})")
        return True
    elif r.status_code == 409:
        print(f"  ~ Exists   {user_data['username']} — skipping")
        return True
    else:
        print(f"  ✗ ERROR    {user_data['username']}  {r.status_code}: {r.text[:120]}")
        return False


def login_user(username: str, password: str) -> Optional[str]:
    r = post("/auth/login", {"username": username, "password": password})
    if r.status_code == 200:
        return r.json()["access_token"]
    print(f"  ✗ Login failed for {username}: {r.status_code}")
    return None


def groq_generate_students(n: int = 5) -> list:
    """Call Groq to generate n synthetic student profiles with varied features."""
    try:
        import groq  # type: ignore
        client = groq.Groq(api_key=GROQ_API_KEY)
    except ImportError:
        try:
            import requests as _req
            # Use raw HTTP if groq package not installed
            prompt = _build_groq_prompt(n)
            resp = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.9, "max_tokens": 2000},
                timeout=30,
            )
            content = resp.json()["choices"][0]["message"]["content"]
            return _parse_groq_json(content)
        except Exception as e:
            print(f"  ! Groq HTTP call failed ({e}) — using fallback profiles")
            return []

    prompt = _build_groq_prompt(n)
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=2000,
        )
        content = completion.choices[0].message.content
        return _parse_groq_json(content)
    except Exception as e:
        print(f"  ! Groq generation failed ({e}) — using fallback profiles")
        return []


def _build_groq_prompt(n: int) -> str:
    return f"""Generate {n} realistic synthetic student profiles for an online learning system.
Return ONLY a JSON array (no markdown, no explanation) where each object has:
- username (string, alphanumeric, unique e.g. "learner_xyz123")
- email (string)
- full_name (string, realistic name)
- password (string, always "Synth1234!")
- role (always "student")
- learner_id (string, format "synth-groq-NNN")
- course_id (one of: "AAA", "BBB", "CCC")
- features (object with ALL of these float keys):
  login_frequency_weekly (0-14),
  avg_session_duration_min (0-300),
  forum_posts_count (0-30),
  video_completion_rate (0.0-1.0),
  quiz_avg_score (0-100),
  quiz_completion_rate (0.0-1.0),
  assignment_submission_rate (0.0-1.0),
  days_since_last_activity (0-60),
  prior_course_completions (0-5),
  current_week_in_course (1-12),
  missed_deadlines_count (0-10),
  help_requests_count (0-15),
  engagement_latent_1 (always 0.0),
  engagement_latent_2 (always 0.0),
  engagement_latent_3 (always 0.0)

Vary risk profiles: some high-risk (low engagement), some low-risk (high engagement), some medium.
"""


def _parse_groq_json(content: str) -> list:
    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        return json.loads(content[start:end])
    except Exception as e:
        print(f"  ! Could not parse Groq JSON: {e}")
        return []


def seed_explain(learner_id: str, features: dict, token: Optional[str] = None):
    body = {
        "features": features,
        "learner_id": learner_id,
        "model": "gbm",
        "audience": "learner",
        "history": [],
    }
    r = post("/explain", body, token=token)
    if r.status_code == 200:
        risk = r.json().get("risk_score", "?")
        print(f"    explain OK  learner={learner_id}  risk={risk:.3f}" if isinstance(risk, float) else f"    explain OK  learner={learner_id}")
    else:
        print(f"    explain ERR {r.status_code}: {r.text[:80]}")


def seed_feedback(learner_id: str, token: Optional[str] = None):
    r = post("/feedback", {
        "learner_id": learner_id,
        "rating": random.randint(3, 5),
        "followed_recommendation": random.random() > 0.3,
        "audience": "learner",
    }, token=token)
    if r.status_code != 200:
        print(f"    feedback ERR {r.status_code}")


def seed_events(learner_id: str, token: Optional[str] = None):
    event_types = [
        ("page_view", "student_dashboard", None),
        ("click", "shap_bar_chart", None),
        ("whatif_slider", "login_frequency_weekly", 5.0),
        ("action_viewed", "assignment_submission_rate", None),
        ("explanation_revisit", learner_id, None),
    ]
    events = [
        {
            "learner_id": learner_id,
            "session_id": f"seed-session-{learner_id}",
            "event_type": et,
            "event_target": tgt,
            "event_value": val,
            "page": "/dashboard/student",
            "client_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        for et, tgt, val in event_types
    ]
    r = post("/events", {"events": events}, token=token)
    if r.status_code != 200:
        print(f"    events ERR {r.status_code}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("XAI Learning System — Demo Data Seeder")
    print(f"Target: {BASE}")
    print("=" * 60)

    # 1. Health check
    r = requests.get(f"{BASE}/health", timeout=5)
    if r.status_code != 200:
        print(f"\n✗ Backend not reachable at {BASE} (status {r.status_code}). Aborting.")
        sys.exit(1)
    print(f"\n✓ Backend healthy — {r.json().get('model_version', 'unknown version')}")

    # 2. Register demo accounts
    print("\n── Registering demo accounts ──")
    for account in DEMO_ACCOUNTS:
        register_user(account)

    # 3. Login as instructor to get token for enrollments
    print("\n── Logging in as demo_instructor ──")
    instructor_token = login_user("demo_instructor", "DemoInstructor2026!")

    # 4. Enroll demo_student under demo_instructor
    if instructor_token:
        print("\n── Enrolling demo_student ──")
        r = post(
            "/auth/enroll",
            None,
            token=instructor_token,
            params={"learner_id": "demo-learner-001", "course_id": "AAA"},
        )
        if r.status_code == 200:
            print("  ✓ Enrolled demo-learner-001 under demo_instructor (AAA)")
        elif r.status_code == 409:
            print("  ~ Already enrolled")
        else:
            print(f"  ~ Enroll note: {r.status_code} — {r.text[:80]}")

    # 5. Generate synthetic students via Groq (or fallback)
    print("\n── Generating synthetic students ──")
    groq_students = []
    if GROQ_API_KEY:
        print("  Using Groq LLM for profile generation…")
        groq_students = groq_generate_students(5)
        if groq_students:
            print(f"  ✓ Groq generated {len(groq_students)} profiles")

    synthetic_students = groq_students if groq_students else FALLBACK_STUDENTS
    if not groq_students:
        print("  Using hardcoded fallback profiles")

    # Register synthetic students and enroll under demo_instructor
    student_tokens = {}
    for s in synthetic_students:
        reg_body = {k: v for k, v in s.items() if k != "features"}
        register_user(reg_body)
        tok = login_user(s["username"], s["password"])
        if tok:
            student_tokens[s["learner_id"]] = (tok, s.get("features"))
        if instructor_token and s.get("learner_id") and s.get("course_id"):
            r = post(
                "/auth/enroll",
                None,
                token=instructor_token,
                params={"learner_id": s["learner_id"], "course_id": s["course_id"]},
            )
            if r.status_code not in (200, 409):
                print(f"  ~ Enroll note {s['learner_id']}: {r.status_code}")

    # 6. Populate explanation history, feedback, events for all students
    demo_student_token = login_user("demo_student", "DemoStudent2026!")
    demo_features = {
        "login_frequency_weekly": 2.5,
        "avg_session_duration_min": 25.0,
        "forum_posts_count": 1,
        "video_completion_rate": 0.45,
        "quiz_avg_score": 52.0,
        "quiz_completion_rate": 0.5,
        "assignment_submission_rate": 0.55,
        "days_since_last_activity": 7,
        "prior_course_completions": 0,
        "current_week_in_course": 6,
        "missed_deadlines_count": 3,
        "help_requests_count": 1,
        "engagement_latent_1": 0.0,
        "engagement_latent_2": 0.0,
        "engagement_latent_3": 0.0,
    }

    print("\n── Seeding explanation history ──")
    print("  demo_student:")
    seed_explain("demo-learner-001", demo_features, demo_student_token)
    seed_feedback("demo-learner-001", demo_student_token)
    seed_events("demo-learner-001", demo_student_token)

    for learner_id, (tok, features) in student_tokens.items():
        if features:
            print(f"  {learner_id}:")
            seed_explain(learner_id, features, tok)
            seed_feedback(learner_id, tok)
            seed_events(learner_id, tok)

    # 7. Summary table
    all_users = [
        ("demo_student",     "DemoStudent2026!",     "student",    "demo-learner-001"),
        ("demo_instructor",  "DemoInstructor2026!",  "instructor", "—"),
        ("demo_admin",       "DemoAdmin2026!",        "admin",      "—"),
    ] + [
        (s["username"], s["password"], "student", s.get("learner_id", "—"))
        for s in synthetic_students
    ]

    print("\n" + "=" * 60)
    print("Demo Accounts Summary")
    print("=" * 60)
    print(f"{'Username':<22} {'Password':<24} {'Role':<12} {'LearnerID'}")
    print("-" * 60)
    for username, password, role, lid in all_users:
        print(f"{username:<22} {password:<24} {role:<12} {lid}")
    print("=" * 60)
    print("\nDone. Run the frontend and log in with any of the above accounts.")


if __name__ == "__main__":
    main()
