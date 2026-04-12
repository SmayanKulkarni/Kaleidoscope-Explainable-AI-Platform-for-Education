#!/usr/bin/env python3
"""Seed a medium OULAD-derived cohort with auth users, snapshot state, and history.

This script:
- reads data/temporal/snapshots.pkl for real weekly learner snapshots,
- reads data/learners.csv for stable learner metadata,
- creates/upserts student accounts, learner profiles, and current StudentSnapshot rows,
- creates/upserts a small instructor cohort,
- seeds explanation history records with snapshot-backed feature payloads,
- exports login credentials for local use.

Usage:
    python scripts/seed_oulad_users.py --cohort-size 50
    python scripts/seed_oulad_users.py --cohort-size 10 --snapshot-week 6 --sample-offset 50
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from backend.app.auth.auth import hash_password
from backend.app.auth.database import _SessionFactory, engine
from backend.app.auth.models import Base, CourseEnrollment, InstructorProfile, LearnerProfile, StudentSnapshot, User
from backend.app.tracker.consistency_store import ExplanationStore

Base.metadata.create_all(bind=engine)

PROJECT_ROOT = ROOT
DATA_DIR = PROJECT_ROOT / "data"
TEMPORAL_PATH = DATA_DIR / "temporal" / "snapshots.pkl"
LEARNERS_PATH = DATA_DIR / "learners.csv"
CREDENTIALS_JSON = DATA_DIR / "oulad_seed_credentials.json"
CREDENTIALS_CSV = DATA_DIR / "oulad_seed_credentials.csv"

STUDENT_PASSWORD = "Student2026!"
INSTRUCTOR_PASSWORD = "Instructor2026!"
INSTRUCTOR_ROWS = [
    ("Dr. Sarah Chen", "sarah.chen", "Computer Science"),
    ("Prof. Marcus Williams", "marcus.williams", "Mathematics"),
    ("Dr. Priya Sharma", "priya.sharma", "Education Technology"),
    ("Prof. James Okafor", "james.okafor", "Engineering"),
    ("Dr. Lisa Tanaka", "lisa.tanaka", "Data Science"),
]

OULAD_FEATURES = [
    "login_frequency_weekly",
    "avg_session_duration_min",
    "forum_posts_count",
    "video_completion_rate",
    "quiz_avg_score",
    "quiz_completion_rate",
    "assignment_submission_rate",
    "days_since_last_activity",
    "prior_course_completions",
    "current_week_in_course",
    "missed_deadlines_count",
    "help_requests_count",
]


def _risk_score(features: dict[str, Any]) -> float:
    low_eng = max(0.0, 1.0 - float(features["video_completion_rate"]))
    low_quiz = max(0.0, 1.0 - float(features["quiz_avg_score"]) / 100.0)
    low_sub = max(0.0, 1.0 - float(features["assignment_submission_rate"]))
    inactive = min(1.0, float(features["days_since_last_activity"]) / 30.0)
    missed = min(1.0, float(features["missed_deadlines_count"]) / 10.0)
    return round(0.25 * low_eng + 0.25 * low_quiz + 0.2 * low_sub + 0.15 * inactive + 0.15 * missed, 3)


def _top_features(features: dict[str, Any]) -> list[dict[str, Any]]:
    weights = {
        "login_frequency_weekly": -0.12,
        "avg_session_duration_min": -0.10,
        "forum_posts_count": -0.08,
        "video_completion_rate": -0.18,
        "quiz_avg_score": -0.20,
        "quiz_completion_rate": -0.15,
        "assignment_submission_rate": -0.17,
        "days_since_last_activity": 0.18,
        "prior_course_completions": -0.05,
        "current_week_in_course": 0.03,
        "missed_deadlines_count": 0.15,
        "help_requests_count": 0.04,
    }
    scored = []
    for feature_name, weight in weights.items():
        value = float(features.get(feature_name, 0.0))
        scored.append({"name": feature_name, "shap": round(weight * value, 4)})
    scored.sort(key=lambda item: abs(item["shap"]), reverse=True)
    return scored[:3]


def _load_snapshots() -> pd.DataFrame:
    if not TEMPORAL_PATH.exists():
        raise FileNotFoundError(f"{TEMPORAL_PATH} not found. Run temporal_builder.py first.")
    with open(TEMPORAL_PATH, "rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or "snapshots" not in raw:
        raise ValueError("snapshots.pkl has unexpected structure")
    return raw["snapshots"].copy()


def _load_learner_meta() -> pd.DataFrame:
    if not LEARNERS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LEARNERS_PATH)
    if "learner_id" in df.columns:
        return df.set_index("learner_id")
    return pd.DataFrame()


def _username_from_learner(learner_id: str) -> str:
    safe = learner_id.lower().replace(" ", "_").replace("-", "_")
    return f"oulad_{safe}"[:64]


def _full_name_for_learner(row: pd.Series) -> str:
    student_id = row.get("id_student")
    module = row.get("code_module", "OULAD")
    presentation = row.get("code_presentation", "")
    return f"OULAD Learner {student_id} {module} {presentation}".strip()


def _snapshot_features(row: pd.Series) -> dict[str, Any]:
    return {feature: row[feature].item() if hasattr(row[feature], "item") else row[feature] for feature in OULAD_FEATURES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed OULAD-derived users, snapshots, and history")
    parser.add_argument("--cohort-size", type=int, default=50)
    parser.add_argument("--snapshot-week", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--credentials-json", type=Path, default=CREDENTIALS_JSON)
    parser.add_argument("--credentials-csv", type=Path, default=CREDENTIALS_CSV)
    args = parser.parse_args()

    random.seed(args.seed)
    snapshots = _load_snapshots()
    learner_meta = _load_learner_meta()
    db = _SessionFactory()
    history_store = ExplanationStore()
    credentials: list[dict[str, Any]] = []

    try:
        if args.snapshot_week is None:
            selected_rows = snapshots.sort_values(["learner_id", "snapshot_week"]).groupby("learner_id").tail(1)
        else:
            selected_rows = snapshots[snapshots["snapshot_week"] == args.snapshot_week].copy()
        unique_learners = selected_rows["learner_id"].drop_duplicates().sort_values().tolist()
        if not unique_learners:
            raise RuntimeError("No learners found in temporal snapshots")

        sample_start = max(0, args.sample_offset)
        sample_end = sample_start + max(0, args.cohort_size)
        sampled_ids = unique_learners[sample_start:sample_end]
        sampled_rows = selected_rows.set_index("learner_id").loc[sampled_ids].reset_index()
        sample_size = len(sampled_ids)

        existing_credentials: list[dict[str, Any]] = []
        if args.credentials_json.exists():
            try:
                existing_credentials = json.loads(args.credentials_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_credentials = []

        # Upsert a small reusable instructor cohort.
        instructor_profile_ids: list[str] = []
        for index, (full_name, username_base, department) in enumerate(INSTRUCTOR_ROWS):
            username = f"oulad_{username_base.replace('.', '_')}"[:64]
            email = f"{username_base}@oulad.local"
            user = db.query(User).filter(User.username == username).first()
            if user is None:
                user = User(
                    username=username,
                    email=email,
                    full_name=full_name,
                    hashed_password=hash_password(INSTRUCTOR_PASSWORD),
                    role="instructor",
                    is_active=True,
                )
                db.add(user)
                db.flush()
            else:
                user = cast(Any, user)
                user.full_name = full_name
                user.email = email
                user.hashed_password = hash_password(INSTRUCTOR_PASSWORD)
                user.role = "instructor"
                user.is_active = True
                db.flush()

            instructor_profile = db.query(InstructorProfile).filter(InstructorProfile.user_id == user.id).first()
            if instructor_profile is None:
                instructor_profile = InstructorProfile(user_id=user.id, department=department, bio="OULAD seeded instructor")
                db.add(instructor_profile)
                db.flush()
            else:
                instructor_profile = cast(Any, instructor_profile)
                instructor_profile.department = department
                instructor_profile.bio = "OULAD seeded instructor"

            instructor_profile_ids.append(str(instructor_profile.id))
            credentials.append({
                "role": "instructor",
                "username": username,
                "password": INSTRUCTOR_PASSWORD,
                "full_name": full_name,
                "user_id": user.id,
                "instructor_profile_id": instructor_profile.id,
                "department": department,
            })

        db.commit()

        # Upsert student accounts, learner profiles, snapshots, and history.
        for index, (_, row) in enumerate(sampled_rows.iterrows()):
            row_dict = row.to_dict()
            learner_id = str(row_dict["learner_id"])
            username = _username_from_learner(learner_id)
            email = f"{username}@oulad.local"
            full_name = _full_name_for_learner(row)
            course_id = f"{row_dict['code_module']}_{row_dict['code_presentation']}"

            meta_row = learner_meta.loc[learner_id] if learner_id in learner_meta.index else None
            prior_completions = int(cast(Any, meta_row).get("prior_course_completions", 0)) if meta_row is not None and "prior_course_completions" in learner_meta.columns else 0

            user = db.query(User).filter(User.username == username).first()
            if user is None:
                user = User(
                    username=username,
                    email=email,
                    full_name=full_name,
                    hashed_password=hash_password(STUDENT_PASSWORD),
                    role="student",
                    is_active=True,
                )
                db.add(user)
                db.flush()
            else:
                user = cast(Any, user)
                user.email = email
                user.full_name = full_name
                user.hashed_password = hash_password(STUDENT_PASSWORD)
                user.role = "student"
                user.is_active = True
                db.flush()

            learner_profile = db.query(LearnerProfile).filter(LearnerProfile.learner_id == learner_id).first()
            if learner_profile is None:
                learner_profile = LearnerProfile(
                    user_id=user.id,
                    learner_id=learner_id,
                    course_id=course_id,
                    current_week=int(row_dict["current_week_in_course"]),
                    prior_completions=prior_completions,
                )
                db.add(learner_profile)
                db.flush()
            else:
                learner_profile = cast(Any, learner_profile)
                learner_profile.user_id = user.id
                learner_profile.course_id = course_id
                learner_profile.current_week = int(row_dict["current_week_in_course"])
                learner_profile.prior_completions = prior_completions

            latest_features = _snapshot_features(row)
            latest_risk = _risk_score(latest_features)
            snapshot = db.query(StudentSnapshot).filter(StudentSnapshot.learner_id == learner_id).first()
            if snapshot is None:
                snapshot = StudentSnapshot(learner_id=learner_id)
                db.add(snapshot)
            snapshot = cast(Any, snapshot)
            snapshot.display_name = full_name
            snapshot.login_frequency_weekly = float(row_dict["login_frequency_weekly"])
            snapshot.avg_session_duration_min = float(row_dict["avg_session_duration_min"])
            snapshot.forum_posts_count = int(row_dict["forum_posts_count"])
            snapshot.video_completion_rate = float(row_dict["video_completion_rate"])
            snapshot.quiz_avg_score = float(row_dict["quiz_avg_score"])
            snapshot.quiz_completion_rate = float(row_dict["quiz_completion_rate"])
            snapshot.assignment_submission_rate = float(row_dict["assignment_submission_rate"])
            snapshot.days_since_last_activity = int(row_dict["days_since_last_activity"])
            snapshot.prior_course_completions = prior_completions
            snapshot.current_week_in_course = int(row_dict["current_week_in_course"])
            snapshot.missed_deadlines_count = int(row_dict["missed_deadlines_count"])
            snapshot.help_requests_count = int(row_dict["help_requests_count"])
            snapshot.dropout_risk_score = latest_risk
            snapshot.risk_trajectory = "at_risk_stable" if latest_risk >= 0.7 else ("stable" if latest_risk >= 0.35 else "improving")
            snapshot.current_module = str(row_dict["code_module"])
            snapshot.current_presentation = str(row_dict["code_presentation"])

            # Backfill real weekly history records from the temporal snapshots.
            weekly_rows = snapshots[snapshots["learner_id"] == learner_id].sort_values("snapshot_week")
            if weekly_rows.empty:
                weekly_rows = pd.DataFrame([row_dict])

            for weekly_row in weekly_rows.to_dict(orient="records"):
                weekly_features = {feature: weekly_row[feature] for feature in OULAD_FEATURES}
                weekly_risk = _risk_score(weekly_features)
                history_store.save(
                    learner_id=learner_id,
                    risk_score=weekly_risk,
                    shap_values={feature: round(float(weekly_features[feature]) / 100.0, 4) if isinstance(weekly_features[feature], (int, float)) else 0.0 for feature in OULAD_FEATURES},
                    top3_features=_top_features(weekly_features),
                    trust_score=round(max(0.25, 1.0 - weekly_risk), 4),
                    model_used="seeded_snapshot",
                    anchor_rule="seeded_from_temporal_snapshot",
                    extra={
                        "features": weekly_features,
                        "snapshot_week": int(weekly_row["snapshot_week"]),
                        "code_module": weekly_row["code_module"],
                        "code_presentation": weekly_row["code_presentation"],
                    },
                )

            # Round-robin instructor enrollment so the instructor UI has useful rosters.
            if instructor_profile_ids:
                instructor_profile_id = instructor_profile_ids[index % len(instructor_profile_ids)]
                existing_enrollment = db.query(CourseEnrollment).filter(
                    CourseEnrollment.learner_profile_id == learner_profile.id,
                    CourseEnrollment.instructor_profile_id == instructor_profile_id,
                    CourseEnrollment.course_id == course_id,
                ).first()
                if existing_enrollment is None:
                    db.add(CourseEnrollment(
                        learner_profile_id=learner_profile.id,
                        instructor_profile_id=instructor_profile_id,
                        course_id=course_id,
                    ))

            credentials.append({
                "role": "student",
                "username": username,
                "password": STUDENT_PASSWORD,
                "full_name": full_name,
                "learner_id": learner_id,
                "user_id": user.id,
                "course_id": course_id,
                "snapshot_week": int(row_dict["current_week_in_course"]),
                "risk_score": latest_risk,
            })

        db.commit()

        merged_credentials_by_username: dict[str, dict[str, Any]] = {}
        for entry in existing_credentials + credentials:
            username = str(entry.get("username", ""))
            if username:
                merged_credentials_by_username[username] = entry

        merged_credentials = list(merged_credentials_by_username.values())

        args.credentials_json.parent.mkdir(parents=True, exist_ok=True)
        args.credentials_json.write_text(json.dumps(merged_credentials, indent=2), encoding="utf-8")

        with args.credentials_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for entry in merged_credentials for key in entry.keys()}))
            writer.writeheader()
            for entry in merged_credentials:
                writer.writerow(entry)

        print(f"Seeded {sample_size} students and {len(INSTRUCTOR_ROWS)} instructors")
        print(f"Credentials JSON: {args.credentials_json}")
        print(f"Credentials CSV : {args.credentials_csv}")
        print("Student password : Student2026!")
        print("Instructor password : Instructor2026!")
    finally:
        db.close()


if __name__ == "__main__":
    main()
