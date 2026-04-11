#!/usr/bin/env python3
"""Generate synthetic recommendation datasets for students and instructors.

This script uses Groq to synthesize recommendation archetypes, then expands
them against the existing OULAD-derived tables in this repository to produce:

1) student recommendation dataset with implicit feedback signals
2) instructor recommendation dataset for assignment guidance
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass
class GroqConfig:
    api_key: str
    model: str


def _extract_json_payload(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _call_groq_json(cfg: GroqConfig, system_prompt: str, user_prompt: str) -> Any:
    from groq import Groq

    client = Groq(api_key=cfg.api_key)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.6,
        max_tokens=1800,
    )
    raw = response.choices[0].message.content or "{}"
    payload = _extract_json_payload(raw)
    return json.loads(payload)


def _load_sources(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    learners = pd.read_csv(root / "data" / "learners.csv")
    student_info = pd.read_csv(root / "data" / "raw" / "studentInfo.csv")
    courses = pd.read_csv(root / "data" / "raw" / "courses.csv")

    merged = learners.merge(
        student_info,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
        suffixes=("", "_info"),
    )
    return merged, student_info, courses


def _build_student_archetypes(cfg: GroqConfig, modules: List[str], count: int) -> List[Dict[str, Any]]:
    system = (
        "You create realistic synthetic learner archetypes for recommendation systems. "
        "Output valid JSON only, no markdown."
    )
    user = f"""
Generate {count} learner archetypes for a course recommendation MVP.
Available course modules: {modules}

Return a JSON object with key "archetypes" and value as an array.
Each archetype must include:
- segment_name (string)
- engagement_level (float between 0 and 1)
- career_focus (float between 0 and 1)
- risk_aversion (float between 0 and 1)
- implicit_feedback_profile (object):
  - clicks_14d_mean (int 5-120)
  - video_watch_ratio_mean (float 0-1)
  - forum_events_14d_mean (int 0-40)
  - quiz_attempts_14d_mean (int 0-25)
- module_affinity (object with module code keys and float values between -0.2 and 0.4)

Constraints:
- Use all modules across archetypes.
- Keep values internally consistent.
"""
    parsed = _call_groq_json(cfg, system, user)
    archetypes = parsed.get("archetypes", []) if isinstance(parsed, dict) else []
    if not archetypes:
        raise ValueError("Groq did not return valid learner archetypes.")
    return archetypes


def _build_instructor_archetypes(cfg: GroqConfig, count: int) -> List[Dict[str, Any]]:
    system = (
        "You create realistic synthetic instructor profiles and intervention styles for educational recommendation systems. "
        "Output valid JSON only, no markdown."
    )
    user = f"""
Generate {count} instructor archetypes for course assignment recommendation.

Return JSON object with key "instructor_archetypes".
Each archetype must include ALL of these fields:
- archetype_name (string, e.g. "Structured Mentor", "Challenge-Driven Coach")
- teaching_style (one of: "scaffolded", "socratic", "direct_instruction", "project_based", "flipped")
- department (one of: "STEM", "Humanities", "Business", "Health", "Computing", "Social_Sciences")
- experience_years_min (int 1-30)
- experience_years_max (int, must be >= experience_years_min, max 35)
- intervention_intensity (float 0-1, how proactively this instructor intervenes)
- remediation_bias (float 0-1, preference for remedial assignments over challenge)
- challenge_bias (float 0-1, preference for stretch assignments)
- enrichment_bias (float 0-1, preference for optional enrichment content)
- peer_review_bias (float 0-1, preference for peer-review tasks)
- cohort_signal_weight (float 0-1)
- student_affinity_weight (float 0-1)
- success_rate_mean (float 0.4-0.95, typical past intervention success rate)
- success_rate_std (float 0.02-0.12)
- avg_cohort_size_mean (int 15-120)
- confidence_baseline (float 0.5-0.95, how confident instructor is in recommendations)

Note: remediation_bias + challenge_bias + enrichment_bias + peer_review_bias should sum close to 1.
cohort_signal_weight + student_affinity_weight should sum close to 1.
"""
    parsed = _call_groq_json(cfg, system, user)
    archetypes = parsed.get("instructor_archetypes", []) if isinstance(parsed, dict) else []
    if not archetypes:
        raise ValueError("Groq did not return valid instructor archetypes.")
    return archetypes


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _age_band_to_score(age_band: Any) -> float:
    text = str(age_band).strip()
    if text == "0-35":
        return 0.55
    if text == "35-55":
        return 0.7
    if text == "55<=":
        return 0.6
    return 0.5


def _imd_to_score(imd_band: Any) -> float:
    text = str(imd_band).strip()
    if not text or text.lower() == "nan":
        return 0.5
    if "-" in text and "%" in text:
        low = text.split("-")[0].strip().replace("%", "")
        try:
            # Higher IMD band is treated as mild access/stability advantage.
            return float(np.clip(float(low) / 100.0, 0.2, 1.0))
        except ValueError:
            return 0.5
    return 0.5


def _build_student_dataset(
    merged: pd.DataFrame,
    courses: pd.DataFrame,
    archetypes: List[Dict[str, Any]],
    n_students: int,
    top_k: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)

    modules = sorted(courses["code_module"].astype(str).unique().tolist())
    course_rows = courses[["code_module", "code_presentation", "module_presentation_length"]].copy()
    course_rows["module_presentation_length"] = pd.to_numeric(
        course_rows["module_presentation_length"], errors="coerce"
    ).fillna(260)

    sampled = merged.sample(n=min(n_students, len(merged)), random_state=seed).reset_index(drop=True)

    outputs: List[Dict[str, Any]] = []
    for idx, row in sampled.iterrows():
        arche = archetypes[idx % len(archetypes)]
        profile = arche.get("implicit_feedback_profile", {})
        module_affinity = arche.get("module_affinity", {})

        login_freq = _to_float(row.get("login_frequency_weekly"), 0.0)
        video_completion = _to_float(row.get("video_completion_rate"), 0.0)
        forum_posts = _to_float(row.get("forum_posts_count"), 0.0)
        quiz_rate = _to_float(row.get("quiz_completion_rate"), 0.0)
        quiz_avg_score = _to_float(row.get("quiz_avg_score"), 0.0)
        assignment_submission_rate = _to_float(row.get("assignment_submission_rate"), 0.0)
        missed_deadlines_count = _to_float(row.get("missed_deadlines_count"), 0.0)
        prior_course_completions = _to_float(row.get("prior_course_completions"), 0.0)
        days_since_last_activity = _to_float(row.get("days_since_last_activity"), 0.0)
        avg_session_duration_min = _to_float(row.get("avg_session_duration_min"), 0.0)
        help_requests_count = _to_float(row.get("help_requests_count"), 0.0)
        current_week_in_course = _to_float(row.get("current_week_in_course"), 0.0)
        studied_credits = _to_float(row.get("studied_credits"), 60.0)
        num_prev_attempts = _to_float(row.get("num_of_prev_attempts"), 0.0)
        dropout_risk = _to_float(row.get("dropout_risk"), 0.0)

        gender = str(row.get("gender", "")).strip()
        region = str(row.get("region", "")).strip()
        highest_education = str(row.get("highest_education", "")).strip()
        imd_band = str(row.get("imd_band", "")).strip()
        age_band = str(row.get("age_band", "")).strip()
        disability = str(row.get("disability", "N")).strip()

        edu_score_map = {
            "No Formal quals": 0.35,
            "Lower Than A Level": 0.45,
            "A Level or Equivalent": 0.6,
            "HE Qualification": 0.75,
            "Post Graduate Qualification": 0.85,
        }
        education_score = edu_score_map.get(highest_education, 0.55)
        age_score = _age_band_to_score(age_band)
        imd_score = _imd_to_score(imd_band)
        accessibility_need_score = 1.0 if disability.upper() == "Y" else 0.0

        clicks_14d = max(
            0,
            _to_int(
                np.random.normal(
                    _to_float(profile.get("clicks_14d_mean"), 20)
                    + login_freq * 4,
                    6,
                )
            ),
        )
        watch_ratio_14d = float(
            np.clip(
                np.random.normal(
                    _to_float(profile.get("video_watch_ratio_mean"), 0.45)
                    + 0.25 * video_completion,
                    0.08,
                ),
                0.0,
                1.0,
            )
        )
        forum_events_14d = max(
            0,
            _to_int(
                np.random.normal(
                    _to_float(profile.get("forum_events_14d_mean"), 4)
                    + forum_posts / 30.0,
                    2,
                )
            ),
        )
        quiz_attempts_14d = max(
            0,
            _to_int(
                np.random.normal(
                    _to_float(profile.get("quiz_attempts_14d_mean"), 3)
                    + quiz_rate * 5,
                    1.5,
                )
            ),
        )

        avg_dwell_time_min_14d = float(
            np.clip(
                np.random.normal(0.6 * avg_session_duration_min + 12.0, 7.0),
                2.0,
                180.0,
            )
        )
        save_events_14d = max(
            0,
            _to_int(
                np.random.normal(
                    0.10 * clicks_14d + 0.20 * forum_events_14d + 0.35 * help_requests_count,
                    2.0,
                )
            ),
        )
        search_events_14d = max(
            0,
            _to_int(
                np.random.normal(
                    0.12 * clicks_14d + 0.15 * help_requests_count,
                    2.5,
                )
            ),
        )
        last_recommendation_interaction_days = max(
            0,
            _to_int(np.random.normal(max(1.0, days_since_last_activity * 0.5), 4.0)),
        )

        activity_recency = float(np.clip(1.0 - (days_since_last_activity / 120.0), 0.0, 1.0))
        learning_momentum = float(
            np.clip(
                0.45 * assignment_submission_rate
                + 0.35 * quiz_rate
                + 0.20 * activity_recency
                - 0.08 * min(missed_deadlines_count / 8.0, 1.0),
                0.0,
                1.0,
            )
        )

        explicit_interest_level = float(
            np.clip(
                0.30 * _to_float(arche.get("engagement_level"), 0.5)
                + 0.22 * _to_float(arche.get("career_focus"), 0.5)
                + 0.14 * (1.0 - dropout_risk)
                + 0.12 * watch_ratio_14d
                + 0.10 * learning_momentum
                + 0.06 * education_score
                + 0.04 * imd_score
                + 0.02 * age_score,
                0.0,
                1.0,
            )
        )

        scored_courses: List[tuple[float, Dict[str, Any]]] = []
        for c in course_rows.to_dict("records"):
            code_module = str(c["code_module"])
            module_bias = _to_float(module_affinity.get(code_module), 0.0)
            score = (
                0.22 * explicit_interest_level
                + 0.16 * (1.0 - dropout_risk)
                + 0.18 * watch_ratio_14d
                + 0.12 * min(clicks_14d / 100.0, 1.0)
                + 0.1 * min(forum_events_14d / 20.0, 1.0)
                + 0.1 * min(quiz_attempts_14d / 15.0, 1.0)
                + 0.06 * learning_momentum
                + 0.05 * min(quiz_avg_score / 100.0, 1.0)
                + 0.04 * min(prior_course_completions / 6.0, 1.0)
                + 0.03 * min(studied_credits / 240.0, 1.0)
                - 0.02 * min(num_prev_attempts / 6.0, 1.0)
                + 0.03 * min(avg_dwell_time_min_14d / 120.0, 1.0)
                + 0.03 * min(save_events_14d / 20.0, 1.0)
                + 0.02 * min(search_events_14d / 20.0, 1.0)
                - 0.02 * min(last_recommendation_interaction_days / 30.0, 1.0)
                + 0.02 * education_score
                + 0.02 * imd_score
                - 0.01 * accessibility_need_score
                + module_bias
            )
            if code_module == str(row.get("code_module")):
                score -= 0.04
            scored_courses.append((score, c))

        scored_courses.sort(key=lambda x: x[0], reverse=True)
        top = scored_courses[:top_k]
        for rank, (score, course_rec) in enumerate(top, start=1):
            outputs.append(
                {
                    "student_id": int(row["id_student"]),
                    "learner_id": row.get("learner_id", f"learner_{idx:05d}"),
                    "current_module": row.get("code_module"),
                    "current_presentation": row.get("code_presentation"),
                    "recommended_module": course_rec["code_module"],
                    "recommended_presentation": course_rec["code_presentation"],
                    "recommendation_score": round(float(score), 4),
                    "rank": rank,
                    "implicit_clicks_14d": clicks_14d,
                    "implicit_video_watch_ratio_14d": round(watch_ratio_14d, 4),
                    "implicit_forum_events_14d": forum_events_14d,
                    "implicit_quiz_attempts_14d": quiz_attempts_14d,
                    "implicit_avg_dwell_time_min_14d": round(avg_dwell_time_min_14d, 3),
                    "implicit_save_events_14d": save_events_14d,
                    "implicit_search_events_14d": search_events_14d,
                    "implicit_last_recommendation_interaction_days": last_recommendation_interaction_days,
                    "explicit_interest_level": round(explicit_interest_level, 4),
                    "learning_momentum": round(learning_momentum, 4),
                    "dropout_risk": int(dropout_risk),
                    "base_login_frequency_weekly": round(login_freq, 4),
                    "base_video_completion_rate": round(video_completion, 4),
                    "base_quiz_completion_rate": round(quiz_rate, 4),
                    "base_quiz_avg_score": round(quiz_avg_score, 4),
                    "base_assignment_submission_rate": round(assignment_submission_rate, 4),
                    "base_missed_deadlines_count": int(round(missed_deadlines_count)),
                    "base_prior_course_completions": int(round(prior_course_completions)),
                    "base_days_since_last_activity": int(round(days_since_last_activity)),
                    "base_studied_credits": int(round(studied_credits)),
                    "base_num_of_prev_attempts": int(round(num_prev_attempts)),
                    "base_avg_session_duration_min": round(avg_session_duration_min, 4),
                    "base_help_requests_count": int(round(help_requests_count)),
                    "base_current_week_in_course": int(round(current_week_in_course)),
                    "explicit_gender": gender,
                    "explicit_region": region,
                    "explicit_highest_education": highest_education,
                    "explicit_imd_band": imd_band,
                    "explicit_age_band": age_band,
                    "explicit_disability": disability,
                    "explicit_education_score": round(education_score, 4),
                    "explicit_imd_score": round(imd_score, 4),
                    "explicit_age_score": round(age_score, 4),
                    "synthetic_segment": arche.get("segment_name", "general"),
                    "recommendation_reason": "blended_current_features_plus_implicit_feedback",
                }
            )

    student_df = pd.DataFrame(outputs)
    student_df["recommendation_score"] = student_df["recommendation_score"].clip(0, 1.2)
    return student_df


def _intervention_type(arche: Dict[str, Any], dropout_risk: float, learning_momentum: float) -> str:
    """Pick intervention type based on archetype biases and student state."""
    rem  = _to_float(arche.get("remediation_bias"),  0.25)
    chal = _to_float(arche.get("challenge_bias"),    0.25)
    enr  = _to_float(arche.get("enrichment_bias"),   0.25)
    peer = _to_float(arche.get("peer_review_bias"),  0.25)

    # At-risk students weight toward remediation; high-momentum toward challenge/enrichment
    rem  += 0.25 * dropout_risk
    chal += 0.15 * learning_momentum
    enr  += 0.10 * learning_momentum
    peer += 0.05

    weights = {"remediation": rem, "challenge": chal, "enrichment": enr, "peer_review": peer}
    return max(weights, key=lambda k: weights[k])


def _intervention_urgency(dropout_risk: float, days_inactive: float, missed_deadlines: float) -> str:
    score = 0.5 * dropout_risk + 0.3 * min(days_inactive / 30.0, 1.0) + 0.2 * min(missed_deadlines / 5.0, 1.0)
    if score > 0.65:
        return "high"
    if score > 0.35:
        return "medium"
    return "low"


def _content_type(intervention: str, teaching_style: str) -> str:
    mapping = {
        ("remediation", "scaffolded"):        "guided_worksheet",
        ("remediation", "direct_instruction"): "video_lecture_recap",
        ("remediation", "socratic"):           "reflection_questions",
        ("challenge", "project_based"):        "mini_project",
        ("challenge", "flipped"):              "advanced_case_study",
        ("challenge", "socratic"):             "debate_prompt",
        ("enrichment", "project_based"):       "optional_extension_task",
        ("enrichment", "flipped"):             "curated_reading_list",
        ("peer_review", "socratic"):           "structured_peer_critique",
        ("peer_review", "scaffolded"):         "rubric_guided_peer_review",
    }
    key = (intervention, teaching_style)
    if key in mapping:
        return mapping[key]
    defaults = {
        "remediation": "practice_exercises",
        "challenge":   "challenge_problem_set",
        "enrichment":  "supplementary_materials",
        "peer_review": "peer_feedback_exchange",
    }
    return defaults.get(intervention, "mixed_content")


def _priority_reason_tag(
    dropout_risk: float,
    learning_momentum: float,
    days_inactive: float,
    quiz_avg_score: float,
    missed_deadlines: float,
) -> str:
    if dropout_risk > 0.65:
        return "high_dropout_risk"
    if missed_deadlines >= 3 and days_inactive > 10:
        return "disengagement_pattern"
    if quiz_avg_score < 40.0:
        return "low_assessment_performance"
    if learning_momentum > 0.7 and dropout_risk < 0.3:
        return "high_achiever_stretch"
    if days_inactive > 14:
        return "prolonged_inactivity"
    if learning_momentum < 0.35:
        return "low_engagement_momentum"
    return "routine_progression"


def _build_instructor_dataset(
    student_df: pd.DataFrame,
    instructor_archetypes: List[Dict[str, Any]],
    n_instructors: int,
    top_k: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed + 101)
    np_rng = np.random.default_rng(seed + 101)

    unique_students = (
        student_df[["student_id", "learner_id", "current_module", "current_presentation"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    instructor_ids = [f"instructor_{i:04d}" for i in range(1, n_instructors + 1)]
    assigned_instructors = [instructor_ids[i % len(instructor_ids)] for i in range(len(unique_students))]
    rng.shuffle(assigned_instructors)
    unique_students["instructor_id"] = assigned_instructors

    # ── Cohort analytics (per recommended_module × presentation) ──
    cohort = student_df.groupby(["recommended_module", "recommended_presentation"], as_index=False).agg(
        cohort_avg_clicks=("implicit_clicks_14d", "mean"),
        cohort_watch_ratio=("implicit_video_watch_ratio_14d", "mean"),
        cohort_avg_forum_events=("implicit_forum_events_14d", "mean"),
        cohort_avg_quiz_score=("base_quiz_avg_score", "mean"),
        cohort_avg_dropout_rate=("dropout_risk", "mean"),
        cohort_size=("student_id", "nunique"),
    )
    max_clicks = max(1.0, cohort["cohort_avg_clicks"].max())
    max_forum   = max(1.0, cohort["cohort_avg_forum_events"].max())
    cohort["cohort_signal"] = (
        0.45 * (cohort["cohort_avg_clicks"] / max_clicks)
        + 0.30 * cohort["cohort_watch_ratio"]
        + 0.15 * (cohort["cohort_avg_forum_events"] / max_forum)
        + 0.10 * (cohort["cohort_avg_quiz_score"] / 100.0)
    )
    # Cohort engagement percentile
    cohort["cohort_engagement_percentile"] = (
        cohort["cohort_signal"].rank(pct=True).round(4)
    )

    merged = student_df.merge(
        unique_students[["student_id", "instructor_id"]],
        on="student_id",
        how="left",
    ).merge(
        cohort[[
            "recommended_module", "recommended_presentation",
            "cohort_signal", "cohort_avg_quiz_score", "cohort_avg_dropout_rate",
            "cohort_size", "cohort_engagement_percentile",
        ]],
        on=["recommended_module", "recommended_presentation"],
        how="left",
    )

    # ── Per-instructor stable profile fields (sampled once per instructor) ──
    instructor_profiles: Dict[str, Dict[str, Any]] = {}
    for iid in instructor_ids:
        arche = instructor_archetypes[hash(iid) % len(instructor_archetypes)]
        exp_min = _to_int(arche.get("experience_years_min"), 2)
        exp_max = _to_int(arche.get("experience_years_max"), max(exp_min + 1, 10))
        exp_years = int(np_rng.integers(exp_min, exp_max + 1))
        cohort_size = max(10, int(np_rng.normal(
            _to_float(arche.get("avg_cohort_size_mean"), 40), 10
        )))
        instructor_profiles[iid] = {
            "department":              arche.get("department", "Computing"),
            "teaching_style":          arche.get("teaching_style", "direct_instruction"),
            "experience_years":        exp_years,
            "instructor_avg_cohort_size": cohort_size,
            "instructor_intervention_intensity": round(
                float(np.clip(_to_float(arche.get("intervention_intensity"), 0.5), 0, 1)), 4
            ),
            "instructor_past_success_rate": round(float(np.clip(
                np_rng.normal(
                    _to_float(arche.get("success_rate_mean"), 0.7),
                    _to_float(arche.get("success_rate_std"), 0.06),
                ), 0.35, 0.97
            )), 4),
            "instructor_confidence_baseline": round(
                float(np.clip(_to_float(arche.get("confidence_baseline"), 0.75), 0.5, 0.98)), 4
            ),
        }

    rows: List[Dict[str, Any]] = []
    for instructor_id, g in merged.groupby("instructor_id"):
        arche = instructor_archetypes[hash(instructor_id) % len(instructor_archetypes)]
        profile = instructor_profiles.get(instructor_id, {})

        cohort_w = float(np.clip(_to_float(arche.get("cohort_signal_weight"), 0.5), 0, 1))
        student_w = float(np.clip(_to_float(arche.get("student_affinity_weight"), 0.5), 0, 1))
        if cohort_w + student_w == 0:
            cohort_w, student_w = 0.5, 0.5
        norm = cohort_w + student_w
        cohort_w /= norm
        student_w /= norm

        g = g.copy()
        g["assignment_priority"] = (
            cohort_w * g["cohort_signal"].fillna(0.0)
            + student_w * g["recommendation_score"].fillna(0.0)
            + 0.08 * (1.0 - g["dropout_risk"].fillna(0.0))
            + 0.05 * profile.get("instructor_intervention_intensity", 0.5) * g["dropout_risk"].fillna(0.0)
        )

        teaching_style = profile.get("teaching_style", "direct_instruction")

        for student_id, sg in g.groupby("student_id"):
            sg = sg.sort_values("assignment_priority", ascending=False).head(top_k)
            for rank, (_, rec) in enumerate(sg.iterrows(), start=1):
                dropout_risk      = float(rec.get("dropout_risk", 0))
                learning_momentum = float(rec.get("learning_momentum", 0.5))
                days_inactive     = float(rec.get("base_days_since_last_activity", 0))
                quiz_avg_score    = float(rec.get("base_quiz_avg_score", 50))
                missed_deadlines  = float(rec.get("base_missed_deadlines_count", 0))
                week_in_course    = float(rec.get("base_current_week_in_course", 1))
                assignment_sub    = float(rec.get("base_assignment_submission_rate", 0.5))
                cohort_size_val   = int(rec.get("cohort_size", 30))

                ivtype    = _intervention_type(arche, dropout_risk, learning_momentum)
                urgency   = _intervention_urgency(dropout_risk, days_inactive, missed_deadlines)
                ctype     = _content_type(ivtype, teaching_style)
                reason    = _priority_reason_tag(
                    dropout_risk, learning_momentum, days_inactive, quiz_avg_score, missed_deadlines
                )

                # Effort hours: remediation/challenge = more effort; enrichment/peer = moderate
                effort_base = {"remediation": 2.5, "challenge": 3.5, "enrichment": 1.5, "peer_review": 2.0}
                effort_hrs = round(float(np.clip(
                    np_rng.normal(effort_base.get(ivtype, 2.0), 0.5), 0.5, 6.0
                )), 2)

                # Predicted improvement: high success rate + high urgency + low momentum → bigger jump
                predicted_improvement = round(float(np.clip(
                    profile.get("instructor_past_success_rate", 0.7)
                    * (0.5 + 0.3 * dropout_risk + 0.2 * (1.0 - learning_momentum))
                    + np_rng.normal(0, 0.04),
                    0.1, 0.95
                )), 4)

                # Instructor confidence adjusts based on how well student profile matches intervention
                conf_adjust = (
                    -0.08 * dropout_risk
                    + 0.06 * assignment_sub
                    + 0.04 * (week_in_course / 12.0)
                    + np_rng.normal(0, 0.03)
                )
                instructor_confidence = round(float(np.clip(
                    profile.get("instructor_confidence_baseline", 0.75) + conf_adjust, 0.3, 0.99
                )), 4)

                # Student percentile within instructor's current cohort
                cohort_quiz_mean = float(rec.get("cohort_avg_quiz_score", 55.0))
                student_vs_cohort_quiz_delta = round(quiz_avg_score - cohort_quiz_mean, 3)

                # Peer interaction potential: forum posts + watch ratio → proxy for peer collab readiness
                peer_collab_readiness = round(float(np.clip(
                    0.5 * min(float(rec.get("implicit_forum_events_14d", 0)) / 15.0, 1.0)
                    + 0.3 * float(rec.get("implicit_video_watch_ratio_14d", 0.4))
                    + 0.2 * (1.0 - min(days_inactive / 21.0, 1.0)),
                    0.0, 1.0
                )), 4)

                # Risk trajectory label: is student getting worse (risk > 0.6 + low momentum)?
                if dropout_risk > 0.6 and learning_momentum < 0.4:
                    risk_trajectory = "deteriorating"
                elif dropout_risk < 0.3 and learning_momentum > 0.6:
                    risk_trajectory = "improving"
                elif dropout_risk > 0.4:
                    risk_trajectory = "at_risk_stable"
                else:
                    risk_trajectory = "stable"

                rows.append({
                    # ── IDs ──
                    "instructor_id":                   instructor_id,
                    "instructor_archetype":            arche.get("archetype_name", "balanced_mentor"),
                    "student_id":                      int(student_id),
                    "learner_id":                      rec.get("learner_id"),
                    # ── Instructor profile ──
                    "instructor_department":           profile.get("department", "Computing"),
                    "instructor_teaching_style":       teaching_style,
                    "instructor_experience_years":     profile.get("experience_years", 5),
                    "instructor_avg_cohort_size":      profile.get("instructor_avg_cohort_size", 40),
                    "instructor_intervention_intensity": profile.get("instructor_intervention_intensity", 0.5),
                    "instructor_past_success_rate":    profile.get("instructor_past_success_rate", 0.7),
                    "instructor_confidence_score":     instructor_confidence,
                    # ── Student context ──
                    "student_current_module":          rec.get("current_module"),
                    "student_current_presentation":    rec.get("current_presentation"),
                    "student_dropout_risk_score":      round(dropout_risk, 4),
                    "student_risk_trajectory":         risk_trajectory,
                    "student_learning_momentum":       round(learning_momentum, 4),
                    "student_quiz_avg_score":          round(quiz_avg_score, 3),
                    "student_assignment_submission_rate": round(assignment_sub, 4),
                    "student_missed_deadlines":        int(missed_deadlines),
                    "student_days_inactive":           int(days_inactive),
                    "student_week_in_course":          int(week_in_course),
                    "student_vs_cohort_quiz_delta":    student_vs_cohort_quiz_delta,
                    "student_peer_collab_readiness":   peer_collab_readiness,
                    # ── Cohort analytics ──
                    "recommended_module":              rec.get("recommended_module"),
                    "recommended_presentation":        rec.get("recommended_presentation"),
                    "cohort_signal_score":             round(float(rec.get("cohort_signal", 0.0)), 4),
                    "cohort_avg_quiz_score":           round(float(rec.get("cohort_avg_quiz_score", 0.0)), 3),
                    "cohort_avg_dropout_rate":         round(float(rec.get("cohort_avg_dropout_rate", 0.0)), 4),
                    "cohort_size":                     cohort_size_val,
                    "cohort_engagement_percentile":    round(float(rec.get("cohort_engagement_percentile", 0.5)), 4),
                    # ── Intervention ──
                    "student_affinity_score":          round(float(rec.get("recommendation_score", 0.0)), 4),
                    "assignment_priority":             round(float(rec.get("assignment_priority", 0.0)), 4),
                    "rank":                            rank,
                    "intervention_type":               ivtype,
                    "intervention_urgency":            urgency,
                    "recommended_content_type":        ctype,
                    "estimated_effort_hours":          effort_hrs,
                    # ── Outcome predictions ──
                    "predicted_improvement_score":     predicted_improvement,
                    "priority_reason_tag":             reason,
                    "recommendation_reason":           "cohort_plus_student_affinity_plus_instructor_style",
                })

    return pd.DataFrame(rows)


def _write_outputs(
    root: Path,
    student_df: pd.DataFrame,
    instructor_df: pd.DataFrame,
    model: str,
    seed: int,
) -> None:
    out_dir = root / "data" / "synthetic" / "recommendations"
    out_dir.mkdir(parents=True, exist_ok=True)

    student_path = out_dir / "student_recommendation_dataset.csv"
    instructor_path = out_dir / "instructor_recommendation_dataset.csv"
    report_path = out_dir / "generation_report.json"

    student_df.to_csv(student_path, index=False)
    instructor_df.to_csv(instructor_path, index=False)

    report = {
        "generator": "groq_archetype_blended_synth",
        "groq_model": model,
        "seed": seed,
        "student_rows": int(len(student_df)),
        "instructor_rows": int(len(instructor_df)),
        "student_unique_students": int(student_df["student_id"].nunique()),
        "instructor_unique_instructors": int(instructor_df["instructor_id"].nunique()),
        "output_files": {
            "student": str(student_path.relative_to(root)),
            "instructor": str(instructor_path.relative_to(root)),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote student dataset: {student_path} ({len(student_df)} rows)")
    print(f"Wrote instructor dataset: {instructor_path} ({len(instructor_df)} rows)")
    print(f"Wrote generation report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic recommendation datasets using Groq")
    parser.add_argument("--n-students", type=int, default=4000, help="Number of students to sample")
    parser.add_argument("--n-instructors", type=int, default=180, help="Number of synthetic instructors")
    parser.add_argument("--top-k", type=int, default=3, help="Recommendations per student")
    parser.add_argument("--archetypes", type=int, default=10, help="Number of Groq learner archetypes")
    parser.add_argument(
        "--instructor-archetypes",
        type=int,
        default=6,
        help="Number of Groq instructor archetypes",
    )
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL, help="Groq model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing. Set it in .env before running this script.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    merged, _, courses = _load_sources(root)
    modules = sorted(courses["code_module"].astype(str).unique().tolist())

    cfg = GroqConfig(api_key=api_key, model=args.model)
    student_archetypes = _build_student_archetypes(cfg, modules, args.archetypes)
    instructor_archetypes = _build_instructor_archetypes(cfg, args.instructor_archetypes)

    student_df = _build_student_dataset(
        merged=merged,
        courses=courses,
        archetypes=student_archetypes,
        n_students=args.n_students,
        top_k=args.top_k,
        seed=args.seed,
    )
    instructor_df = _build_instructor_dataset(
        student_df=student_df,
        instructor_archetypes=instructor_archetypes,
        n_instructors=args.n_instructors,
        top_k=args.top_k,
        seed=args.seed,
    )

    _write_outputs(root, student_df, instructor_df, args.model, args.seed)


if __name__ == "__main__":
    main()
