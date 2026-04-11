#!/usr/bin/env python3
"""
Temporal Feature Engineering + Monte Carlo Forward Simulator
=============================================================
Produces cumulative temporal snapshots of learner features at multiple
time cutoffs using real OULAD data, plus a Monte Carlo simulator that
projects plausible future feature trajectories.

Temporal Snapshots
------------------
For each student-course enrolment, features are computed using only data
up to day T (cumulative window). Default cutoff weeks: 2, 4, 6, 8, 10, 12.
This yields up to 6 snapshots per student, showing how their engagement
profile evolved over time.

Monte Carlo Forward Simulator
-----------------------------
Given a student's feature vector at week T, simulate N plausible futures
by sampling feature deltas from the empirical transition distributions of
similar students between consecutive snapshot windows. Returns a distribution
of model-predicted outcomes.

Usage:
  python backend/app/model/temporal_builder.py [--raw-dir PATH] [--data-dir PATH]

Output:
  data/temporal/snapshots.pkl    — Dict {snapshots: DataFrame, cutoff_weeks: list}
  data/temporal/transitions.pkl  — Empirical transition matrices for MC simulator
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import (
    FEATURE_COLUMNS,
    GROUP_KEY,
    ID_COLUMN,
    TARGET_COLUMN,
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DIR,
    _load_raw,
    _synthetic_fill_help,
    download_oulad,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default snapshot cutoff points (weeks from course start)
DEFAULT_CUTOFF_WEEKS = [2, 4, 6, 8, 10, 12]

TEMPORAL_DIR = DATA_DIR / "temporal"


# ──────────────────────────────────────────────────────────────────────────────
# Temporal feature builders (cumulative up to cutoff_day)
# ──────────────────────────────────────────────────────────────────────────────

def _temporal_login_frequency_weekly(sv: pd.DataFrame) -> pd.DataFrame:
    """Average distinct active days per week (on filtered VLE data)."""
    df = sv[GROUP_KEY + ["date"]].copy()
    df["week"] = df["date"] // 7
    weekly = (
        df.groupby(GROUP_KEY + ["week"])["date"]
        .nunique()
        .reset_index(name="active_days")
    )
    return (
        weekly.groupby(GROUP_KEY)["active_days"]
        .mean()
        .reset_index(name="login_frequency_weekly")
    )


def _temporal_avg_session_duration_min(sv: pd.DataFrame) -> pd.DataFrame:
    """Proxy: sum_click/day × 2.5 min, capped 180 min/day, averaged."""
    daily = (
        sv.groupby(GROUP_KEY + ["date"])["sum_click"]
        .sum()
        .reset_index()
    )
    daily["est_min"] = (daily["sum_click"] * 2.5).clip(upper=180.0)
    return (
        daily.groupby(GROUP_KEY)["est_min"]
        .mean()
        .reset_index(name="avg_session_duration_min")
    )


def _temporal_forum_posts_count(sv: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    """Clicks on forumng/ouwiki/oucollaborate (filtered VLE data)."""
    forum_types = {"forumng", "ouwiki", "oucollaborate"}
    typed = sv.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )
    return (
        typed[typed["activity_type"].isin(forum_types)]
        .groupby(GROUP_KEY)["sum_click"]
        .sum()
        .reset_index(name="forum_posts_count")
    )


def _temporal_video_completion_rate(
    sv: pd.DataFrame, vle: pd.DataFrame
) -> pd.DataFrame:
    """Proportion of oucontent/subpage sites accessed vs total available."""
    content_type = {"oucontent", "subpage"}
    typed = sv.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )
    total = (
        vle[vle["activity_type"].isin(content_type)]
        .groupby(["code_module", "code_presentation"])["id_site"]
        .nunique()
        .reset_index(name="total_content_sites")
    )
    accessed = (
        typed[typed["activity_type"].isin(content_type)]
        .groupby(GROUP_KEY)["id_site"]
        .nunique()
        .reset_index(name="accessed_content_sites")
    )
    merged = accessed.merge(total, on=["code_module", "code_presentation"], how="left")
    merged["video_completion_rate"] = (
        merged["accessed_content_sites"]
        / merged["total_content_sites"].replace(0, np.nan)
    ).clip(0.0, 1.0)
    return merged[GROUP_KEY + ["video_completion_rate"]]


def _temporal_quiz_stats(
    sa: pd.DataFrame, asmts: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    """
    quiz_avg_score + quiz_completion_rate from CMA rows.
    Only counts CMAs with due date <= cutoff_day as the denominator,
    and only submissions with date_submitted <= cutoff_day.
    """
    cma = asmts[asmts["assessment_type"] == "CMA"].copy()
    # Only count CMAs due by cutoff
    cma_due = cma[cma["date"].fillna(9999) <= cutoff_day]
    total_cma = (
        cma_due.groupby(["code_module", "code_presentation"])["id_assessment"]
        .nunique()
        .reset_index(name="total_cma")
    )
    # Filter submissions by cutoff
    sa_cut = sa[sa["date_submitted"].fillna(9999) <= cutoff_day]
    subs = sa_cut.merge(
        cma_due[["id_assessment", "code_module", "code_presentation"]],
        on="id_assessment",
        how="inner",
    )
    if subs.empty:
        return pd.DataFrame(columns=GROUP_KEY + ["quiz_avg_score", "quiz_completion_rate"])
    stats = (
        subs.groupby(GROUP_KEY)
        .agg(quiz_avg_score=("score", "mean"), submitted_cma=("id_assessment", "nunique"))
        .reset_index()
    )
    stats = stats.merge(total_cma, on=["code_module", "code_presentation"], how="left")
    stats["quiz_completion_rate"] = (
        stats["submitted_cma"] / stats["total_cma"].replace(0, np.nan)
    ).clip(0.0, 1.0)
    stats["quiz_avg_score"] = stats["quiz_avg_score"].fillna(0.0)
    return stats[GROUP_KEY + ["quiz_avg_score", "quiz_completion_rate"]]


def _temporal_assignment_stats(
    sa: pd.DataFrame, asmts: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    """
    assignment_submission_rate + missed_deadlines_count from TMA rows.
    Only counts TMAs with due date <= cutoff_day.
    """
    tma = asmts[asmts["assessment_type"] == "TMA"].copy()
    tma_due = tma[tma["date"].fillna(9999) <= cutoff_day]
    total_tma = (
        tma_due.groupby(["code_module", "code_presentation"])["id_assessment"]
        .nunique()
        .reset_index(name="total_tma")
    )
    sa_cut = sa[sa["date_submitted"].fillna(9999) <= cutoff_day]
    subs = sa_cut.merge(
        tma_due[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="inner",
    )
    if subs.empty:
        return pd.DataFrame(
            columns=GROUP_KEY + ["assignment_submission_rate", "missed_deadlines_count"]
        )
    subs["is_late"] = (
        subs["date_submitted"].notna() & (subs["date_submitted"] > subs["date"])
    )
    stats = (
        subs.groupby(GROUP_KEY)
        .agg(submitted_tma=("id_assessment", "nunique"), late_count=("is_late", "sum"))
        .reset_index()
    )
    stats = stats.merge(total_tma, on=["code_module", "code_presentation"], how="left")
    stats["assignment_submission_rate"] = (
        stats["submitted_tma"] / stats["total_tma"].replace(0, np.nan)
    ).clip(0.0, 1.0)
    stats["missed_deadlines_count"] = (
        (stats["total_tma"] - stats["submitted_tma"]).clip(lower=0) + stats["late_count"]
    ).astype(int)
    return stats[GROUP_KEY + ["assignment_submission_rate", "missed_deadlines_count"]]


def _temporal_days_since_last_activity(
    sv: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    """
    Days between last VLE interaction and the cutoff day.
    Also returns last_active_day for current_week computation.
    """
    last = (
        sv.groupby(GROUP_KEY)["date"]
        .max()
        .reset_index(name="last_active_day")
    )
    last["days_since_last_activity"] = (
        cutoff_day - last["last_active_day"]
    ).clip(lower=0).astype(int)
    return last[GROUP_KEY + ["days_since_last_activity", "last_active_day"]]


def _temporal_help_requests_count(
    sv: pd.DataFrame, vle: pd.DataFrame
) -> pd.DataFrame:
    """Clicks on questionnaire/resource/glossary (filtered VLE data)."""
    help_types = {"questionnaire", "resource", "glossary"}
    typed = sv.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )
    return (
        typed[typed["activity_type"].isin(help_types)]
        .groupby(GROUP_KEY)["sum_click"]
        .sum()
        .reset_index(name="help_requests_count")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Single-snapshot builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_snapshot(
    tables: dict,
    cutoff_day: int,
    cutoff_week: int,
    student_info_base: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute all LEARNER_FEATURES using only data up to cutoff_day.
    Returns DataFrame with GROUP_KEY + FEATURE_COLUMNS + [snapshot_week, dropout_risk].
    """
    sv_full = tables["studentVle"]
    vle = tables["vle"]
    sa_full = tables["studentAssessment"]
    asmts = tables["assessments"]

    # Filter VLE interactions to <= cutoff_day
    sv = sv_full[sv_full["date"] <= cutoff_day].copy()
    if sv.empty:
        logger.warning("  Week %d (day %d): no VLE data — skipping.", cutoff_week, cutoff_day)
        return pd.DataFrame()

    # Build features on filtered data
    login_freq = _temporal_login_frequency_weekly(sv)
    avg_session = _temporal_avg_session_duration_min(sv)
    forum_cnt = _temporal_forum_posts_count(sv, vle)
    video_rate = _temporal_video_completion_rate(sv, vle)
    quiz_stats = _temporal_quiz_stats(sa_full, asmts, cutoff_day)
    asgn_stats = _temporal_assignment_stats(sa_full, asmts, cutoff_day)
    last_act = _temporal_days_since_last_activity(sv, cutoff_day)
    help_cnt = _temporal_help_requests_count(sv, vle)

    # current_week is fixed for this snapshot
    current_wk = last_act[GROUP_KEY].copy()
    current_wk["current_week_in_course"] = cutoff_week

    # Merge onto base
    df = student_info_base.copy()
    for feat_df, cols in [
        (login_freq,                                         ["login_frequency_weekly"]),
        (avg_session,                                        ["avg_session_duration_min"]),
        (forum_cnt,                                          ["forum_posts_count"]),
        (video_rate,                                         ["video_completion_rate"]),
        (quiz_stats,                                         ["quiz_avg_score", "quiz_completion_rate"]),
        (asgn_stats,                                         ["assignment_submission_rate", "missed_deadlines_count"]),
        (last_act[GROUP_KEY + ["days_since_last_activity"]], ["days_since_last_activity"]),
        (current_wk,                                         ["current_week_in_course"]),
        (help_cnt,                                           ["help_requests_count"]),
    ]:
        if feat_df.empty:
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
        else:
            df = df.merge(feat_df[GROUP_KEY + cols], on=GROUP_KEY, how="left")

    # Fill NaNs
    float_fills = {
        "login_frequency_weekly": 0.0,
        "avg_session_duration_min": 0.0,
        "video_completion_rate": 0.0,
        "quiz_avg_score": 0.0,
        "quiz_completion_rate": 0.0,
        "assignment_submission_rate": 0.0,
    }
    int_fills = {
        "forum_posts_count": 0,
        "help_requests_count": 0,
        "missed_deadlines_count": 0,
        "days_since_last_activity": cutoff_day,  # no activity → inactive since start
        "current_week_in_course": cutoff_week,
        "prior_course_completions": 0,
    }
    for col, val in float_fills.items():
        df[col] = df[col].fillna(val)
    for col, val in int_fills.items():
        df[col] = df[col].fillna(val).astype(int)

    # Synthetic help fill
    rng = np.random.default_rng(seed + cutoff_week)
    df = _synthetic_fill_help(df, rng)

    df["snapshot_week"] = cutoff_week
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Full temporal pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_temporal_snapshots(
    raw_dir: Path,
    cutoff_weeks: Optional[list] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build cumulative temporal snapshots for all students at each cutoff week.

    Returns a DataFrame with columns:
      learner_id, snapshot_week, <FEATURE_COLUMNS>, dropout_risk
    """
    if cutoff_weeks is None:
        cutoff_weeks = DEFAULT_CUTOFF_WEEKS

    tables = _load_raw(raw_dir)
    si = tables["studentInfo"]

    # Base: one row per student-course
    base = si[GROUP_KEY + ["num_of_prev_attempts", "final_result"]].copy()
    base.rename(columns={"num_of_prev_attempts": "prior_course_completions"}, inplace=True)
    base[TARGET_COLUMN] = (base["final_result"] == "Withdrawn").astype(int)

    all_snapshots = []
    for week in cutoff_weeks:
        cutoff_day = week * 7
        logger.info("Building snapshot @ week %d (day <= %d) ...", week, cutoff_day)
        snap = _build_snapshot(tables, cutoff_day, week, base, seed=seed)
        if not snap.empty:
            all_snapshots.append(snap)
            logger.info(
                "  Week %d: %d learners with VLE activity",
                week, len(snap),
            )

    combined = pd.concat(all_snapshots, ignore_index=True)

    # Assign stable learner IDs (consistent with static pipeline by GROUP_KEY hash)
    combined[ID_COLUMN] = (
        combined["code_module"].astype(str) + "_"
        + combined["code_presentation"].astype(str) + "_"
        + combined["id_student"].astype(str)
    )

    # Sort for clean output
    combined = combined.sort_values([ID_COLUMN, "snapshot_week"]).reset_index(drop=True)

    # Keep only needed columns
    keep_cols = [ID_COLUMN, "snapshot_week"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    extra = [c for c in GROUP_KEY if c in combined.columns]
    combined = combined[extra + keep_cols]

    logger.info(
        "Temporal snapshots complete: %d total rows (%d students × %d weeks max)",
        len(combined),
        combined[ID_COLUMN].nunique(),
        len(cutoff_weeks),
    )
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Transition matrix builder (for Monte Carlo)
# ──────────────────────────────────────────────────────────────────────────────

def compute_transition_distributions(
    snapshots: pd.DataFrame,
) -> dict:
    """
    Compute empirical feature-delta distributions between consecutive snapshots.

    For each pair of consecutive cutoff weeks (t → t+Δ), compute the change in
    each feature for every student who has both snapshots. Group by risk bucket
    (high/low based on the source snapshot's feature values).

    Returns:
        dict with keys:
          - "deltas": DataFrame of per-student per-transition feature deltas
          - "summary": dict of {(from_week, to_week): {feature: {mean, std, q25, q75}}}
          - "cutoff_weeks": sorted list of weeks present
    """
    weeks = sorted(snapshots["snapshot_week"].unique())
    all_deltas = []

    for i in range(len(weeks) - 1):
        w_from, w_to = weeks[i], weeks[i + 1]
        snap_from = snapshots[snapshots["snapshot_week"] == w_from].set_index(ID_COLUMN)
        snap_to = snapshots[snapshots["snapshot_week"] == w_to].set_index(ID_COLUMN)

        # Students present in both snapshots
        common_ids = snap_from.index.intersection(snap_to.index)
        if len(common_ids) == 0:
            continue

        delta = snap_to.loc[common_ids, FEATURE_COLUMNS] - snap_from.loc[common_ids, FEATURE_COLUMNS]
        delta["from_week"] = w_from
        delta["to_week"] = w_to
        delta[TARGET_COLUMN] = snap_to.loc[common_ids, TARGET_COLUMN]
        delta.index.name = ID_COLUMN
        all_deltas.append(delta.reset_index())

    if not all_deltas:
        logger.warning("No valid transitions found.")
        return {"deltas": pd.DataFrame(), "summary": {}, "cutoff_weeks": weeks}

    deltas_df = pd.concat(all_deltas, ignore_index=True)

    # Compute summary statistics per transition
    summary = {}
    for (wf, wt), grp in deltas_df.groupby(["from_week", "to_week"]):
        summary[(wf, wt)] = {}
        for feat in FEATURE_COLUMNS:
            vals = grp[feat].dropna()
            summary[(wf, wt)][feat] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
            }

    logger.info(
        "Transition distributions: %d transitions across %d week-pairs, %d students",
        len(deltas_df),
        len(summary),
        deltas_df[ID_COLUMN].nunique(),
    )
    return {"deltas": deltas_df, "summary": summary, "cutoff_weeks": weeks}


# ──────────────────────────────────────────────────────────────────────────────
# Monte Carlo Forward Simulator
# ──────────────────────────────────────────────────────────────────────────────

class MonteCarloSimulator:
    """
    Projects plausible future feature trajectories from a student's current state
    using empirical transition distributions from real OULAD temporal data.

    Usage:
        sim = MonteCarloSimulator(transitions_data)
        results = sim.simulate(
            current_features={...},
            current_week=6,
            target_week=12,
            n_simulations=1000,
            model=trained_model,  # optional: get outcome distributions
        )
    """

    # Feature value bounds (physical limits)
    FEATURE_BOUNDS = {
        "login_frequency_weekly":     (0.0, 7.0),
        "avg_session_duration_min":   (0.0, 180.0),
        "forum_posts_count":          (0, None),
        "video_completion_rate":      (0.0, 1.0),
        "quiz_avg_score":             (0.0, 100.0),
        "quiz_completion_rate":       (0.0, 1.0),
        "assignment_submission_rate": (0.0, 1.0),
        "days_since_last_activity":   (0, None),
        "prior_course_completions":   (0, None),
        "current_week_in_course":     (1, None),
        "missed_deadlines_count":     (0, None),
        "help_requests_count":        (0, None),
    }

    def __init__(self, transitions: dict, seed: int = 42):
        """
        Args:
            transitions: Output of compute_transition_distributions()
            seed: RNG seed for reproducibility
        """
        self.deltas = transitions["deltas"]
        self.summary = transitions["summary"]
        self.cutoff_weeks = transitions["cutoff_weeks"]
        self.rng = np.random.default_rng(seed)

    def _get_closest_transition(self, from_week: int, to_week: int) -> tuple:
        """Find the closest available transition pair."""
        available = list(self.summary.keys())
        if not available:
            raise ValueError("No transition data available.")

        # Exact match
        if (from_week, to_week) in available:
            return (from_week, to_week)

        # Find closest by week gap similarity
        target_gap = to_week - from_week
        best = min(available, key=lambda x: abs((x[1] - x[0]) - target_gap))
        return best

    def _sample_delta(self, transition_key: tuple) -> dict:
        """Sample a feature delta vector from the empirical distribution."""
        mask = (
            (self.deltas["from_week"] == transition_key[0])
            & (self.deltas["to_week"] == transition_key[1])
        )
        pool = self.deltas[mask]
        if pool.empty:
            # Fallback: sample from summary statistics (Gaussian)
            stats = self.summary[transition_key]
            return {
                feat: self.rng.normal(s["mean"], max(s["std"], 0.01))
                for feat, s in stats.items()
            }

        # Sample a random row from real student deltas
        row = pool.sample(n=1, random_state=int(self.rng.integers(0, 2**31))).iloc[0]
        return {feat: row[feat] for feat in FEATURE_COLUMNS}

    def _clip_features(self, features: dict) -> dict:
        """Clip features to valid physical bounds."""
        clipped = {}
        for feat, val in features.items():
            bounds = self.FEATURE_BOUNDS.get(feat, (None, None))
            lo, hi = bounds
            v = val
            if lo is not None:
                v = max(v, lo)
            if hi is not None:
                v = min(v, hi)
            clipped[feat] = v
        return clipped

    def simulate(
        self,
        current_features: dict,
        current_week: int,
        target_week: int,
        n_simulations: int = 1000,
        model=None,
    ) -> dict:
        """
        Run Monte Carlo forward simulation.

        Args:
            current_features: Dict of {feature_name: value} at current_week
            current_week: Current snapshot week
            target_week: Target week to project to
            n_simulations: Number of Monte Carlo trajectories
            model: Optional sklearn model with predict_proba() for outcome distribution

        Returns:
            dict with:
              - "trajectories": list of N dicts (projected feature vectors at target_week)
              - "feature_distributions": {feature: {mean, std, q10, q50, q90}}
              - "outcome_distribution": {dropout_prob_mean, dropout_prob_std, ...} (if model)
              - "current_week": int
              - "target_week": int
              - "n_simulations": int
        """
        if current_week >= target_week:
            raise ValueError(f"current_week ({current_week}) must be < target_week ({target_week})")

        # Determine step sequence through available transitions
        steps = []
        week_cursor = current_week
        step_size = 2  # default 2-week steps
        while week_cursor < target_week:
            next_week = min(week_cursor + step_size, target_week)
            steps.append((week_cursor, next_week))
            week_cursor = next_week

        # Run simulations
        trajectories = []
        for _ in range(n_simulations):
            state = dict(current_features)
            for from_w, to_w in steps:
                tk = self._get_closest_transition(from_w, to_w)
                delta = self._sample_delta(tk)
                # Apply delta
                for feat in FEATURE_COLUMNS:
                    state[feat] = state.get(feat, 0) + delta.get(feat, 0)
                state = self._clip_features(state)
                state["current_week_in_course"] = to_w
            trajectories.append(dict(state))

        # Compute feature distributions
        feat_dist = {}
        for feat in FEATURE_COLUMNS:
            vals = np.array([t[feat] for t in trajectories])
            feat_dist[feat] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "q10": float(np.percentile(vals, 10)),
                "q50": float(np.percentile(vals, 50)),
                "q90": float(np.percentile(vals, 90)),
            }

        result = {
            "trajectories": trajectories,
            "feature_distributions": feat_dist,
            "current_week": current_week,
            "target_week": target_week,
            "n_simulations": n_simulations,
        }

        # If model provided, compute outcome distribution
        if model is not None:
            X_sim = np.array([[t[f] for f in FEATURE_COLUMNS] for t in trajectories])
            probs = model.predict_proba(X_sim)[:, 1]
            result["outcome_distribution"] = {
                "dropout_prob_mean": float(np.mean(probs)),
                "dropout_prob_std": float(np.std(probs)),
                "dropout_prob_q10": float(np.percentile(probs, 10)),
                "dropout_prob_q50": float(np.percentile(probs, 50)),
                "dropout_prob_q90": float(np.percentile(probs, 90)),
                "dropout_rate": float(np.mean(probs > 0.5)),
            }

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────────────────────────────────────

def save_temporal_outputs(
    snapshots: pd.DataFrame,
    transitions: dict,
    output_dir: Path,
) -> None:
    """Save snapshots and transition data to pickle files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    snap_path = output_dir / "snapshots.pkl"
    with open(snap_path, "wb") as f:
        pickle.dump(
            {
                "snapshots": snapshots,
                "cutoff_weeks": sorted(snapshots["snapshot_week"].unique().tolist()),
                "feature_columns": FEATURE_COLUMNS,
                "n_students": int(snapshots[ID_COLUMN].nunique()),
            },
            f,
        )
    logger.info("Saved %s  (%d rows)", snap_path, len(snapshots))

    trans_path = output_dir / "transitions.pkl"
    with open(trans_path, "wb") as f:
        pickle.dump(transitions, f)
    logger.info("Saved %s  (%d transition rows)", trans_path, len(transitions["deltas"]))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build temporal snapshots and MC transition data from OULAD."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help=f"Directory with OULAD raw CSVs. Default: {RAW_DIR}",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help=f"Parent output directory. Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed.",
    )
    parser.add_argument(
        "--weeks", type=str, default="2,4,6,8,10,12",
        help="Comma-separated cutoff weeks. Default: 2,4,6,8,10,12",
    )
    args = parser.parse_args()

    cutoff_weeks = [int(w.strip()) for w in args.weeks.split(",")]

    if not download_oulad(args.raw_dir):
        import sys
        sys.exit(1)

    # Build temporal snapshots
    snapshots = build_temporal_snapshots(args.raw_dir, cutoff_weeks, seed=args.seed)

    # Compute transition distributions for MC simulator
    transitions = compute_transition_distributions(snapshots)

    # Save
    output_dir = args.data_dir / "temporal"
    save_temporal_outputs(snapshots, transitions, output_dir)

    # Quick summary stats
    logger.info("\n── Summary ──")
    for wk in cutoff_weeks:
        n = len(snapshots[snapshots["snapshot_week"] == wk])
        dr = snapshots[snapshots["snapshot_week"] == wk][TARGET_COLUMN].mean() * 100
        logger.info("  Week %2d: %6d learners | dropout_rate=%.1f%%", wk, n, dr)

    logger.info("Done.")


if __name__ == "__main__":
    main()
