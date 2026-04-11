#!/usr/bin/env python3
"""
OULAD Feature Engineering Pipeline
====================================
Downloads and processes the Open University Learning Analytics Dataset (OULAD)
into the LEARNER_FEATURES schema required by the XAI Learning Recommendation System.

Synthetic Approximations (clearly documented):
  avg_session_duration_min  : sum_click/day × 2.5 min/click, capped at 180 min/day, averaged
  forum_posts_count         : sum_click on activity_type in {'forumng','ouwiki','forum'}
  video_completion_rate     : distinct 'oucontent' sites accessed / total available in course
  help_requests_count       : sum_click on {'questionnaire','resource'}; zero-rows filled with
                              Poisson noise correlated to engagement score (seed=42)

Usage:
  python backend/app/model/data_loader.py [--raw-dir PATH] [--data-dir PATH] [--seed N]

Output:
  data/learners.csv   — Full feature matrix with learner_id + all features + dropout_risk
  data/train.pkl      — Dict {X, y, learner_ids, feature_names}  (80% split, stratified)
  data/test.pkl       — Dict {X, y, learner_ids, feature_names}  (20% split, stratified)
"""

import argparse
import logging
import pickle
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"

KAGGLE_SLUG  = "anlgrbz/student-demographics-online-education-dataoulad"
OULAD_DIRECT = "https://analyse.kmi.open.ac.uk/open-dataset/download"  # no-auth GET → anonymisedData.zip

EXPECTED_FILES = [
    "studentVle.csv",
    "vle.csv",
    "studentAssessment.csv",
    "assessments.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "courses.csv",
]

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "dropout_risk"
ID_COLUMN     = "learner_id"
GROUP_KEY     = ["code_module", "code_presentation", "id_student"]


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def _kaggle_available() -> bool:
    try:
        r = subprocess.run(["kaggle", "--version"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _resolve_raw_file(raw_dir: Path, fname: str) -> Path:
    """Return path to fname, trying case-insensitive fallback."""
    direct = raw_dir / fname
    if direct.exists():
        return direct
    matches = list(raw_dir.glob(f"**/{fname}"))
    if not matches:
        matches = list(raw_dir.glob(f"**/{fname.lower()}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Expected file not found: {direct}\n"
        "Run the script again after placing OULAD CSVs in the --raw-dir directory."
    )


def _direct_download_oulad(raw_dir: Path) -> bool:
    """
    Download OULAD anonymisedData.zip directly from the Open University's
    public endpoint (no authentication required) and extract into raw_dir.
    """
    zip_path = raw_dir / "anonymisedData.zip"
    logger.info("Downloading OULAD from %s ...", OULAD_DIRECT)
    try:
        urllib.request.urlretrieve(OULAD_DIRECT, zip_path)
    except Exception as exc:
        logger.warning("Direct download failed: %s", exc)
        return False

    logger.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    zip_path.unlink(missing_ok=True)
    logger.info("Extraction complete.")
    return True


def download_oulad(raw_dir: Path) -> bool:
    """
    Ensure OULAD raw CSV files are present in raw_dir.
    Download priority:
      1. Files already present  → skip
      2. Direct GET from Open University endpoint (no auth)
      3. Kaggle CLI (requires ~/.kaggle/kaggle.json)
      4. Print manual instructions and return False
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    if all((raw_dir / f).exists() for f in EXPECTED_FILES):
        logger.info("Raw OULAD files already present in %s — skipping download.", raw_dir)
        return True

    # Primary: direct unauthenticated download from Open University
    if _direct_download_oulad(raw_dir):
        if all((raw_dir / f).exists() for f in EXPECTED_FILES):
            return True
        # Files might be nested in a subdirectory after extraction
        for sub in sorted(raw_dir.iterdir()):
            if sub.is_dir() and (sub / "studentVle.csv").exists():
                logger.info("CSVs found in subdirectory %s — moving to %s", sub, raw_dir)
                for csv in sub.glob("*.csv"):
                    csv.rename(raw_dir / csv.name)
                sub.rmdir()
                return True
        logger.warning("Direct download succeeded but expected CSVs not found. Trying Kaggle...")

    # Fallback: Kaggle CLI
    if _kaggle_available():
        logger.info("Trying Kaggle CLI for dataset '%s'...", KAGGLE_SLUG)
        result = subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", KAGGLE_SLUG,
             "-p", str(raw_dir),
             "--unzip"],
        )
        if result.returncode == 0:
            logger.info("Kaggle download complete.")
            return True
        logger.warning("Kaggle download failed (exit %d).", result.returncode)

    logger.error(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Automatic download failed — Manual steps:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Option A — Official OULAD (no login, ~44 MB ZIP):\n"
        "    curl -L -o /tmp/oulad.zip https://analyse.kmi.open.ac.uk/open-dataset/download\n"
        "    unzip /tmp/oulad.zip -d %s\n\n"
        "  Option B — Kaggle (~60 MB, requires API key):\n"
        "    pip install kaggle\n"
        "    # place ~/.kaggle/kaggle.json\n"
        "    kaggle datasets download -d %s -p %s --unzip\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
        raw_dir, KAGGLE_SLUG, raw_dir,
    )
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Raw file loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_raw(raw_dir: Path) -> dict:
    logger.info("Loading raw OULAD CSV files from %s ...", raw_dir)
    tables = {}
    for fname in EXPECTED_FILES:
        fpath = _resolve_raw_file(raw_dir, fname)
        key = fname.replace(".csv", "")
        tables[key] = pd.read_csv(fpath)
        logger.info("  %-30s  %d rows", fname, len(tables[key]))
    return tables


# ──────────────────────────────────────────────────────────────────────────────
# Individual feature builders
# ──────────────────────────────────────────────────────────────────────────────

def _feat_login_frequency_weekly(sv: pd.DataFrame) -> pd.DataFrame:
    """Average number of distinct active days per calendar week."""
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


def _feat_avg_session_duration_min(sv: pd.DataFrame) -> pd.DataFrame:
    """
    SYNTHETIC APPROXIMATION
    Proxy: sum_click per day × 2.5 min/click (cap 180 min/day), then averaged.
    Basis: Macfadyen & Dawson (2010) LMS interaction studies.
    """
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


def _feat_forum_posts_count(sv: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    """
    APPROXIMATE FROM VLE
    Counts total clicks on activity_type in {'forumng','ouwiki','oucollaborate'}.
    Represents combined forum reads + writes — treated as a scaled proxy for posts.
    """
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


def _feat_video_completion_rate(sv: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    """
    APPROXIMATE FROM VLE
    Proportion of distinct 'oucontent' resource sites accessed vs total available
    in that module-presentation. 'oucontent' is OULAD's primary content-delivery type
    (covers video lectures, readings, and structured learning objects).
    """
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


def _feat_quiz_stats(sa: pd.DataFrame, asmts: pd.DataFrame) -> pd.DataFrame:
    """
    quiz_avg_score + quiz_completion_rate from CMA (Computer Marked Assessment) rows.
    CMA = auto-graded quizzes in OULAD.
    """
    cma = asmts[asmts["assessment_type"] == "CMA"].copy()
    total_cma = (
        cma.groupby(["code_module", "code_presentation"])["id_assessment"]
        .nunique()
        .reset_index(name="total_cma")
    )
    subs = sa.merge(
        cma[["id_assessment", "code_module", "code_presentation"]],
        on="id_assessment",
        how="inner",
    )
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


def _feat_assignment_stats(sa: pd.DataFrame, asmts: pd.DataFrame) -> pd.DataFrame:
    """
    assignment_submission_rate + missed_deadlines_count from TMA (Tutor Marked Assessment) rows.
    Missed = not submitted + submitted after the due date.
    """
    tma = asmts[asmts["assessment_type"] == "TMA"].copy()
    total_tma = (
        tma.groupby(["code_module", "code_presentation"])["id_assessment"]
        .nunique()
        .reset_index(name="total_tma")
    )
    subs = sa.merge(
        tma[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="inner",
    )
    subs["is_late"] = (
        subs["date_submitted"].notna()
        & (subs["date_submitted"] > subs["date"])
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


def _feat_days_since_last_activity(sv: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    """
    Days between last VLE interaction and end of the module presentation.
    High value = learner disengaged early.
    Returns both days_since_last_activity and last_active_day (needed by current_week).
    """
    last = (
        sv.groupby(GROUP_KEY)["date"]
        .max()
        .reset_index(name="last_active_day")
    )
    merged = last.merge(
        courses[["code_module", "code_presentation", "module_presentation_length"]],
        on=["code_module", "code_presentation"],
        how="left",
    )
    merged["days_since_last_activity"] = (
        merged["module_presentation_length"] - merged["last_active_day"]
    ).clip(lower=0).astype(int)
    return merged[GROUP_KEY + ["days_since_last_activity", "last_active_day"]]


def _feat_current_week(last_act_df: pd.DataFrame) -> pd.DataFrame:
    """Derive current_week_in_course from last_active_day (1-indexed)."""
    df = last_act_df[GROUP_KEY + ["last_active_day"]].copy()
    df["current_week_in_course"] = ((df["last_active_day"] / 7) + 1).clip(lower=1).astype(int)
    return df[GROUP_KEY + ["current_week_in_course"]]


def _feat_help_requests_count(sv: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    """
    APPROXIMATE FROM VLE
    sum_click on activity_type in {'questionnaire','resource'}.
    These types represent learners explicitly seeking support materials / feedback forms.
    """
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


def _synthetic_fill_help(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    SYNTHETIC GAP FILL
    Students whose help_requests_count == 0 (no questionnaire/resource clicks) receive
    a Poisson-sampled count correlated positively with engagement.

    λ = clip((login_freq/7 + quiz_completion_rate) / 2 × 8 + 0.5,  0.1, 15)

    Rationale: engaged learners are more likely to seek help resources.
    Clearly synthetic — used only where direct VLE proxy is absent.
    """
    zero_mask = df["help_requests_count"] == 0
    n_zero = zero_mask.sum()
    if n_zero == 0:
        return df
    logger.info(
        "  Synthetic gap fill: %d students with help_requests_count=0 → Poisson fill", n_zero
    )
    engagement = (
        (df.loc[zero_mask, "login_frequency_weekly"].fillna(0) / 7.0)
        + df.loc[zero_mask, "quiz_completion_rate"].fillna(0)
    ) / 2.0
    lam = (engagement * 8.0 + 0.5).clip(0.1, 15.0)
    df.loc[zero_mask, "help_requests_count"] = rng.poisson(lam=lam.values).astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(raw_dir: Path, seed: int = 42) -> pd.DataFrame:
    """
    End-to-end OULAD → LEARNER_FEATURES engineering.

    Returns a DataFrame with columns:
      learner_id, code_module, code_presentation,
      <FEATURE_COLUMNS>, dropout_risk
    """
    tables = _load_raw(raw_dir)

    sv    = tables["studentVle"]
    vle   = tables["vle"]
    sa    = tables["studentAssessment"]
    asmts = tables["assessments"]
    si    = tables["studentInfo"]
    crs   = tables["courses"]

    logger.info("Engineering features...")

    login_freq   = _feat_login_frequency_weekly(sv)
    avg_session  = _feat_avg_session_duration_min(sv)
    forum_cnt    = _feat_forum_posts_count(sv, vle)
    video_rate   = _feat_video_completion_rate(sv, vle)
    quiz_stats   = _feat_quiz_stats(sa, asmts)
    asgn_stats   = _feat_assignment_stats(sa, asmts)
    last_act     = _feat_days_since_last_activity(sv, crs)
    current_wk   = _feat_current_week(last_act)
    help_cnt     = _feat_help_requests_count(sv, vle)

    # Base: one row per student-course, target label
    base = si[GROUP_KEY + ["num_of_prev_attempts", "final_result"]].copy()
    base.rename(columns={"num_of_prev_attempts": "prior_course_completions"}, inplace=True)
    base["dropout_risk"] = (base["final_result"] == "Withdrawn").astype(int)

    logger.info("Merging %d feature DataFrames...", 9)
    df = base.copy()
    for feat_df, cols in [
        (login_freq,                                               ["login_frequency_weekly"]),
        (avg_session,                                              ["avg_session_duration_min"]),
        (forum_cnt,                                                ["forum_posts_count"]),
        (video_rate,                                               ["video_completion_rate"]),
        (quiz_stats,                                               ["quiz_avg_score", "quiz_completion_rate"]),
        (asgn_stats,                                               ["assignment_submission_rate", "missed_deadlines_count"]),
        (last_act[GROUP_KEY + ["days_since_last_activity"]],       ["days_since_last_activity"]),
        (current_wk,                                               ["current_week_in_course"]),
        (help_cnt,                                                 ["help_requests_count"]),
    ]:
        df = df.merge(feat_df[GROUP_KEY + cols], on=GROUP_KEY, how="left")

    # Fill missings
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
        "days_since_last_activity": 0,
        "current_week_in_course": 1,
        "prior_course_completions": 0,
    }
    for col, val in float_fills.items():
        df[col] = df[col].fillna(val)
    for col, val in int_fills.items():
        df[col] = df[col].fillna(val).astype(int)

    # Synthetic gap fill for help_requests_count
    rng = np.random.default_rng(seed)
    df  = _synthetic_fill_help(df, rng)

    # Stable learner IDs
    df = df.reset_index(drop=True)
    df[ID_COLUMN] = [f"learner_{str(i + 1).zfill(5)}" for i in range(len(df))]

    dropout_rate = df[TARGET_COLUMN].mean() * 100
    logger.info(
        "Feature engineering complete: %d learners | dropout_rate=%.1f%%",
        len(df), dropout_rate,
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Output serialisation
# ──────────────────────────────────────────────────────────────────────────────

def save_outputs(
    df: pd.DataFrame,
    data_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    # Full CSV
    csv_path = data_dir / "learners.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved  %s  (%d rows, %d columns)", csv_path, len(df), len(df.columns))

    # Train / test splits
    X   = df[FEATURE_COLUMNS].values.astype(float)
    y   = df[TARGET_COLUMN].values.astype(int)
    ids = df[ID_COLUMN].values

    X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
        X, y, ids,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    for split, X_s, y_s, ids_s in [
        ("train", X_tr, y_tr, ids_tr),
        ("test",  X_te, y_te, ids_te),
    ]:
        payload = {
            "X":             X_s,
            "y":             y_s,
            "learner_ids":   ids_s,
            "feature_names": FEATURE_COLUMNS,
        }
        pkl_path = data_dir / f"{split}.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info(
            "Saved  %-20s  n=%-6d  dropout_rate=%.1f%%",
            pkl_path.name, len(y_s), y_s.mean() * 100,
        )

    logger.info("All outputs written to %s", data_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OULAD and engineer LEARNER_FEATURES schema."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory with (or to receive) OULAD raw CSVs. Default: {RAW_DIR}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory for learners.csv / *.pkl. Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for synthetic gap fills."
    )
    args = parser.parse_args()

    if not download_oulad(args.raw_dir):
        sys.exit(1)

    df = engineer_features(args.raw_dir, seed=args.seed)
    save_outputs(df, args.data_dir, seed=args.seed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
