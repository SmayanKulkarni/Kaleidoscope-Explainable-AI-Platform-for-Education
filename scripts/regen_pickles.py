"""Regenerate data/train.pkl and data/test.pkl from data/learners.csv using
the current numpy/pandas environment, avoiding StringDtype pickle incompatibility."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OULAD_COLUMNS = [
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

ROOT_DIR     = Path(__file__).resolve().parents[1]
DATA_DIR     = ROOT_DIR / "data"
MODELS_DIR   = ROOT_DIR / "models"

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gbm_blob = pickle.load(open(MODELS_DIR / "gbm.pkl", "rb"))

MODEL_FEATURE_NAMES = gbm_blob["feature_names"]
N_MODEL_FEATURES    = len(MODEL_FEATURE_NAMES)
print(f"Model expects {N_MODEL_FEATURES} features: {MODEL_FEATURE_NAMES}")

df = pd.read_csv(DATA_DIR / "learners.csv")
print(f"Loaded learners.csv: {len(df)} rows")

X_oulad = df[OULAD_COLUMNS].values.astype(np.float32)

n_extra  = N_MODEL_FEATURES - len(OULAD_COLUMNS)
X_padded = np.hstack([X_oulad, np.zeros((len(df), n_extra), dtype=np.float32)])
print(f"X_train shape after padding: {X_padded.shape}")

y   = df["dropout_risk"].values.astype(np.int32)
ids = np.array(df["learner_id"].tolist(), dtype=object)

X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
    X_padded, y, ids, test_size=0.2, random_state=42, stratify=y
)

MAX_EXPLAINER_ROWS = 3000
if len(X_tr) > MAX_EXPLAINER_ROWS:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_tr), MAX_EXPLAINER_ROWS, replace=False)
    idx.sort()
    X_tr, y_tr, ids_tr = X_tr[idx], y_tr[idx], ids_tr[idx]
    print(f"Subsampled train split → {len(X_tr)} rows for explainer init")

for split, X_s, y_s, ids_s in [
    ("train", X_tr, y_tr, ids_tr),
    ("test",  X_te, y_te, ids_te),
]:
    path = DATA_DIR / f"{split}.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {"X": X_s, "y": y_s, "learner_ids": ids_s, "feature_names": MODEL_FEATURE_NAMES},
            f,
            protocol=4,
        )
    print(f"Saved {path.name}  n={len(y_s)}  dropout={y_s.mean() * 100:.1f}%")

print("\nVerifying reload...")
blob = pickle.load(open(DATA_DIR / "train.pkl", "rb"))
print(f"  X shape       : {blob['X'].shape}")
print(f"  feature_names : {blob['feature_names']}")
print(f"  ids[:3]       : {blob['learner_ids'][:3]}")
print("Done — pickles are compatible with current environment.")
