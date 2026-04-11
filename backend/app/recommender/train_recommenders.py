#!/usr/bin/env python3
"""Train recommendation engines (student + instructor) with LambdaMART."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.mlops.mlflow_config import configure_mlflow
from backend.app.recommender.ranking_metrics import map_at_k, ndcg_at_k, recall_at_k

RECO_DIR = PROJECT_ROOT / "data" / "synthetic" / "recommendations"
MODELS_DIR = PROJECT_ROOT / "models" / "recommenders"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    group_train: List[int]
    X_test: pd.DataFrame
    y_test: np.ndarray
    y_test_bin: np.ndarray
    group_test: List[int]
    feature_columns: List[str]
    metadata: Dict[str, object]


def _cast_and_encode(df: pd.DataFrame, fit_maps: Dict[str, Dict[str, int]] | None = None):
    out = df.copy()
    maps = {} if fit_maps is None else fit_maps

    for col in out.columns:
        if out[col].dtype == "O":
            out[col] = out[col].fillna("unknown").astype(str)
            if fit_maps is None:
                vals = sorted(out[col].unique().tolist())
                mapping = {v: i for i, v in enumerate(vals)}
                maps[col] = mapping
            mapping = maps.get(col, {"unknown": 0})
            out[col] = out[col].map(mapping).fillna(-1).astype(int)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out, maps


def _prepare_data(
    df: pd.DataFrame,
    query_col: str,
    score_col: str,
    drop_cols: List[str],
    seed: int,
) -> PreparedData:
    working = df.copy()
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce").fillna(0.0)

    q = working[[query_col]].drop_duplicates().sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split = int(len(q) * 0.8)
    train_q = set(q.iloc[:split][query_col].tolist())
    test_q = set(q.iloc[split:][query_col].tolist())

    train_df = working[working[query_col].isin(train_q)].copy()
    test_df = working[working[query_col].isin(test_q)].copy()

    train_df["_rel"] = pd.qcut(train_df[score_col], q=5, labels=False, duplicates="drop").astype(int)
    bins = np.quantile(train_df[score_col].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    test_df["_rel"] = np.digitize(test_df[score_col].to_numpy(), bins, right=True).astype(int)

    feature_columns = [c for c in working.columns if c not in set(drop_cols + ["_rel"])]
    X_train_raw = train_df[feature_columns]
    X_test_raw = test_df[feature_columns]

    X_train, enc_maps = _cast_and_encode(X_train_raw)
    X_test, _ = _cast_and_encode(X_test_raw, fit_maps=enc_maps)

    train_sorted_idx = np.argsort(train_df[query_col].to_numpy(), kind="mergesort")
    test_sorted_idx = np.argsort(test_df[query_col].to_numpy(), kind="mergesort")

    train_df = train_df.iloc[train_sorted_idx].reset_index(drop=True)
    test_df = test_df.iloc[test_sorted_idx].reset_index(drop=True)
    X_train = X_train.iloc[train_sorted_idx].reset_index(drop=True)
    X_test = X_test.iloc[test_sorted_idx].reset_index(drop=True)

    group_train = train_df.groupby(query_col).size().astype(int).tolist()
    group_test = test_df.groupby(query_col).size().astype(int).tolist()

    test_bin = (test_df["_rel"] >= 3).astype(int).to_numpy()

    metadata = {
        "query_col": query_col,
        "score_col": score_col,
        "encoder_maps": enc_maps,
        "train_queries": int(len(group_train)),
        "test_queries": int(len(group_test)),
    }

    return PreparedData(
        X_train=X_train,
        y_train=train_df["_rel"].to_numpy(),
        group_train=group_train,
        X_test=X_test,
        y_test=test_df["_rel"].to_numpy(),
        y_test_bin=test_bin,
        group_test=group_test,
        feature_columns=feature_columns,
        metadata=metadata,
    )


def _train_ranker(data: PreparedData, run_name: str, registered_model_name: str, seed: int):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "learning_rate": 0.05,
        "n_estimators": 250,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "random_state": seed,
        "verbosity": -1,
    }

    model = lgb.LGBMRanker(**params)

    model.fit(
        data.X_train,
        data.y_train,
        group=data.group_train,
        eval_set=[(data.X_test, data.y_test)],
        eval_group=[data.group_test],
        eval_at=[1, 3, 5],
    )

    pred_test = model.predict(data.X_test)
    metrics = {
        "ndcg_at_3": round(ndcg_at_k(data.y_test, pred_test, data.group_test, k=3), 4),
        "recall_at_3": round(recall_at_k(data.y_test_bin, pred_test, data.group_test, k=3), 4),
        "map_at_3": round(map_at_k(data.y_test_bin, pred_test, data.group_test, k=3), 4),
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in params.items() if not isinstance(v, list)})
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, "model", registered_model_name=registered_model_name)

    return model, metrics


def _save_artifact(file_name: str, model, data: PreparedData, metrics: Dict[str, float]):
    path = MODELS_DIR / file_name
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_columns": data.feature_columns,
                "metadata": data.metadata,
                "metrics": metrics,
            },
            f,
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="Train student and instructor recommendation rankers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_mlflow(experiment="xai-recommendation-engine")

    student_path = RECO_DIR / "student_recommendation_dataset.csv"
    instructor_path = RECO_DIR / "instructor_recommendation_dataset.csv"
    if not student_path.exists() or not instructor_path.exists():
        raise FileNotFoundError("Recommendation datasets not found. Run synthetic generation first.")

    student_df = pd.read_csv(student_path)
    instructor_df = pd.read_csv(instructor_path)

    student_data = _prepare_data(
        df=student_df,
        query_col="student_id",
        score_col="recommendation_score",
        drop_cols=[
            "recommendation_reason",
            "rank",
            "recommendation_score",
            "student_id",
        ],
        seed=args.seed,
    )

    instructor_data = _prepare_data(
        df=instructor_df,
        query_col="instructor_id",
        score_col="assignment_priority",
        drop_cols=[
            "recommendation_reason",
            "rank",
            "assignment_priority",
            "instructor_id",
        ],
        seed=args.seed,
    )

    log.info("Training student ranker ...")
    student_model, student_metrics = _train_ranker(
        student_data,
        run_name="student-ranker-lgbm",
        registered_model_name="student-recommendation-ranker",
        seed=args.seed,
    )

    log.info("Training instructor ranker ...")
    instructor_model, instructor_metrics = _train_ranker(
        instructor_data,
        run_name="instructor-ranker-lgbm",
        registered_model_name="instructor-recommendation-ranker",
        seed=args.seed,
    )

    sp = _save_artifact("student_ranker.pkl", student_model, student_data, student_metrics)
    ip = _save_artifact("instructor_ranker.pkl", instructor_model, instructor_data, instructor_metrics)

    summary = {
        "student": student_metrics,
        "instructor": instructor_metrics,
        "artifacts": {
            "student_ranker": str(sp.relative_to(PROJECT_ROOT)),
            "instructor_ranker": str(ip.relative_to(PROJECT_ROOT)),
        },
    }
    summary_path = MODELS_DIR / "recommendation_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("Student metrics: %s", student_metrics)
    log.info("Instructor metrics: %s", instructor_metrics)
    log.info("Saved summary -> %s", summary_path)


if __name__ == "__main__":
    main()
