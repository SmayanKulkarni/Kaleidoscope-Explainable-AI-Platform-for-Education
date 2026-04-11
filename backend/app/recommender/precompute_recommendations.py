#!/usr/bin/env python3
"""Batch precompute top-k recommendations from trained rankers."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECO_DIR = PROJECT_ROOT / "data" / "synthetic" / "recommendations"
MODEL_DIR = PROJECT_ROOT / "models" / "recommenders"
OUT_DIR = PROJECT_ROOT / "data" / "recommendations" / "precomputed"


def _load_artifact(name: str):
    with open(MODEL_DIR / name, "rb") as f:
        return pickle.load(f)


def _encode(df: pd.DataFrame, feature_columns, encoder_maps):
    feat = df[feature_columns].copy()
    for col in feature_columns:
        if feat[col].dtype == "O":
            mp = encoder_maps.get(col, {"unknown": 0})
            feat[col] = feat[col].fillna("unknown").astype(str).map(mp).fillna(-1).astype(int)
        else:
            feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(0.0)
    return feat


def _topk(df: pd.DataFrame, query_col: str, score_col: str, k: int):
    ranked = df.sort_values([query_col, score_col], ascending=[True, False])
    ranked["precomputed_rank"] = ranked.groupby(query_col).cumcount() + 1
    return ranked[ranked["precomputed_rank"] <= k].copy()


def main():
    parser = argparse.ArgumentParser(description="Precompute recommendations from trained rankers")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    student_df = pd.read_csv(RECO_DIR / "student_recommendation_dataset.csv")
    instructor_df = pd.read_csv(RECO_DIR / "instructor_recommendation_dataset.csv")

    student_art = _load_artifact("student_ranker.pkl")
    instructor_art = _load_artifact("instructor_ranker.pkl")

    sX = _encode(student_df, student_art["feature_columns"], student_art["metadata"]["encoder_maps"])
    iX = _encode(instructor_df, instructor_art["feature_columns"], instructor_art["metadata"]["encoder_maps"])

    student_df = student_df.copy()
    instructor_df = instructor_df.copy()
    student_df["model_score"] = student_art["model"].predict(sX)
    instructor_df["model_score"] = instructor_art["model"].predict(iX)

    student_topk = _topk(student_df, query_col="student_id", score_col="model_score", k=args.top_k)
    instructor_topk = _topk(instructor_df, query_col="instructor_id", score_col="model_score", k=args.top_k)

    sp = OUT_DIR / "student_topk.csv"
    ip = OUT_DIR / "instructor_topk.csv"
    rp = OUT_DIR / "precompute_report.json"
    student_topk.to_csv(sp, index=False)
    instructor_topk.to_csv(ip, index=False)

    report = {
        "top_k": args.top_k,
        "student_rows": int(len(student_topk)),
        "instructor_rows": int(len(instructor_topk)),
        "paths": {
            "student": str(sp.relative_to(PROJECT_ROOT)),
            "instructor": str(ip.relative_to(PROJECT_ROOT)),
        },
    }
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {sp} ({len(student_topk)} rows)")
    print(f"Wrote {ip} ({len(instructor_topk)} rows)")
    print(f"Wrote {rp}")


if __name__ == "__main__":
    main()
