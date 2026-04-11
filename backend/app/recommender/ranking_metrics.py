from __future__ import annotations

from typing import Iterable

import numpy as np


def _dcg(rels: np.ndarray) -> float:
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2 ** rels - 1) * discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, group_sizes: Iterable[int], k: int = 3) -> float:
    vals = []
    idx = 0
    for size in group_sizes:
        g_true = y_true[idx: idx + size]
        g_score = y_score[idx: idx + size]
        idx += size
        order = np.argsort(-g_score)
        rel_pred = g_true[order][:k]
        rel_ideal = np.sort(g_true)[::-1][:k]
        denom = _dcg(rel_ideal)
        vals.append(0.0 if denom == 0 else _dcg(rel_pred) / denom)
    return float(np.mean(vals)) if vals else 0.0


def recall_at_k(y_true_bin: np.ndarray, y_score: np.ndarray, group_sizes: Iterable[int], k: int = 3) -> float:
    vals = []
    idx = 0
    for size in group_sizes:
        g_true = y_true_bin[idx: idx + size]
        g_score = y_score[idx: idx + size]
        idx += size
        positives = int(g_true.sum())
        if positives == 0:
            continue
        top_idx = np.argsort(-g_score)[:k]
        hits = int(g_true[top_idx].sum())
        vals.append(hits / positives)
    return float(np.mean(vals)) if vals else 0.0


def map_at_k(y_true_bin: np.ndarray, y_score: np.ndarray, group_sizes: Iterable[int], k: int = 3) -> float:
    vals = []
    idx = 0
    for size in group_sizes:
        g_true = y_true_bin[idx: idx + size]
        g_score = y_score[idx: idx + size]
        idx += size

        order = np.argsort(-g_score)[:k]
        ranked_true = g_true[order]
        positives = int(g_true.sum())
        if positives == 0:
            continue

        cum_hits = 0
        precisions = []
        for rank_pos, val in enumerate(ranked_true, start=1):
            if val == 1:
                cum_hits += 1
                precisions.append(cum_hits / rank_pos)
        vals.append(float(np.sum(precisions) / min(positives, k)) if precisions else 0.0)
    return float(np.mean(vals)) if vals else 0.0
