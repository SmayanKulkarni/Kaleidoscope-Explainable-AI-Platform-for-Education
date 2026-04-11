"""
Prototype Explainer
====================
Finds the most similar past learners using k-NN with cosine similarity
on a StandardScaler-normalised feature space.

Public API
----------
PrototypeExplainer(X_train, y_train, learner_ids, feature_names)
    .explain(features_dict, k=3) -> PrototypeResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


@dataclass
class PrototypeMatch:
    learner_id:     str
    outcome:        str         # "completed" | "dropped_out"
    similarity:     float       # 0-1 (cosine)
    diff_features:  list[dict]  # [{name, learner_value, prototype_value, delta}]

    def to_dict(self) -> dict:
        return {
            "learner_id":    self.learner_id,
            "outcome":       self.outcome,
            "similarity":    round(self.similarity, 4),
            "diff_features": self.diff_features,
        }


@dataclass
class PrototypeResult:
    matches:              list[PrototypeMatch]
    n_similar_completed:  int
    n_similar_dropout:    int
    motivational_text:    str

    def to_dict(self) -> dict:
        return {
            "matches":             [m.to_dict() for m in self.matches],
            "n_similar_completed": self.n_similar_completed,
            "n_similar_dropout":   self.n_similar_dropout,
            "motivational_text":   self.motivational_text,
        }


class PrototypeExplainer:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        learner_ids: list | np.ndarray,
        feature_names: list[str],
        n_neighbors: int = 10,
    ):
        self.feature_names = feature_names
        self.y_train       = np.asarray(y_train)
        self.learner_ids   = np.asarray(learner_ids)
        self.X_train       = X_train

        self.scaler   = StandardScaler().fit(X_train)
        self.X_scaled = self.scaler.transform(X_train)

        self.nn = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(X_train)),
            metric="cosine",
            algorithm="brute",
        )
        self.nn.fit(self.X_scaled)

        log.info("PrototypeExplainer initialised  train=%d  k=%d", len(X_train), n_neighbors)

    def explain(self, features_dict: dict, k: int = 3) -> PrototypeResult:
        X = np.array(
            [features_dict[f] for f in self.feature_names], dtype=np.float64
        ).reshape(1, -1)
        X_sc = self.scaler.transform(X)

        distances, indices = self.nn.kneighbors(X_sc, n_neighbors=min(k * 2, len(self.X_train)))
        distances = distances[0]
        indices   = indices[0]

        # Cosine distance → similarity
        similarities = 1.0 - distances

        matches = []
        for idx, sim in zip(indices, similarities):
            if len(matches) >= k:
                break
            outcome = "dropped_out" if self.y_train[idx] == 1 else "completed"
            lid     = str(self.learner_ids[idx])

            # Top-2 distinguishing features (biggest absolute diff)
            diff = np.abs(self.X_train[idx] - X[0])
            top_diff_idx = np.argsort(-diff)[:2]
            diff_features = []
            for di in top_diff_idx:
                fname = self.feature_names[di]
                diff_features.append({
                    "name":            fname,
                    "learner_value":   round(float(X[0, di]), 4),
                    "prototype_value": round(float(self.X_train[idx, di]), 4),
                    "delta":           round(float(self.X_train[idx, di] - X[0, di]), 4),
                })

            matches.append(PrototypeMatch(
                learner_id=lid,
                outcome=outcome,
                similarity=float(sim),
                diff_features=diff_features,
            ))

        n_completed = sum(1 for m in matches if m.outcome == "completed")
        n_dropout   = sum(1 for m in matches if m.outcome == "dropped_out")

        # Build motivational narrative from match outcomes
        if n_completed > n_dropout:
            motivational = (
                f"Good news: {n_completed} out of {len(matches)} similar learners "
                f"successfully completed their course."
            )
            if matches:
                best = max((m for m in matches if m.outcome == "completed"),
                           key=lambda m: m.similarity, default=None)
                if best and best.diff_features:
                    top = best.diff_features[0]
                    if top["delta"] > 0:
                        motivational += (
                            f" A key difference: they had higher {top['name'].replace('_', ' ')} "
                            f"({top['prototype_value']} vs your {top['learner_value']})."
                        )
        elif n_dropout > 0:
            motivational = (
                f"Most similar learners ({n_dropout}/{len(matches)}) dropped out, "
                f"but early action can change your trajectory."
            )
        else:
            motivational = "We found similar learners in the dataset."

        return PrototypeResult(
            matches=matches,
            n_similar_completed=n_completed,
            n_similar_dropout=n_dropout,
            motivational_text=motivational,
        )
