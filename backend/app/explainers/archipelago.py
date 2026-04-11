"""
Archipelago Interaction Detector
=================================
Computes SHAP interaction values via TreeExplainer to surface
feature-pair interactions (amplifying or dampening).

Public API
----------
ArchipelagoExplainer(tree_explainer, feature_names)
    .get_interactions(features_dict, top_k=5) -> list[InteractionEffect]
    .to_narrative(interactions) -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shap

log = logging.getLogger(__name__)


@dataclass
class InteractionEffect:
    feature_a:        str
    feature_b:        str
    interaction_score: float
    direction:         str   # "amplifying" | "dampening"

    def to_dict(self) -> dict:
        return {
            "feature_a":         self.feature_a,
            "feature_b":         self.feature_b,
            "interaction_score": round(self.interaction_score, 6),
            "direction":         self.direction,
        }


class ArchipelagoExplainer:
    def __init__(
        self,
        tree_explainer: shap.TreeExplainer,
        feature_names: list[str],
    ):
        self.tree_explainer = tree_explainer
        self.feature_names  = feature_names
        log.info("ArchipelagoExplainer initialised  features=%d", len(feature_names))

    def get_interactions(
        self, features_dict: dict, top_k: int = 5
    ) -> list[InteractionEffect]:
        X = np.array(
            [features_dict[f] for f in self.feature_names], dtype=np.float64
        ).reshape(1, -1)

        interaction_matrix = self.tree_explainer.shap_interaction_values(X)

        # interaction_matrix may be a list for multi-output; take class-1
        if isinstance(interaction_matrix, list):
            interaction_matrix = interaction_matrix[1]

        # Shape: (1, n_features, n_features)
        iv = interaction_matrix[0]
        n = len(self.feature_names)

        # Extract upper-triangle off-diagonal interactions
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                score = float(iv[i, j] + iv[j, i])  # symmetrise
                if abs(score) > 1e-8:
                    pairs.append((i, j, score))

        # Sort by absolute interaction strength
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_pairs = pairs[:top_k]

        interactions = []
        for i, j, score in top_pairs:
            direction = "amplifying" if score > 0 else "dampening"
            interactions.append(InteractionEffect(
                feature_a=self.feature_names[i],
                feature_b=self.feature_names[j],
                interaction_score=score,
                direction=direction,
            ))

        return interactions

    def to_narrative(self, interactions: list[InteractionEffect]) -> dict:
        """
        Convert interactions to structured narrative dict for LLM narration payload.
        """
        if not interactions:
            return {"interaction_summary": "No significant feature interactions detected."}

        summaries = []
        for ix in interactions:
            fa = ix.feature_a.replace("_", " ")
            fb = ix.feature_b.replace("_", " ")
            verb = "amplify" if ix.direction == "amplifying" else "dampen"
            summaries.append(
                f"{fa} and {fb} {verb} each other's effect on dropout risk "
                f"(interaction score: {ix.interaction_score:.4f})."
            )

        return {
            "interaction_summary": " ".join(summaries),
            "top_interaction": {
                "features": [interactions[0].feature_a, interactions[0].feature_b],
                "direction": interactions[0].direction,
                "score": round(interactions[0].interaction_score, 6),
            },
            "n_interactions": len(interactions),
        }
