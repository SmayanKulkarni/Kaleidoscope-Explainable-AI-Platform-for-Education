"""
Action Ranker
==============
Ranks prescriptive actions from DiCE counterfactuals using a composite score
derived from SHAP magnitude, learned actionability weights, and causal annotations.

Actionability weights are estimated from training data variance (how much each
feature changes naturally across students), not hardcoded.

Public API
----------
ActionRanker(feature_names, X_train, causal_annotator)
    .rank(dice_actions, shap_values) -> list[RankedAction]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RankedAction:
    feature:            str
    current_value:      float
    target_value:       float
    direction:          str
    shap_magnitude:     float
    actionability:      float       # learned from data
    causal_weight:      float       # 1.0 if causal, 0.7 if correlational, 0.3 if confounder
    priority_score:     float       # composite
    priority_rank:      int
    estimated_impact:   float       # delta risk from DiCE
    plain_language:     str

    def to_dict(self) -> dict:
        return {
            "feature":          self.feature,
            "current_value":    round(self.current_value, 4),
            "target_value":     round(self.target_value, 4),
            "direction":        self.direction,
            "shap_magnitude":   round(self.shap_magnitude, 5),
            "actionability":    round(self.actionability, 4),
            "causal_weight":    round(self.causal_weight, 2),
            "priority_score":   round(self.priority_score, 5),
            "priority_rank":    self.priority_rank,
            "estimated_impact": round(self.estimated_impact, 4),
            "plain_language":   self.plain_language,
        }


# Human-readable templates for each feature direction
_ACTION_TEMPLATES = {
    "login_frequency_weekly":     {"increase": "Log in more frequently each week",
                                   "decrease": "Your login frequency is already strong"},
    "avg_session_duration_min":   {"increase": "Spend more time per study session",
                                   "decrease": "Shorter, focused sessions may help"},
    "forum_posts_count":          {"increase": "Participate more in forum discussions",
                                   "decrease": "Forum activity is not a priority right now"},
    "video_completion_rate":      {"increase": "Watch more of the course videos",
                                   "decrease": "Video completion is already sufficient"},
    "quiz_avg_score":             {"increase": "Focus on improving quiz performance",
                                   "decrease": "Quiz scores are not the main concern"},
    "quiz_completion_rate":       {"increase": "Complete more quizzes",
                                   "decrease": "Quiz completion rate is adequate"},
    "assignment_submission_rate": {"increase": "Submit more assignments on time",
                                   "decrease": "Assignment submissions are on track"},
    "days_since_last_activity":   {"increase": "Take a break if needed",
                                   "decrease": "Re-engage with the course — activity has dropped"},
    "missed_deadlines_count":     {"increase": "This would increase risk",
                                   "decrease": "Catch up on missed deadlines"},
    "help_requests_count":        {"increase": "Don't hesitate to ask for help",
                                   "decrease": "You're already seeking help effectively"},
    "prior_course_completions":   {"increase": "This is a historical feature",
                                   "decrease": "This is a historical feature"},
    "current_week_in_course":     {"increase": "This progresses naturally",
                                   "decrease": "This progresses naturally"},
}


class ActionRanker:
    CAUSAL_WEIGHTS = {"causal": 1.0, "correlational": 0.7, "confounder": 0.3}

    def __init__(
        self,
        feature_names: list[str],
        X_train: np.ndarray,
        causal_annotator=None,
    ):
        self.feature_names    = feature_names
        self.causal_annotator = causal_annotator

        # Learn actionability from data: normalised IQR as proxy for
        # "how much this feature naturally varies across students"
        iqrs = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)
        max_iqr = iqrs.max() if iqrs.max() > 0 else 1.0
        self.actionability_weights = {}
        for i, f in enumerate(feature_names):
            self.actionability_weights[f] = float(np.clip(iqrs[i] / max_iqr, 0.1, 1.0))

        # Immutable + latent features get zero actionability
        _non_actionable = [
            "prior_course_completions",
            "current_week_in_course",
            "engagement_latent_1",
            "engagement_latent_2",
            "engagement_latent_3",
        ]
        for f in _non_actionable:
            if f in self.actionability_weights:
                self.actionability_weights[f] = 0.0

        log.info("ActionRanker initialised  actionability learned from IQR of %d features", len(feature_names))

    def _get_causal_weight(self, feature: str) -> float:
        if self.causal_annotator is None:
            return 0.85  # default: assume moderate causality
        effects = self.causal_annotator.get_causal_effects()
        effect  = effects.get(feature)
        if effect is None:
            return 0.85
        if feature in self.causal_annotator.CONFOUNDERS:
            return self.CAUSAL_WEIGHTS["confounder"]
        if effect.is_causal:
            return self.CAUSAL_WEIGHTS["causal"]
        return self.CAUSAL_WEIGHTS["correlational"]

    def _plain_language(self, feature: str, direction: str) -> str:
        templates = _ACTION_TEMPLATES.get(feature, {})
        return templates.get(direction, f"Adjust {feature.replace('_', ' ')}")

    def rank(
        self,
        dice_actions: list,
        shap_values: dict[str, float],
        base_risk: float = 0.0,
        cf_risk: float = 0.0,
    ) -> list[RankedAction]:
        """
        Parameters
        ----------
        dice_actions : list[PrescriptiveAction]  from DiCEExplainer
        shap_values : dict from SHAPExplainer.explain()
        base_risk : original risk score
        cf_risk : counterfactual risk score
        """
        total_delta = abs(base_risk - cf_risk) if cf_risk > 0 else 0.0
        n_actions   = max(len(dice_actions), 1)

        ranked = []
        for action in dice_actions:
            # Skip latent engagement features — not actionable by the learner
            if action.feature.startswith("engagement_latent_"):
                continue

            shap_mag       = abs(shap_values.get(action.feature, 0.0))
            actionability  = self.actionability_weights.get(action.feature, 0.5)
            causal_w       = self._get_causal_weight(action.feature)

            priority_score = shap_mag * actionability * causal_w

            # Estimate per-feature impact as proportional share of total delta
            estimated_impact = (action.magnitude / max(
                sum(a.magnitude for a in dice_actions), 1e-8
            )) * total_delta if total_delta > 0 else shap_mag

            ranked.append(RankedAction(
                feature=action.feature,
                current_value=action.current_value,
                target_value=action.target_value,
                direction=action.direction,
                shap_magnitude=shap_mag,
                actionability=actionability,
                causal_weight=causal_w,
                priority_score=priority_score,
                priority_rank=0,
                estimated_impact=estimated_impact,
                plain_language=self._plain_language(action.feature, action.direction),
            ))

        ranked.sort(key=lambda a: a.priority_score, reverse=True)
        for i, a in enumerate(ranked):
            a.priority_rank = i + 1

        return ranked
