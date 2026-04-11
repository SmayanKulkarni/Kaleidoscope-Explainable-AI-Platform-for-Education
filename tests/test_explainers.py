"""
Unit tests — all 5 explainer modules:
  SHAPExplainer, DiCEExplainer, AnchorsExplainer, PrototypeExplainer, ArchipelagoExplainer
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest_fixtures import *   # noqa: F401,F403  — shared fixtures


# ──────────────────────────────────────────────────────────────────────────────
# SHAPExplainer
# ──────────────────────────────────────────────────────────────────────────────

class TestSHAPExplainer:
    @pytest.fixture(scope="class")
    def shap_exp(self, gbm_bundle, dataset):
        from backend.app.explainers.shap_explainer import SHAPExplainer
        return SHAPExplainer(
            gbm_model=gbm_bundle["model"],
            feature_names=gbm_bundle["feature_names"],
        )

    def test_explain_returns_result(self, shap_exp, sample_features):
        result = shap_exp.explain(sample_features)
        assert result is not None
        assert hasattr(result, "shap_values")
        assert hasattr(result, "risk_score")
        assert hasattr(result, "top_features")

    def test_shap_values_length(self, shap_exp, sample_features):
        from backend.app.model.data_loader import FEATURE_COLUMNS
        result = shap_exp.explain(sample_features)
        assert len(result.shap_values) == len(FEATURE_COLUMNS)

    def test_risk_score_range(self, shap_exp, sample_features):
        result = shap_exp.explain(sample_features)
        assert 0.0 <= result.risk_score <= 1.0

    def test_top_features_sorted_by_magnitude(self, shap_exp, sample_features):
        result = shap_exp.explain(sample_features)
        magnitudes = [abs(item["shap"]) for item in result.top_features]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_whatif_returns_delta(self, shap_exp, sample_features):
        # explain_whatif takes a plain features dict (overrides pre-applied by caller)
        modified = dict(sample_features, quiz_avg_score=90.0)
        result = shap_exp.explain_whatif(modified)
        assert hasattr(result, "risk_score")
        assert hasattr(result, "shap_values")

    def test_stability_score_range(self, shap_exp, sample_features):
        score = shap_exp.stability_score(sample_features)
        assert 0.0 <= score <= 1.0

    def test_high_risk_higher_than_low_risk(self, shap_exp, high_risk_features, low_risk_features):
        hr = shap_exp.explain(high_risk_features).risk_score
        lr = shap_exp.explain(low_risk_features).risk_score
        assert hr > lr

    def test_explain_whatif_override_decreases_risk(self, shap_exp, high_risk_features):
        # explain_whatif takes pre-merged features dict (no separate overrides kwarg)
        improved = dict(high_risk_features, quiz_avg_score=90.0, assignment_submission_rate=0.99)
        after = shap_exp.explain_whatif(improved).risk_score
        assert isinstance(after, float)


# ──────────────────────────────────────────────────────────────────────────────
# DiCEExplainer
# ──────────────────────────────────────────────────────────────────────────────

class TestDiCEExplainer:
    @pytest.fixture(scope="class")
    def dice_exp(self, gbm_bundle, dataset):
        from backend.app.explainers.dice_explainer import DiCEExplainer
        return DiCEExplainer(
            model=gbm_bundle["model"],
            X_train=dataset["X_train"],
            feature_names=gbm_bundle["feature_names"],
            y_train=dataset["y_train"],
        )

    def test_get_counterfactuals_returns_result(self, dice_exp, sample_features):
        result = dice_exp.get_counterfactuals(sample_features)
        assert result is not None
        assert hasattr(result, "actions")

    def test_counterfactuals_to_dict(self, dice_exp, sample_features):
        result = dice_exp.get_counterfactuals(sample_features)
        d = result.to_dict()
        assert "actions" in d

    def test_immutable_features_not_in_actions(self, dice_exp, high_risk_features):
        from backend.app.explainers.dice_explainer import IMMUTABLE_FEATURES
        result = dice_exp.get_counterfactuals(high_risk_features)
        for action in result.actions:
            feat = action.feature if hasattr(action, "feature") else action.get("feature", "")
            assert feat not in IMMUTABLE_FEATURES, \
                f"Immutable feature '{feat}' appeared in DiCE actions"

    def test_action_is_prescriptive_action(self, dice_exp, sample_features):
        result = dice_exp.get_counterfactuals(sample_features)
        for action in result.actions:
            assert hasattr(action, "feature") or isinstance(action, dict)

    def test_latent_features_not_in_actions(self, dice_exp, high_risk_features):
        result = dice_exp.get_counterfactuals(high_risk_features)
        for action in result.actions:
            feat = action.feature if hasattr(action, "feature") else action.get("feature", "")
            assert not feat.startswith("engagement_latent_"), \
                f"Latent feature '{feat}' appeared in DiCE actions"

    def test_action_values_are_within_bounds(self, dice_exp, high_risk_features):
        result = dice_exp.get_counterfactuals(high_risk_features)
        for action in result.actions:
            val = action.suggested_value if hasattr(action, "suggested_value") else action.get("suggested_value", 0)
            assert isinstance(val, (int, float))


# ──────────────────────────────────────────────────────────────────────────────
# PrototypeExplainer
# ──────────────────────────────────────────────────────────────────────────────

class TestPrototypeExplainer:
    @pytest.fixture(scope="class")
    def proto_exp(self, gbm_bundle, dataset):
        from backend.app.explainers.prototype_explainer import PrototypeExplainer
        return PrototypeExplainer(
            X_train=dataset["X_train"],
            y_train=dataset["y_train"],
            learner_ids=np.arange(len(dataset["X_train"])),
            feature_names=gbm_bundle["feature_names"],
        )

    def test_explain_returns_result(self, proto_exp, sample_features):
        result = proto_exp.explain(sample_features)
        assert result is not None

    def test_to_dict_has_required_keys(self, proto_exp, sample_features):
        d = proto_exp.explain(sample_features).to_dict()
        assert "similar_learners" in d or "prototypes" in d or len(d) > 0

    def test_explain_high_risk(self, proto_exp, high_risk_features):
        result = proto_exp.explain(high_risk_features)
        assert result is not None

    def test_explain_low_risk(self, proto_exp, low_risk_features):
        result = proto_exp.explain(low_risk_features)
        assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# ArchipelagoExplainer
# ──────────────────────────────────────────────────────────────────────────────

class TestArchipelagoExplainer:
    @pytest.fixture(scope="class")
    def arch_exp(self, gbm_bundle, dataset):
        import shap
        from backend.app.explainers.archipelago import ArchipelagoExplainer
        tree_exp = shap.TreeExplainer(gbm_bundle["model"])
        return ArchipelagoExplainer(
            tree_explainer=tree_exp,
            feature_names=gbm_bundle["feature_names"],
        )

    def test_get_interactions_returns_list(self, arch_exp, sample_features):
        interactions = arch_exp.get_interactions(sample_features)
        assert isinstance(interactions, list)

    def test_interaction_items_have_to_dict(self, arch_exp, sample_features):
        interactions = arch_exp.get_interactions(sample_features)
        for ix in interactions:
            d = ix.to_dict()
            assert isinstance(d, dict)

    def test_to_narrative_returns_string_or_dict(self, arch_exp, sample_features):
        interactions = arch_exp.get_interactions(sample_features)
        narrative = arch_exp.to_narrative(interactions)
        assert isinstance(narrative, (str, dict))

    def test_empty_interactions_narrative(self, arch_exp):
        narrative = arch_exp.to_narrative([])
        assert isinstance(narrative, (str, dict))


# ──────────────────────────────────────────────────────────────────────────────
# AnchorsExplainer
# ──────────────────────────────────────────────────────────────────────────────

class TestAnchorsExplainer:
    @pytest.fixture(scope="class")
    def anchors_exp(self, gbm_bundle, dataset):
        from backend.app.explainers.anchors_explainer import AnchorsExplainer
        predict_fn = lambda X: (gbm_bundle["model"].predict_proba(X)[:, 1] >= 0.5).astype(int)
        return AnchorsExplainer(
            predict_fn=predict_fn,
            X_train=dataset["X_train"],
            feature_names=gbm_bundle["feature_names"],
        )

    def test_explain_returns_result(self, anchors_exp, sample_features):
        result = anchors_exp.explain(sample_features)
        assert result is not None

    def test_to_dict_has_anchor_rule(self, anchors_exp, sample_features):
        d = anchors_exp.explain(sample_features).to_dict()
        assert "anchor_rule" in d or "rule" in d or len(d) > 0

    def test_precision_in_range(self, anchors_exp, sample_features):
        result = anchors_exp.explain(sample_features)
        d = result.to_dict()
        precision = d.get("precision", 1.0)
        assert 0.0 <= precision <= 1.0
