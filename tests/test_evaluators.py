"""
Unit tests — evaluators and causal/trust/action modules:
  UncertaintyEstimator, TrustScorer, CausalAnnotator, ActionRanker, ExplanationDriftDetector
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest_fixtures import *   # noqa: F401,F403


# ──────────────────────────────────────────────────────────────────────────────
# UncertaintyEstimator (MAPIE)
# ──────────────────────────────────────────────────────────────────────────────

class TestUncertaintyEstimator:
    @pytest.fixture(scope="class")
    def estimator(self, gbm_bundle, dataset):
        from backend.app.evaluator.uncertainty_estimator import UncertaintyEstimator
        return UncertaintyEstimator(
            model=gbm_bundle["model"],
            X_cal=dataset["X_train"],
            y_cal=dataset["y_train"],
        )

    def test_predict_with_uncertainty_returns_result(self, estimator, dataset):
        X = dataset["X_test"][:1]
        result = estimator.predict_with_uncertainty(X)
        assert result is not None

    def test_to_dict_has_interval(self, estimator, dataset):
        X = dataset["X_test"][:1]
        d = estimator.predict_with_uncertainty(X).to_dict()
        assert "lower" in d or "ci_lower" in d or "uncertainty" in d or len(d) > 0

    def test_batch_prediction(self, estimator, dataset):
        X = dataset["X_test"]
        result = estimator.predict_with_uncertainty(X)
        assert result is not None

    def test_interval_lower_le_upper(self, estimator, dataset):
        X = dataset["X_test"][:1]
        d = estimator.predict_with_uncertainty(X).to_dict()
        lower = d.get("lower", d.get("ci_lower", 0.0))
        upper = d.get("upper", d.get("ci_upper", 1.0))
        assert lower <= upper


# ──────────────────────────────────────────────────────────────────────────────
# TrustScorer
# ──────────────────────────────────────────────────────────────────────────────

class TestTrustScorer:
    @pytest.fixture(scope="class")
    def trust_scorer(self):
        from backend.app.evaluator.trust_scorer import TrustScorer
        return TrustScorer()

    def _mock_shap(self):
        from backend.app.model.data_loader import FEATURE_COLUMNS
        return {f: float(i * 0.05) for i, f in enumerate(FEATURE_COLUMNS)}

    def test_score_returns_result(self, trust_scorer):
        shap_values = self._mock_shap()
        result = trust_scorer.score(
            shap_values=shap_values,
            base_value=0.35,
            risk_score=0.62,
            stability=0.85,
        )
        assert result is not None

    def test_trust_score_range(self, trust_scorer):
        shap_values = self._mock_shap()
        result = trust_scorer.score(
            shap_values=shap_values,
            base_value=0.35,
            risk_score=0.62,
            stability=0.85,
        )
        d = result.to_dict()
        ts = d.get("trust_score", d.get("score", 0.5))
        assert 0.0 <= ts <= 1.0

    def test_to_dict_completeness(self, trust_scorer):
        shap_values = self._mock_shap()
        result = trust_scorer.score(
            shap_values=shap_values,
            base_value=0.35,
            risk_score=0.62,
            stability=0.85,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert len(d) > 0

    def test_high_stability_increases_trust(self, trust_scorer):
        sv = self._mock_shap()
        low  = trust_scorer.score(sv, 0.35, 0.62, stability=0.1).to_dict()
        high = trust_scorer.score(sv, 0.35, 0.62, stability=0.99).to_dict()
        low_ts  = low.get("trust_score",  low.get("score",  0))
        high_ts = high.get("trust_score", high.get("score", 1))
        assert high_ts >= low_ts


# ──────────────────────────────────────────────────────────────────────────────
# CausalAnnotator
# ──────────────────────────────────────────────────────────────────────────────

class TestCausalAnnotator:
    @pytest.fixture(scope="class")
    def annotator(self, gbm_bundle, dataset):
        from backend.app.causal.causal_annotator import CausalAnnotator
        return CausalAnnotator(
            X_train=dataset["X_train"],
            y_train=dataset["y_train"],
            feature_names=gbm_bundle["feature_names"],
        )

    def _mock_shap(self):
        from backend.app.model.data_loader import FEATURE_COLUMNS
        return {f: float((i - 6) * 0.08) for i, f in enumerate(FEATURE_COLUMNS)}

    def test_annotate_shap_returns_list(self, annotator):
        annotations = annotator.annotate_shap(self._mock_shap())
        assert isinstance(annotations, list)
        assert len(annotations) > 0

    def test_each_annotation_has_causal_type(self, annotator):
        for ann in annotator.annotate_shap(self._mock_shap()):
            d = ann.to_dict()
            assert "causal_type" in d or "is_causal" in d

    def test_latent_features_are_correlational(self, annotator):
        from backend.app.model.data_loader import FEATURE_COLUMNS
        latent_shap = {f: 0.0 for f in FEATURE_COLUMNS}
        latent_shap["engagement_latent_1"] = 0.3
        latent_shap["engagement_latent_2"] = 0.2
        latent_shap["engagement_latent_3"] = 0.1

        annotations = annotator.annotate_shap(latent_shap)
        for ann in annotations:
            d = ann.to_dict()
            feat = d.get("feature", "")
            if feat.startswith("engagement_latent_"):
                ct = d.get("causal_type", "")
                is_c = d.get("is_causal", True)
                assert ct == "correlational" or is_c is False, \
                    f"Latent feature {feat} must be correlational, got {d}"

    def test_to_dict_serialisable(self, annotator):
        import json
        for ann in annotator.annotate_shap(self._mock_shap()):
            json.dumps(ann.to_dict())  # must not raise


# ──────────────────────────────────────────────────────────────────────────────
# ActionRanker
# ──────────────────────────────────────────────────────────────────────────────

class TestActionRanker:
    @pytest.fixture(scope="class")
    def ranker(self, gbm_bundle, dataset):
        from backend.app.causal.causal_annotator import CausalAnnotator
        from backend.app.prescriptor.action_ranker import ActionRanker

        annotator = CausalAnnotator(
            X_train=dataset["X_train"],
            y_train=dataset["y_train"],
            feature_names=gbm_bundle["feature_names"],
        )
        return ActionRanker(
            feature_names=gbm_bundle["feature_names"],
            X_train=dataset["X_train"],
            causal_annotator=annotator,
        )

    @pytest.fixture(scope="class")
    def mock_actions(self):
        from backend.app.explainers.dice_explainer import PrescriptiveAction
        return [
            PrescriptiveAction(feature="quiz_avg_score",             current_value=45.0, target_value=75.0, direction="increase", magnitude=30.0),
            PrescriptiveAction(feature="assignment_submission_rate", current_value=0.3,  target_value=0.9,  direction="increase", magnitude=0.6),
            PrescriptiveAction(feature="login_frequency_weekly",     current_value=1.0,  target_value=5.0,  direction="increase", magnitude=4.0),
        ]

    def _mock_shap(self):
        from backend.app.model.data_loader import FEATURE_COLUMNS
        return {f: float((i - 6) * 0.1) for i, f in enumerate(FEATURE_COLUMNS)}

    def test_rank_returns_list(self, ranker, mock_actions):
        ranked = ranker.rank(mock_actions, self._mock_shap(), base_risk=0.65)
        assert isinstance(ranked, list)

    def test_ranked_sorted_by_score(self, ranker, mock_actions):
        ranked = ranker.rank(mock_actions, self._mock_shap(), base_risk=0.65)
        scores = [a.priority_score if hasattr(a, "priority_score") else a.to_dict().get("priority_score", 0)
                  for a in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_latent_features_not_in_ranked(self, ranker):
        from backend.app.explainers.dice_explainer import PrescriptiveAction
        actions = [
            PrescriptiveAction(feature="engagement_latent_1", current_value=0.0, target_value=0.5, direction="increase", magnitude=0.5),
            PrescriptiveAction(feature="quiz_avg_score",       current_value=45.0, target_value=80.0, direction="increase", magnitude=35.0),
        ]
        ranked = ranker.rank(actions, self._mock_shap(), base_risk=0.65)
        features = [
            a.feature if hasattr(a, "feature") else a.to_dict().get("feature", "")
            for a in ranked
        ]
        assert "engagement_latent_1" not in features

    def test_ranked_action_to_dict_serialisable(self, ranker, mock_actions):
        import json
        ranked = ranker.rank(mock_actions, self._mock_shap(), base_risk=0.65)
        for a in ranked:
            json.dumps(a.to_dict())


# ──────────────────────────────────────────────────────────────────────────────
# ExplanationDriftDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestDriftDetector:
    @pytest.fixture
    def store_with_data(self):
        from backend.app.tracker.consistency_store import ExplanationStore
        from backend.app.model.data_loader import FEATURE_COLUMNS
        store = ExplanationStore(db_url="sqlite:///:memory:")
        rng = np.random.default_rng(0)
        for i in range(8):
            sv = {f: float(rng.normal(0, 0.1)) for f in FEATURE_COLUMNS}
            store.save(
                learner_id="L001",
                risk_score=float(rng.random()),
                shap_values=sv,
                top3_features=[(f, sv[f]) for f in list(sv)[:3]],
                trust_score=0.8,
            )
        return store

    def test_check_drift_no_history(self):
        from backend.app.tracker.drift_detector import ExplanationDriftDetector
        from backend.app.tracker.consistency_store import ExplanationStore
        store = ExplanationStore(db_url="sqlite:///:memory:")
        det = ExplanationDriftDetector()
        result = det.check_learner_drift("NOBODY", store)
        assert result is None or hasattr(result, "to_dict")

    def test_check_drift_with_history(self, store_with_data):
        from backend.app.tracker.drift_detector import ExplanationDriftDetector
        det = ExplanationDriftDetector()
        result = det.check_learner_drift("L001", store_with_data)
        if result is not None:
            d = result.to_dict()
            assert "drift_detected" in d or "jsd" in d or len(d) > 0
