"""
XAIClient — HTTP client wrapping the XAI Learning Recommendation System API.
"""

from __future__ import annotations

import os
from typing import Any, Optional

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    import urllib.request, json as _json
    _HAS_HTTPX = False


class XAIClient:
    """
    Client for the XAI Learning Recommendation System backend.

    Parameters
    ----------
    base_url : str
        Base URL of the running FastAPI server (default: http://localhost:8000).
    token : str, optional
        Bearer token for authenticated endpoints (/explain/me, /explain/me/history).
        Can also be set via XAI_SDK_TOKEN env var.
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        token: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._token   = token or os.getenv("XAI_SDK_TOKEN")

    # ── internal ───────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        if _HAS_HTTPX:
            r = httpx.post(url, json=body, headers=self._headers(), timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        # stdlib fallback
        import json, urllib.request, urllib.error
        data = json.dumps(body).encode()
        req  = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        if _HAS_HTTPX:
            r = httpx.get(url, headers=self._headers(), timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        import json, urllib.request
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    # ── public API ─────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """GET /health — server status, model version, GPU info."""
        return self._get("/health")

    def predict(
        self,
        features: dict[str, Any],
        model: str = "gbm",
    ) -> dict:
        """
        POST /predict — predict dropout risk with MAPIE uncertainty.

        Returns
        -------
        {risk_score, risk_label, model_used, uncertainty}
        """
        return self._post("/predict", {**features, "__model": model})

    def explain(
        self,
        features: dict[str, Any],
        learner_id: str = "anonymous",
        model: str = "gbm",
        audience: str = "both",
    ) -> dict:
        """
        POST /explain — full multi-layer XAI explanation.

        Returns all of: shap_values, interactions, anchor_rule, prototypes,
        counterfactual, ranked_actions, causal_annotations, trust_score,
        uncertainty, explanation_drift, narratives.
        """
        return self._post("/explain", {
            "features":   features,
            "learner_id": learner_id,
            "model":      model,
            "audience":   audience,
        })

    def whatif(
        self,
        features: dict[str, Any],
        overrides: dict[str, float],
    ) -> dict:
        """
        POST /whatif — recompute SHAP after feature overrides.

        Returns
        -------
        {shap_values, risk_score, risk_delta, top_features}
        """
        return self._post("/whatif", {"features": features, "overrides": overrides})

    def counterfactual(self, features: dict[str, Any]) -> dict:
        """
        POST /counterfactual — DiCE counterfactuals + ranked prescriptive actions.

        Returns
        -------
        {actions, counterfactuals, best_cf, changed_features, ranked_actions}
        """
        return self._post("/counterfactual", features)

    def simulate(
        self,
        features: dict[str, Any],
        current_week: int = 6,
        target_week: int = 12,
        n_simulations: int = 1000,
    ) -> dict:
        """
        POST /simulate — Monte Carlo forward projection.

        Returns
        -------
        {feature_distributions, outcome_distribution, risk_percentiles}
        """
        return self._post("/simulate", {
            "features":      features,
            "current_week":  current_week,
            "target_week":   target_week,
            "n_simulations": n_simulations,
        })

    def history(self, learner_id: str) -> dict:
        """
        GET /history/{learner_id} — explanation timeline + drift flags.

        Returns
        -------
        {learner_id, timeline, history, drift_flags}
        """
        return self._get(f"/history/{learner_id}")

    def feedback(
        self,
        learner_id: str,
        rating: Optional[int] = None,
        followed_recommendation: Optional[bool] = None,
        explanation_id: Optional[str] = None,
        correction_feature: Optional[str] = None,
        correction_comment: Optional[str] = None,
        audience: Optional[str] = None,
    ) -> dict:
        """POST /feedback — submit explicit learner/instructor feedback."""
        body: dict = {"learner_id": learner_id}
        if rating is not None:                        body["rating"] = rating
        if followed_recommendation is not None:       body["followed_recommendation"] = followed_recommendation
        if explanation_id is not None:                body["explanation_id"] = explanation_id
        if correction_feature is not None:            body["correction_feature"] = correction_feature
        if correction_comment is not None:            body["correction_comment"] = correction_comment
        if audience is not None:                      body["audience"] = audience
        return self._post("/feedback", body)

    def feedback_stats(self) -> dict:
        """GET /feedback/stats — aggregate follow rates, avg rating."""
        return self._get("/feedback/stats")

    def mlops_health(self) -> dict:
        """GET /mlops/health — model version, drift status, prediction count."""
        return self._get("/mlops/health")

    def mlops_metrics(self) -> dict:
        """GET /mlops/metrics — training AUC/F1/Brier from training_summary.json."""
        return self._get("/mlops/metrics")

    def mlops_drift_report(self) -> dict:
        """GET /mlops/drift-report — Evidently data drift report."""
        return self._get("/mlops/drift-report")

    # ── convenience ────────────────────────────────────────────────────────────

    @property
    def top_action(self) -> Optional[str]:
        """Not a method — use client.explain(...) and access ['ranked_actions'][0]['plain_language']."""
        return None
