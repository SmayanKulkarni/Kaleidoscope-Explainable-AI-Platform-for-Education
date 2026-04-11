"""
xai-learner-sdk
===============
Thin Python client for the XAI Learning Recommendation System API.

Quickstart
----------
    pip install -e backend/sdk

    from xai_sdk import XAIClient

    client = XAIClient(base_url="http://localhost:8000")

    # Predict dropout risk
    result = client.predict(features)
    print(result["risk_score"], result["risk_label"])

    # Full multi-layer explanation
    explanation = client.explain(features, learner_id="learner_042")
    print(explanation["trust_score"])
    print(explanation["top_features"])

    # Counterfactual actions
    actions = client.counterfactual(features)
    for a in actions.get("ranked_actions", []):
        print(a["plain_language"])

    # Explanation history + drift
    history = client.history("learner_042")

    # What-if scenario
    delta = client.whatif(features, overrides={"quiz_avg_score": 80.0})
    print(delta["risk_delta"])

    # Monte Carlo forward simulation
    sim = client.simulate(features, current_week=6, target_week=12, n_simulations=1000)
    print(sim["outcome_distribution"])
"""

from __future__ import annotations

from .client import XAIClient

__all__ = ["XAIClient"]
__version__ = "0.1.0"
