"""
LLM Narrator — Dual-Audience Narration Layer
=============================================
Uses Groq (llama-3.3-70b-versatile) to narrate pre-computed XAI outputs.

CRITICAL RULE:
    The LLM receives ONLY computed data payloads (SHAP values, DiCE actions,
    Anchor rules, trust scores, etc).  It NEVER generates explanations itself —
    only narrates what the algorithms produced.

Public API
----------
LLMNarrator(api_key)
    .narrate(payload, audience)  -> NarrationResult
    .narrate_learner(payload)    -> str
    .narrate_instructor(payload) -> str
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompts — strict narration-only, no explanation invention
# ──────────────────────────────────────────────────────────────────────────────

LEARNER_SYSTEM_PROMPT = """
You are a supportive learning coach narrating pre-computed AI analysis results for a student.

STRICT RULES:
- ONLY describe what is present in the data payload provided to you.
- NEVER invent, guess, or fabricate explanations, causes, or details.
- NEVER say "I think" or "probably" — state only what the data shows.
- Keep your response under 100 words.
- Use plain, encouraging, everyday language — no jargon.
- End with exactly ONE specific, actionable sentence starting with "Your top priority:"
- Do not mention SHAP, DiCE, algorithms, or model internals by name.
"""

INSTRUCTOR_SYSTEM_PROMPT = """
You are an educational data analyst summarising pre-computed XAI model results for an instructor.

STRICT RULES:
- ONLY report what is in the data payload. Do not invent interpretations.
- NEVER say "I believe" or "might be" — report only what the data shows.
- Keep your response under 150 words.
- Be precise and data-driven. You may reference feature names, SHAP values, and confidence metrics.
- Structure your response as: (1) Risk summary, (2) Key drivers, (3) Recommended intervention.
- End with exactly ONE specific intervention recommendation.
"""

LEARNER_USER_TEMPLATE = """
Here is the pre-computed analysis for this student:

Risk Level: {risk_label} (score: {risk_score:.2f})
Model Confidence: {confidence}

Top 3 factors contributing to this risk:
{top3_features}

Decision rule that applies: {anchor_rule}

Most similar past learners: {prototype_narrative}

Top recommended action: {top_action}

Causal factors (verified as directly causing risk): {causal_features}

Narrate this in plain, encouraging language for the student. Under 100 words. End with "Your top priority: [one action]".
"""

INSTRUCTOR_USER_TEMPLATE = """
Pre-computed XAI results for Learner {learner_id}:

Risk Score: {risk_score:.4f} ({risk_label}) | Model Confidence: {confidence}
Trust Score: {trust_score} ({trust_label})
Stability: {stability:.2f}

SHAP Top-3 (signed attribution):
{shap_top3}

Decision Rule (Anchors, precision={anchor_precision:.0%}):
  {anchor_rule}

Causal factors: {causal_features}
Correlational factors: {correlational_features}

Top Intervention: {top_action}
Estimated impact of top action: {top_action_impact}

Interaction note: {interaction_narrative}

Summarise for the instructor. Under 150 words. Structure: risk summary → key drivers → intervention.
"""


# ── Recommendation-specific prompts ───────────────────────────────────────────

RECO_LEARNER_SYSTEM_PROMPT = """
You are a supportive learning coach narrating pre-computed AI content recommendation results for a student.

STRICT RULES:
- ONLY describe what is present in the data payload provided to you.
- NEVER invent, guess, or fabricate explanations, causes, or details.
- NEVER say "I think" or "probably" — state only what the data shows.
- Keep your response under 100 words.
- Use plain, encouraging, everyday language — no jargon.
- End with exactly ONE specific, actionable sentence starting with "Your next step:"
- Do not mention SHAP, algorithms, or model internals by name.
"""

RECO_INSTRUCTOR_SYSTEM_PROMPT = """
You are an educational data analyst summarising pre-computed AI recommendation results for an instructor.

STRICT RULES:
- ONLY report what is in the data payload. Do not invent interpretations.
- NEVER say "I believe" or "might be" — report only what the data shows.
- Keep your response under 150 words.
- Be precise and data-driven. You may reference feature names and score values.
- Structure your response as: (1) Recommendation summary, (2) Key drivers, (3) Suggested follow-up.
- End with exactly ONE specific follow-up recommendation.
"""

RECO_LEARNER_USER_TEMPLATE = """
Here are the pre-computed recommendation results for this student:

Top recommended item: {item_id} (relevance score: {score:.3f})
Explanation confidence (trust score): {trust_score}

Top 3 reasons this item was recommended:
{top3_features}

Decision rule: {anchor_rule}

Similar learners who benefited: {prototype_summary}

Narrate this recommendation in plain, encouraging language for the student. Under 100 words. End with "Your next step: [one action]".
"""

RECO_INSTRUCTOR_USER_TEMPLATE = """
Pre-computed recommendation results for Learner {learner_id}:

Item recommended: {item_id} | Relevance score: {score:.4f}
Trust score: {trust_score} | SHAP stability: {stability:.2f}
Anchor precision: {anchor_precision:.0%}

SHAP Top-3 (signed attribution):
{shap_top3}

Decision rule: {anchor_rule}

Feature interactions: {interaction_note}
Diversity score: {diversity_score}

Summarise for the instructor. Under 150 words. Structure: recommendation summary → key drivers → follow-up.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NarrationResult:
    learner_text:     Optional[str]
    instructor_text:  Optional[str]
    model_used:       str
    tokens_used:      int = 0

    def to_dict(self) -> dict:
        return {
            "learner":     self.learner_text,
            "instructor":  self.instructor_text,
            "model_used":  self.model_used,
            "tokens_used": self.tokens_used,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Payload helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_top3(shap_values: dict) -> str:
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    lines = []
    for feat, val in sorted_features:
        direction = "increases" if val > 0 else "decreases"
        lines.append(f"  - {feat.replace('_', ' ')}: {direction} risk (SHAP={val:+.3f})")
    return "\n".join(lines)


def _format_top3_plain(shap_values: dict) -> str:
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    lines = []
    for feat, val in sorted_features:
        direction = "increasing" if val > 0 else "helping reduce"
        plain = feat.replace("_", " ")
        lines.append(f"  - {plain} ({direction} your risk)")
    return "\n".join(lines)


def _get_top_action_str(ranked_actions: list) -> str:
    if not ranked_actions:
        return "No actions available"
    a = ranked_actions[0]
    plain = a.get("plain_language") or f"Improve {a.get('feature', 'engagement')}"
    impact = a.get("estimated_impact", 0)
    return f"{plain} (estimated {impact*100:.0f}% risk reduction)"


def _split_causal(causal_annotations: list) -> tuple[list[str], list[str]]:
    causal, corr = [], []
    for ann in causal_annotations:
        feat = ann.get("feature", "").replace("_", " ")
        if ann.get("type") == "causal":
            causal.append(feat)
        else:
            corr.append(feat)
    return causal, corr


def _build_reco_learner_payload(explain_resp: dict, learner_id: str) -> str:
    shap = explain_resp.get("shap_values", {})
    item_id    = explain_resp.get("item_id", "recommended item")
    score      = float(explain_resp.get("score", 0.0))
    trust      = explain_resp.get("trust_score") or {}
    trust_val  = trust.get("trust_score", "N/A") if isinstance(trust, dict) else "N/A"
    anchor     = explain_resp.get("anchor_rule") or "No specific rule derived"
    protos     = explain_resp.get("prototypes") or []
    proto_summary = (
        f"{len(protos)} similar learners found with avg similarity "
        f"{round(sum(p.get('similarity', 0) for p in protos) / len(protos), 2)}"
        if protos else "No similar learner data available"
    )
    return RECO_LEARNER_USER_TEMPLATE.format(
        item_id         = item_id,
        score           = score,
        trust_score     = trust_val,
        top3_features   = _format_top3_plain(shap),
        anchor_rule     = anchor,
        prototype_summary = proto_summary,
    )


def _build_reco_instructor_payload(explain_resp: dict, learner_id: str) -> str:
    shap        = explain_resp.get("shap_values", {})
    item_id     = explain_resp.get("item_id", "recommended item")
    score       = float(explain_resp.get("score", 0.0))
    stability   = float(explain_resp.get("shap_stability", 0.0))
    trust       = explain_resp.get("trust_score") or {}
    trust_val   = trust.get("trust_score", "N/A") if isinstance(trust, dict) else "N/A"
    anchor      = explain_resp.get("anchor_rule") or "No anchor rule derived"
    anchor_prec = float(explain_resp.get("anchor_precision", 0.0))
    interactions = explain_resp.get("feature_interactions") or []
    int_note    = (
        f"{interactions[0]['features'][0]} ↔ {interactions[0]['features'][1]} "
        f"({interactions[0].get('direction', 'unknown')})"
        if interactions else "No significant interactions"
    )
    diversity   = explain_resp.get("diversity_score", "N/A")
    return RECO_INSTRUCTOR_USER_TEMPLATE.format(
        learner_id      = learner_id,
        item_id         = item_id,
        score           = score,
        trust_score     = trust_val,
        stability       = stability,
        anchor_precision = anchor_prec,
        shap_top3       = _format_top3(shap),
        anchor_rule     = anchor,
        interaction_note = int_note,
        diversity_score  = diversity,
    )


def _build_learner_payload(explain_resp: dict, learner_id: str) -> str:
    shap = explain_resp.get("shap_values", {})
    risk_score = explain_resp.get("risk_score", 0.0)
    risk_label = explain_resp.get("risk_label", "unknown").upper()
    unc = explain_resp.get("uncertainty", {})
    confidence = unc.get("uncertainty_label", "unknown") if unc else "unknown"

    anchor = explain_resp.get("anchor_rule") or {}
    anchor_rule = anchor.get("human_readable") or anchor.get("anchor_rule") or "No specific rule derived"

    proto = explain_resp.get("prototypes") or {}
    prototype_narrative = proto.get("narrative") or "No similar learner data available"

    ranked = explain_resp.get("ranked_actions") or []
    top_action = _get_top_action_str(ranked)

    causal_anns = explain_resp.get("causal_annotations") or []
    causal_feats, _ = _split_causal(causal_anns)
    causal_str = ", ".join(causal_feats[:3]) if causal_feats else "none identified"

    return LEARNER_USER_TEMPLATE.format(
        risk_label=risk_label,
        risk_score=risk_score,
        confidence=confidence,
        top3_features=_format_top3_plain(shap),
        anchor_rule=anchor_rule,
        prototype_narrative=prototype_narrative,
        top_action=top_action,
        causal_features=causal_str,
    )


def _build_instructor_payload(explain_resp: dict, learner_id: str) -> str:
    shap = explain_resp.get("shap_values", {})
    risk_score = explain_resp.get("risk_score", 0.0)
    risk_label = explain_resp.get("risk_label", "unknown").upper()
    stability = explain_resp.get("stability", 0.0)

    unc = explain_resp.get("uncertainty", {})
    confidence = unc.get("uncertainty_label", "unknown") if unc else "unknown"

    trust = explain_resp.get("trust_score") or {}
    trust_score = trust.get("trust_score", "N/A")
    trust_label = trust.get("label", "N/A")

    anchor = explain_resp.get("anchor_rule") or {}
    anchor_rule = anchor.get("anchor_rule") or "No anchor rule derived"
    anchor_precision = anchor.get("precision") or 0.0

    causal_anns = explain_resp.get("causal_annotations") or []
    causal_feats, corr_feats = _split_causal(causal_anns)

    ranked = explain_resp.get("ranked_actions") or []
    top_action = _get_top_action_str(ranked)
    top_impact = f"{ranked[0].get('estimated_impact', 0)*100:.0f}% predicted risk reduction" if ranked else "N/A"

    int_narrative = explain_resp.get("interaction_narrative") or "No significant interactions detected"

    return INSTRUCTOR_USER_TEMPLATE.format(
        learner_id=learner_id,
        risk_score=risk_score,
        risk_label=risk_label,
        confidence=confidence,
        trust_score=trust_score,
        trust_label=trust_label,
        stability=stability,
        shap_top3=_format_top3(shap),
        anchor_rule=anchor_rule,
        anchor_precision=anchor_precision,
        causal_features=", ".join(causal_feats[:4]) or "none identified",
        correlational_features=", ".join(corr_feats[:4]) or "none identified",
        top_action=top_action,
        top_action_impact=top_impact,
        interaction_narrative=int_narrative,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Narrator class
# ──────────────────────────────────────────────────────────────────────────────

class LLMNarrator:
    """
    Narrates pre-computed XAI outputs via Groq.

    Parameters
    ----------
    api_key  : Groq API key. Falls back to GROQ_API_KEY env var.
    model    : Groq model ID. Default: llama-3.3-70b-versatile.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.model = model
        self._client = None
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if resolved_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=resolved_key)
                log.info("LLMNarrator initialised  model=%s", model)
            except ImportError:
                log.warning("groq package not installed — narration disabled. Run: pip install groq")
        else:
            log.warning("GROQ_API_KEY not set — narration disabled. Set env var to enable.")

    @property
    def available(self) -> bool:
        return self._client is not None

    def _call(self, system_prompt: str, user_message: str) -> tuple[str, int]:
        """Make a Groq API call. Returns (text, tokens_used)."""
        if not self._client:
            return "[Narration unavailable — GROQ_API_KEY not configured]", 0
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user",   "content": user_message.strip()},
                ],
                max_tokens=256,
                temperature=0.3,
            )
            text   = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0
            return text, tokens
        except Exception as e:
            log.error("Groq narration error: %s", e)
            return f"[Narration error: {e}]", 0

    def narrate_learner(self, explain_resp: dict, learner_id: str = "you") -> str:
        """Generate motivational learner narrative from pre-computed XAI data."""
        payload = _build_learner_payload(explain_resp, learner_id)
        text, _ = self._call(LEARNER_SYSTEM_PROMPT, payload)
        return text

    def narrate_instructor(self, explain_resp: dict, learner_id: str = "learner") -> str:
        """Generate technical instructor narrative from pre-computed XAI data."""
        payload = _build_instructor_payload(explain_resp, learner_id)
        text, _ = self._call(INSTRUCTOR_SYSTEM_PROMPT, payload)
        return text

    def narrate(
        self,
        explain_resp: dict,
        learner_id: str = "learner",
        audience: str = "both",
        context_type: str = "dropout",
    ) -> NarrationResult:
        """
        Generate narrations for the requested audience(s).

        Parameters
        ----------
        explain_resp : Full /explain or /recommend/explain response dict
        learner_id   : Learner identifier (for instructor narrative)
        audience     : "learner" | "instructor" | "both"
        context_type : "dropout" | "recommendation" — selects prompt templates
        """
        learner_text    = None
        instructor_text = None
        total_tokens    = 0

        if context_type == "recommendation":
            if audience in ("learner", "both"):
                pl = _build_reco_learner_payload(explain_resp, learner_id)
                learner_text, tok = self._call(RECO_LEARNER_SYSTEM_PROMPT, pl)
                total_tokens += tok
            if audience in ("instructor", "both"):
                pi = _build_reco_instructor_payload(explain_resp, learner_id)
                instructor_text, tok = self._call(RECO_INSTRUCTOR_SYSTEM_PROMPT, pi)
                total_tokens += tok
        else:
            if audience in ("learner", "both"):
                pl = _build_learner_payload(explain_resp, learner_id)
                learner_text, tok = self._call(LEARNER_SYSTEM_PROMPT, pl)
                total_tokens += tok
            if audience in ("instructor", "both"):
                pi = _build_instructor_payload(explain_resp, learner_id)
                instructor_text, tok = self._call(INSTRUCTOR_SYSTEM_PROMPT, pi)
                total_tokens += tok

        return NarrationResult(
            learner_text=learner_text,
            instructor_text=instructor_text,
            model_used=self.model,
            tokens_used=total_tokens,
        )
