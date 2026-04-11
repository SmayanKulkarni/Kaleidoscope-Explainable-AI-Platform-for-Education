"""
Implicit Signal Aggregator (Step 2)
=====================================
Aggregates raw interaction events + explicit feedback into per-learner
feature vectors used as input to the Engagement Autoencoder.

Produces 20 signals per learner:
  - 15 implicit signals  (from EventStore)
  -  5 explicit signals  (from FeedbackStore)

Learners with no events get all-zero implicit vectors (cold-start).

Public API
----------
ImplicitAggregator(event_store, feedback_store)
    .aggregate_learner(learner_id)  -> ImplicitSignals
    .aggregate_all()                -> dict[learner_id, ImplicitSignals]
    .to_matrix(signals_dict)        -> (np.ndarray, list[str], list[str])
                                       (matrix, feature_names, learner_ids)
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Signal names — used as feature names for the autoencoder and GBM
# ──────────────────────────────────────────────────────────────────────────────

IMPLICIT_FEATURE_NAMES: list[str] = [
    "dashboard_visits_weekly",
    "avg_explanation_time_sec",
    "whatif_interaction_count",
    "action_view_rate",
    "action_dismiss_rate",
    "scroll_depth_avg",
    "time_to_first_action_avg",
    "prototype_click_rate",
    "explanation_revisit_rate",
    "resource_download_count",
    "unique_features_explored",
    "session_count",
    "session_duration_avg_sec",
    "engagement_consistency",
    "days_active_on_platform",
]

EXPLICIT_FEATURE_NAMES: list[str] = [
    "avg_explanation_rating",
    "recommendation_follow_rate",
    "correction_count",
    "feedback_frequency",
    "latest_rating_trend",
]

ALL_ENGAGEMENT_FEATURE_NAMES: list[str] = IMPLICIT_FEATURE_NAMES + EXPLICIT_FEATURE_NAMES

N_IMPLICIT = len(IMPLICIT_FEATURE_NAMES)   # 15
N_EXPLICIT = len(EXPLICIT_FEATURE_NAMES)   # 5
N_TOTAL    = N_IMPLICIT + N_EXPLICIT       # 20

# Latent feature names appended to the 12 OULAD features during retraining
LATENT_FEATURE_NAMES: list[str] = [
    "engagement_latent_1",
    "engagement_latent_2",
    "engagement_latent_3",
]


# ──────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ImplicitSignals:
    learner_id: str

    # Implicit (15)
    dashboard_visits_weekly:      float = 0.0
    avg_explanation_time_sec:     float = 0.0
    whatif_interaction_count:     float = 0.0
    action_view_rate:             float = 0.0
    action_dismiss_rate:          float = 0.0
    scroll_depth_avg:             float = 0.0
    time_to_first_action_avg:     float = 0.0
    prototype_click_rate:         float = 0.0
    explanation_revisit_rate:     float = 0.0
    resource_download_count:      float = 0.0
    unique_features_explored:     float = 0.0
    session_count:                float = 0.0
    session_duration_avg_sec:     float = 0.0
    engagement_consistency:       float = 0.0
    days_active_on_platform:      float = 0.0

    # Explicit (5)
    avg_explanation_rating:       float = 0.0
    recommendation_follow_rate:   float = 0.0
    correction_count:             float = 0.0
    feedback_frequency:           float = 0.0
    latest_rating_trend:          float = 0.0

    def to_vector(self) -> list[float]:
        return [
            self.dashboard_visits_weekly,
            self.avg_explanation_time_sec,
            self.whatif_interaction_count,
            self.action_view_rate,
            self.action_dismiss_rate,
            self.scroll_depth_avg,
            self.time_to_first_action_avg,
            self.prototype_click_rate,
            self.explanation_revisit_rate,
            self.resource_download_count,
            self.unique_features_explored,
            self.session_count,
            self.session_duration_avg_sec,
            self.engagement_consistency,
            self.days_active_on_platform,
            self.avg_explanation_rating,
            self.recommendation_follow_rate,
            self.correction_count,
            self.feedback_frequency,
            self.latest_rating_trend,
        ]

    def is_cold_start(self) -> bool:
        """True if this learner has no real engagement data at all."""
        return self.session_count == 0 and self.avg_explanation_rating == 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cold_start"] = self.is_cold_start()
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Aggregator
# ──────────────────────────────────────────────────────────────────────────────

class ImplicitAggregator:
    """
    Aggregates raw events + feedback for one or all learners.

    Parameters
    ----------
    event_store    : EventStore instance
    feedback_store : FeedbackStore instance
    lookback_days  : only consider events within this window (default 90)
    """

    def __init__(self, event_store, feedback_store, lookback_days: int = 90):
        self._events   = event_store
        self._feedback = feedback_store
        self._lookback = lookback_days

    # ── Public ────────────────────────────────────────────────────────────────

    def aggregate_learner(self, learner_id: str) -> ImplicitSignals:
        since = datetime.now(timezone.utc) - timedelta(days=self._lookback)
        events = self._events.get_learner_events(learner_id, since=since)
        signals = self._compute_implicit(learner_id, events)
        self._fill_explicit(learner_id, signals)
        return signals

    def aggregate_all(self) -> dict[str, ImplicitSignals]:
        """Aggregate every learner that has at least one event. Merge with feedback-only learners."""
        learner_ids_with_events = set(self._events.get_all_learner_ids())
        # Also include learners who have only feedback (no events yet)
        learner_ids_with_feedback = set(
            r["learner_id"]
            for r in self._feedback.get_learner_feedback("")
        ) if hasattr(self._feedback, "get_all_learner_ids") else set()

        all_ids = learner_ids_with_events | learner_ids_with_feedback
        result: dict[str, ImplicitSignals] = {}
        for lid in all_ids:
            result[lid] = self.aggregate_learner(lid)
        log.info(
            "ImplicitAggregator.aggregate_all  learners=%d  with_events=%d",
            len(all_ids), len(learner_ids_with_events)
        )
        return result

    def to_matrix(
        self, signals_dict: dict[str, ImplicitSignals]
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Convert aggregated signals to numpy matrix.
        Returns (matrix, feature_names, learner_ids).
        matrix shape: (n_learners, 20)
        """
        learner_ids = sorted(signals_dict.keys())
        matrix = np.array(
            [signals_dict[lid].to_vector() for lid in learner_ids],
            dtype=np.float32,
        )
        return matrix, ALL_ENGAGEMENT_FEATURE_NAMES, learner_ids

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_implicit(self, learner_id: str, events: list[dict]) -> ImplicitSignals:
        if not events:
            return ImplicitSignals(learner_id=learner_id)

        s = ImplicitSignals(learner_id=learner_id)

        # ── Session signals ──────────────────────────────────────
        sessions: dict[str, list[dict]] = {}
        for e in events:
            sid = e.get("session_id") or "default"
            sessions.setdefault(sid, []).append(e)

        s.session_count = float(len(sessions))

        # Session durations (seconds from first to last event in session)
        durations = []
        for evs in sessions.values():
            ts_list = []
            for ev in evs:
                ts_str = ev.get("server_ts")
                if ts_str:
                    try:
                        ts_list.append(datetime.fromisoformat(ts_str.replace("Z", "+00:00")))
                    except ValueError:
                        pass
            if len(ts_list) >= 2:
                durations.append((max(ts_list) - min(ts_list)).total_seconds())
        s.session_duration_avg_sec = float(np.mean(durations)) if durations else 0.0

        # ── Days active ──────────────────────────────────────────
        active_days: set[str] = set()
        for e in events:
            ts_str = e.get("server_ts")
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    active_days.add(dt.date().isoformat())
                except ValueError:
                    pass
        s.days_active_on_platform = float(len(active_days))

        # ── Dashboard visits weekly ──────────────────────────────
        pv_events = [e for e in events if e["event_type"] == "page_view"
                     and e.get("page") == "dashboard"]
        weeks_span = max(1, self._lookback // 7)
        s.dashboard_visits_weekly = len(pv_events) / weeks_span

        # ── Avg time on explanation pages ────────────────────────
        explain_times = [
            e["event_value"] / 1000.0
            for e in events
            if e["event_type"] == "page_view"
            and e.get("page") in ("explain", "explanation", "dashboard")
            and e.get("event_value") is not None
        ]
        s.avg_explanation_time_sec = float(np.mean(explain_times)) if explain_times else 0.0

        # ── What-If interactions ──────────────────────────────────
        wi_events = [e for e in events if e["event_type"] == "whatif_slider"]
        s.whatif_interaction_count = float(len(wi_events))
        s.unique_features_explored = float(len(set(
            e["event_target"] for e in wi_events if e.get("event_target")
        )))

        # ── Action view / dismiss rates ───────────────────────────
        viewed    = [e for e in events if e["event_type"] == "action_viewed"]
        dismissed = [e for e in events if e["event_type"] == "action_dismissed"]
        total_actions = len(viewed) + len(dismissed)
        s.action_view_rate    = len(viewed)    / total_actions if total_actions else 0.0
        s.action_dismiss_rate = len(dismissed) / total_actions if total_actions else 0.0

        # ── Scroll depth ──────────────────────────────────────────
        scroll_vals = [
            e["event_value"]
            for e in events
            if e["event_type"] == "scroll" and e.get("event_value") is not None
        ]
        s.scroll_depth_avg = float(np.mean(scroll_vals)) if scroll_vals else 0.0

        # ── Time to first action (ms → sec) ───────────────────────
        tfa_vals = [
            e["event_value"] / 1000.0
            for e in events
            if e["event_type"] == "time_to_first_action" and e.get("event_value") is not None
        ]
        s.time_to_first_action_avg = float(np.mean(tfa_vals)) if tfa_vals else 0.0

        # ── Prototype clicks ──────────────────────────────────────
        proto_clicks = [e for e in events if e["event_type"] == "prototype_click"]
        proto_shown  = [e for e in events if e["event_type"] == "page_view"
                        and e.get("page") == "explain"]   # proxy: each explain view shows prototypes
        s.prototype_click_rate = (
            len(proto_clicks) / len(proto_shown) if proto_shown else 0.0
        )

        # ── Explanation revisit rate ──────────────────────────────
        revisit_events = [e for e in events if e["event_type"] == "explanation_revisit"]
        total_explain_views = len(proto_shown)
        s.explanation_revisit_rate = (
            len(revisit_events) / total_explain_views if total_explain_views else 0.0
        )

        # ── Resource downloads ────────────────────────────────────
        s.resource_download_count = float(sum(
            1 for e in events if e["event_type"] == "resource_download"
        ))

        # ── Engagement consistency (inverse std of daily event counts) ────────
        if active_days:
            day_counts: dict[str, int] = {}
            for e in events:
                ts_str = e.get("server_ts")
                if ts_str:
                    try:
                        day = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).date().isoformat()
                        day_counts[day] = day_counts.get(day, 0) + 1
                    except ValueError:
                        pass
            counts = list(day_counts.values())
            std = statistics.stdev(counts) if len(counts) >= 2 else 0.0
            # Lower std = more consistent. Normalise to [0,1]: 1/(1+std)
            s.engagement_consistency = 1.0 / (1.0 + std)
        else:
            s.engagement_consistency = 0.0

        return s

    def _fill_explicit(self, learner_id: str, signals: ImplicitSignals) -> None:
        """Pull explicit feedback signals from FeedbackStore and fill into signals."""
        try:
            records = self._feedback.get_learner_feedback(learner_id)
        except Exception as e:
            log.warning("Could not fetch feedback for %s: %s", learner_id, e)
            return

        if not records:
            return

        # avg_explanation_rating
        ratings = [r["rating"] for r in records if r.get("rating") is not None]
        if ratings:
            signals.avg_explanation_rating = float(np.mean(ratings))

        # recommendation_follow_rate
        follow_vals = [r["followed_recommendation"] for r in records
                       if r.get("followed_recommendation") is not None]
        if follow_vals:
            signals.recommendation_follow_rate = float(np.mean([float(v) for v in follow_vals]))

        # correction_count
        signals.correction_count = float(sum(
            1 for r in records if r.get("correction_feature") is not None
        ))

        # feedback_frequency (feedbacks per week in lookback window)
        weeks_span = max(1, self._lookback // 7)
        signals.feedback_frequency = len(records) / weeks_span

        # latest_rating_trend: slope of last 5 ratings (positive = improving)
        if len(ratings) >= 2:
            window = ratings[-5:]
            xs = list(range(len(window)))
            x_mean = np.mean(xs)
            y_mean = np.mean(window)
            numerator   = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, window))
            denominator = sum((x - x_mean) ** 2 for x in xs)
            slope = numerator / denominator if denominator != 0 else 0.0
            signals.latest_rating_trend = float(np.clip(slope, -5.0, 5.0))
