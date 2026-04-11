"""
Event Schemas — Pydantic models for POST /events endpoint.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# All valid event types the frontend may emit
EventType = Literal[
    "page_view",
    "click",
    "scroll",
    "whatif_slider",
    "action_viewed",
    "action_dismissed",
    "explanation_revisit",
    "resource_download",
    "time_to_first_action",
    "prototype_click",
    "session_start",
    "session_end",
]


class EventPayload(BaseModel):
    learner_id:   str         = Field(min_length=1)
    session_id:   Optional[str]   = None
    event_type:   EventType
    event_target: Optional[str]   = None
    event_value:  Optional[float] = None   # time_ms, scroll_pct, slider_value, count
    page:         Optional[str]   = None
    extra:        Optional[Any]   = None   # arbitrary JSON metadata
    client_ts:    Optional[str]   = None   # ISO-8601 from browser


class EventBatchRequest(BaseModel):
    events: list[EventPayload] = Field(min_length=1, max_length=500)


class EventBatchResponse(BaseModel):
    recorded: int
    message:  str = "ok"
