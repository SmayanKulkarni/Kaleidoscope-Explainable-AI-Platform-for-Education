"""
Event Schemas — Pydantic models for POST /events endpoint.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

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

    @field_validator("event_type", mode="before")
    @classmethod
    def normalize_event_type_aliases(cls, v: str) -> str:
        # Accept common frontend aliases and map to canonical event names
        alias_map = {
            "whatif_interaction": "whatif_slider",
            "action_view": "action_viewed",
            "action_dismiss": "action_dismissed",
            "focus_start": "session_start",
            "focus_end": "session_end",
        }
        if isinstance(v, str):
            return alias_map.get(v, v)
        return v


class EventBatchRequest(BaseModel):
    events: list[EventPayload] = Field(min_length=1, max_length=500)


class EventBatchResponse(BaseModel):
    recorded: int
    message:  str = "ok"
