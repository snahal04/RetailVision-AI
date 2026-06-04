"""Pydantic models for pipeline events (Part A schema)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    queue_depth: int | None = None
    sku_zone: str | None = None
    session_seq: int = Field(ge=1)


class StoreEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: EventType
    timestamp: str
    zone_id: str | None = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: EventMetadata

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_utc_iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def frame_to_timestamp(clip_start_utc: str, frame_idx: int, fps: float) -> str:
    """ISO-8601 UTC from clip start + frame offset."""
    base = datetime.fromisoformat(clip_start_utc.replace("Z", "+00:00"))
    offset_sec = frame_idx / max(fps, 1e-6)
    ts = base.timestamp() + offset_sec
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
