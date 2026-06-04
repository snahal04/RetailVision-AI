"""API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from pipeline.schemas import StoreEvent


class IngestRequest(BaseModel):
    events: list[dict[str, Any]] = Field(..., max_length=500)


class IngestErrorItem(BaseModel):
    index: int
    event_id: str | None = None
    message: str


class IngestResponse(BaseModel):
    accepted: int
    duplicate: int
    rejected: int
    errors: list[IngestErrorItem]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    trace_id: str | None = None


class ZoneDwellMetric(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visit_count: int


class MetricsResponse(BaseModel):
    store_id: str
    as_of: str
    unique_visitors: int
    conversion_rate: float | None
    avg_dwell_by_zone: list[ZoneDwellMetric]
    current_queue_depth: int
    queue_abandonment_rate: float | None
    note: str | None = None


class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float | None


class FunnelResponse(BaseModel):
    store_id: str
    window_start: str
    window_end: str
    sessions_total: int
    stages: list[FunnelStage]


class HeatmapZone(BaseModel):
    zone_id: str
    visit_frequency: int
    avg_dwell_ms: float
    intensity: int


class HeatmapResponse(BaseModel):
    store_id: str
    zones: list[HeatmapZone]
    data_confidence: Literal["HIGH", "LOW"]
    sessions_in_window: int


class AnomalyItem(BaseModel):
    anomaly_type: str
    severity: Literal["INFO", "WARN", "CRITICAL"]
    message: str
    suggested_action: str
    detected_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnomaliesResponse(BaseModel):
    store_id: str
    anomalies: list[AnomalyItem]


class StoreFeedStatus(BaseModel):
    store_id: str
    last_event_at: str | None
    lag_seconds: float | None
    status: Literal["OK", "STALE_FEED", "NO_DATA"]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unavailable"]
    stores: list[StoreFeedStatus]
    warnings: list[str]


def parse_store_event(data: dict[str, Any]) -> StoreEvent:
    return StoreEvent.model_validate(data)
