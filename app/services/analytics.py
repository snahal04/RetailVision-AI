"""Metrics, funnel, heatmap, and anomaly computations."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import HEATMAP_MIN_SESSIONS, STALE_FEED_MINUTES
from app.models import EventRecord
from app.schemas import (
    AnomaliesResponse,
    AnomalyItem,
    FunnelResponse,
    FunnelStage,
    HealthResponse,
    HeatmapResponse,
    HeatmapZone,
    MetricsResponse,
    StoreFeedStatus,
    ZoneDwellMetric,
)
from app.services.sessions import VisitSession, build_sessions


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _day_window(db: Session, store_id: str) -> tuple[datetime, datetime]:
    latest = (
        db.query(func.max(EventRecord.timestamp))
        .filter(EventRecord.store_id == store_id)
        .scalar()
    )
    end = latest if latest else _utc_now()
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, end


def _events_in_window(
    db: Session, store_id: str, start: datetime, end: datetime
) -> list[EventRecord]:
    return (
        db.query(EventRecord)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= start,
            EventRecord.timestamp <= end,
        )
        .order_by(EventRecord.timestamp)
        .all()
    )


def get_metrics(db: Session, store_id: str) -> MetricsResponse:
    start, end = _day_window(db, store_id)
    events = _events_in_window(db, store_id, start, end)
    sessions = build_sessions(events, customer_only=True)

    entry_sessions = [s for s in sessions if s.started_at]
    unique_visitors = len({s.visitor_id for s in entry_sessions})
    purchases = sum(1 for s in entry_sessions if s.had_purchase)
    conversion: float | None = None
    note = None
    if entry_sessions:
        conversion = round(purchases / len(entry_sessions), 4)
    else:
        note = "No customer entry sessions in window; conversion unavailable."

    zone_dwell: dict[str, list[int]] = defaultdict(list)
    zone_visits: dict[str, int] = defaultdict(int)
    for s in entry_sessions:
        for z, ms in s.zone_dwell_ms.items():
            zone_dwell[z].append(ms)
        for z in s.zones_visited:
            zone_visits[z] += 1

    avg_dwell = [
        ZoneDwellMetric(
            zone_id=z,
            avg_dwell_ms=round(sum(vals) / len(vals), 1) if vals else 0.0,
            visit_count=zone_visits.get(z, 0),
        )
        for z, vals in sorted(zone_dwell.items())
    ]

    queue_joins = [e for e in events if e.event_type == "BILLING_QUEUE_JOIN" and not e.is_staff]
    abandons = [e for e in events if e.event_type == "BILLING_QUEUE_ABANDON" and not e.is_staff]
    abandonment_rate: float | None = None
    if queue_joins:
        abandonment_rate = round(len(abandons) / len(queue_joins), 4)

    current_queue = 0
    if queue_joins:
        last = max(queue_joins, key=lambda e: e.timestamp)
        current_queue = last.queue_depth or 0

    return MetricsResponse(
        store_id=store_id,
        as_of=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        unique_visitors=unique_visitors,
        conversion_rate=conversion,
        avg_dwell_by_zone=avg_dwell,
        current_queue_depth=current_queue,
        queue_abandonment_rate=abandonment_rate,
        note=note,
    )


def get_funnel(db: Session, store_id: str) -> FunnelResponse:
    start, end = _day_window(db, store_id)
    events = _events_in_window(db, store_id, start, end)
    sessions = build_sessions(events, customer_only=True)

    total = len(sessions)
    entered = sum(1 for s in sessions if s.started_at)
    zone_visit = sum(1 for s in sessions if s.had_zone_visit)
    billing = sum(1 for s in sessions if s.had_billing_queue)
    purchase = sum(1 for s in sessions if s.had_purchase)

    counts = [
        ("Entry", entered),
        ("Zone Visit", zone_visit),
        ("Billing Queue", billing),
        ("Purchase", purchase),
    ]
    stages: list[FunnelStage] = []
    prev = None
    for name, count in counts:
        drop = None
        if prev is not None and prev > 0:
            drop = round((1 - count / prev) * 100, 2)
        stages.append(FunnelStage(stage=name, count=count, drop_off_pct=drop))
        prev = count

    return FunnelResponse(
        store_id=store_id,
        window_start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        window_end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        sessions_total=total,
        stages=stages,
    )


def get_heatmap(db: Session, store_id: str) -> HeatmapResponse:
    start, end = _day_window(db, store_id)
    events = _events_in_window(db, store_id, start, end)
    sessions = build_sessions(events, customer_only=True)

    visit_freq: dict[str, int] = defaultdict(int)
    dwell_totals: dict[str, list[int]] = defaultdict(list)
    for s in sessions:
        for z in s.zones_visited:
            visit_freq[z] += 1
        for z, ms in s.zone_dwell_ms.items():
            dwell_totals[z].append(ms)

    max_freq = max(visit_freq.values(), default=1)
    max_dwell = max((sum(v) / len(v) for v in dwell_totals.values() if v), default=1.0)

    zones: list[HeatmapZone] = []
    all_z = sorted(set(visit_freq) | set(dwell_totals))
    for z in all_z:
        freq = visit_freq.get(z, 0)
        avg_dwell = sum(dwell_totals[z]) / len(dwell_totals[z]) if dwell_totals[z] else 0.0
        norm_freq = (freq / max_freq) * 50 if max_freq else 0
        norm_dwell = (avg_dwell / max_dwell) * 50 if max_dwell else 0
        intensity = min(100, int(round(norm_freq + norm_dwell)))
        zones.append(
            HeatmapZone(
                zone_id=z,
                visit_frequency=freq,
                avg_dwell_ms=round(avg_dwell, 1),
                intensity=intensity,
            )
        )

    confidence = "HIGH" if len(sessions) >= HEATMAP_MIN_SESSIONS else "LOW"
    return HeatmapResponse(
        store_id=store_id,
        zones=zones,
        data_confidence=confidence,
        sessions_in_window=len(sessions),
    )


def get_anomalies(db: Session, store_id: str) -> AnomaliesResponse:
    now = _utc_now()
    start, end = _day_window(db, store_id)
    events = _events_in_window(db, store_id, start, end)
    sessions = build_sessions(events, customer_only=True)
    anomalies: list[AnomalyItem] = []

    queue_joins = [e for e in events if e.event_type == "BILLING_QUEUE_JOIN" and not e.is_staff]
    if queue_joins:
        latest_depth = max(queue_joins, key=lambda e: e.timestamp).queue_depth or 0
        if latest_depth >= 4:
            anomalies.append(
                AnomalyItem(
                    anomaly_type="QUEUE_SPIKE",
                    severity="CRITICAL" if latest_depth >= 6 else "WARN",
                    message=f"Billing queue depth is {latest_depth} (threshold 4).",
                    suggested_action="Open additional billing counter or redirect staff to checkout.",
                    detected_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    metadata={"queue_depth": latest_depth},
                )
            )

    metrics_today = get_metrics(db, store_id)
    seven_day_start = end - timedelta(days=7)
    hist_events = _events_in_window(db, store_id, seven_day_start, end - timedelta(days=1))
    hist_sessions = build_sessions(hist_events, customer_only=True)
    hist_rates = []
    for day in range(7):
        d0 = (seven_day_start + timedelta(days=day)).replace(hour=0, minute=0, second=0, microsecond=0)
        d1 = d0 + timedelta(days=1)
        day_sess = [s for s in hist_sessions if d0 <= s.started_at < d1]
        if day_sess:
            p = sum(1 for s in day_sess if s.had_purchase) / len(day_sess)
            hist_rates.append(p)

    if hist_rates and metrics_today.conversion_rate is not None:
        avg_7d = sum(hist_rates) / len(hist_rates)
        if avg_7d > 0 and metrics_today.conversion_rate < avg_7d * 0.7:
            anomalies.append(
                AnomalyItem(
                    anomaly_type="CONVERSION_DROP",
                    severity="WARN",
                    message=(
                        f"Today's conversion {metrics_today.conversion_rate:.1%} "
                        f"is below 70% of 7-day average {avg_7d:.1%}."
                    ),
                    suggested_action="Review staffing at billing and in-store promotions.",
                    detected_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    metadata={
                        "today": metrics_today.conversion_rate,
                        "avg_7d": round(avg_7d, 4),
                    },
                )
            )

    thirty_min_ago = end - timedelta(minutes=30)
    recent_zone = [
        e for e in events
        if e.event_type in ("ZONE_ENTER", "ZONE_DWELL")
        and not e.is_staff
        and e.zone_id
        and e.timestamp >= thirty_min_ago
    ]
    active_zones = {e.zone_id for e in recent_zone}
    all_zones = {e.zone_id for e in events if e.zone_id}
    for z in all_zones - active_zones:
        anomalies.append(
            AnomalyItem(
                anomaly_type="DEAD_ZONE",
                severity="INFO",
                message=f"Zone '{z}' had no visits in the last 30 minutes.",
                suggested_action="Verify camera coverage or run a zone promotion.",
                detected_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                metadata={"zone_id": z},
            )
        )

    if not sessions and events:
        anomalies.append(
            AnomalyItem(
                anomaly_type="ALL_STAFF_ACTIVITY",
                severity="INFO",
                message="Events present but no customer sessions detected (possible all-staff period).",
                suggested_action="Confirm staff classification thresholds on entry camera.",
                detected_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                metadata={},
            )
        )

    return AnomaliesResponse(store_id=store_id, anomalies=anomalies)


def get_health(db: Session) -> HealthResponse:
    now = _utc_now()
    store_ids = [r[0] for r in db.query(EventRecord.store_id).distinct().all()]
    if not store_ids:
        store_ids = ["STORE_BLR_002"]

    stores: list[StoreFeedStatus] = []
    warnings: list[str] = []
    stale_any = False

    for sid in store_ids:
        last = (
            db.query(func.max(EventRecord.timestamp))
            .filter(EventRecord.store_id == sid)
            .scalar()
        )
        if last is None:
            stores.append(StoreFeedStatus(store_id=sid, last_event_at=None, lag_seconds=None, status="NO_DATA"))
            warnings.append(f"Store {sid} has no ingested events.")
            continue
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        lag = (now - last).total_seconds()
        status = "STALE_FEED" if lag > STALE_FEED_MINUTES * 60 else "OK"
        if status == "STALE_FEED":
            stale_any = True
            warnings.append(f"STALE_FEED: {sid} last event {int(lag)}s ago.")
        stores.append(
            StoreFeedStatus(
                store_id=sid,
                last_event_at=last.strftime("%Y-%m-%dT%H:%M:%SZ"),
                lag_seconds=round(lag, 1),
                status=status,
            )
        )

    overall = "degraded" if stale_any else "healthy"
    return HealthResponse(status=overall, stores=stores, warnings=warnings)
