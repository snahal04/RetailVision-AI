# PROMPT: Unit tests for analytics service functions (metrics, funnel, heatmap, anomalies)
# using SQLAlchemy session and EventRecord rows without HTTP.
# CHANGES MADE: Direct calls to get_metrics/get_funnel/get_heatmap/get_anomalies/get_health.

import uuid
from datetime import datetime, timezone

from app.models import EventRecord
from app.services.analytics import (
    get_anomalies,
    get_funnel,
    get_health,
    get_heatmap,
    get_metrics,
)


def _add(db, **kwargs):
    defaults = dict(
        event_id=str(uuid.uuid4()),
        store_id="STORE_BLR_002",
        camera_id="CAM_ENTRY_01",
        visitor_id="VIS_UNIT01",
        event_type="ENTRY",
        timestamp=datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc),
        zone_id=None,
        dwell_ms=0,
        is_staff=False,
        confidence=0.9,
        queue_depth=None,
        sku_zone=None,
        session_seq=1,
        raw_json="{}",
    )
    defaults.update(kwargs)
    db.add(EventRecord(**defaults))


def test_get_metrics_with_data(db_session):
    _add(db_session, event_type="ENTRY")
    _add(
        db_session,
        event_type="ZONE_ENTER",
        zone_id="SKINCARE",
        timestamp=datetime(2026, 3, 3, 12, 5, tzinfo=timezone.utc),
    )
    db_session.commit()
    m = get_metrics(db_session, "STORE_BLR_002")
    assert m.unique_visitors >= 1
    assert m.conversion_rate is not None


def test_get_funnel_and_heatmap(db_session):
    _add(db_session)
    _add(
        db_session,
        event_type="ZONE_ENTER",
        zone_id="FRAGRANCE",
        timestamp=datetime(2026, 3, 3, 12, 2, tzinfo=timezone.utc),
    )
    db_session.commit()
    assert get_funnel(db_session, "STORE_BLR_002").sessions_total >= 1
    h = get_heatmap(db_session, "STORE_BLR_002")
    assert h.data_confidence == "LOW"


def test_get_anomalies_queue_spike(db_session):
    _add(
        db_session,
        event_type="BILLING_QUEUE_JOIN",
        zone_id="BILLING",
        queue_depth=5,
        timestamp=datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc),
    )
    db_session.commit()
    a = get_anomalies(db_session, "STORE_BLR_002")
    types = {x.anomaly_type for x in a.anomalies}
    assert "QUEUE_SPIKE" in types


def test_get_health_with_store(db_session):
    _add(db_session)
    db_session.commit()
    h = get_health(db_session)
    assert h.status in ("healthy", "degraded")
    assert len(h.stores) >= 1
