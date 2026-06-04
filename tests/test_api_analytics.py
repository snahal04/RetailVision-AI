# PROMPT: Generate pytest tests for store analytics API covering empty store metrics,
# conversion funnel with re-entry sessions not double-counting visitors, all-staff clips,
# and zero-purchase conversion rate handling.
# CHANGES MADE: Built minimal event sequences per scenario; funnel asserts session counts
# not visitor counts; fixed timestamps to same day window.

import uuid

from datetime import datetime, timezone

from app.models import EventRecord
from app.services.sessions import build_sessions


def _ingest(client, events):
    return client.post("/events/ingest", json={"events": events})


def _ev(sample_event, **kw):
    d = sample_event(**kw)
    d["event_id"] = str(uuid.uuid4())
    return d


def test_metrics_empty_store(client):
    r = client.get("/stores/STORE_EMPTY/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["unique_visitors"] == 0
    assert body["conversion_rate"] is None


def test_funnel_customer_journey(client, sample_event):
    vid = "VIS_FUNNEL01"
    events = [
        _ev(sample_event, visitor_id=vid, event_type="ENTRY", timestamp="2026-03-03T11:00:00Z"),
        _ev(sample_event, visitor_id=vid, event_type="ZONE_ENTER", zone_id="SKINCARE",
            timestamp="2026-03-03T11:01:00Z"),
        _ev(sample_event, visitor_id=vid, event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
            timestamp="2026-03-03T11:05:00Z", metadata={"queue_depth": 2, "sku_zone": "CHECKOUT", "session_seq": 3}),
        _ev(sample_event, visitor_id=vid, event_type="ZONE_ENTER", zone_id="POS",
            timestamp="2026-03-03T11:06:00Z"),
        _ev(sample_event, visitor_id=vid, event_type="EXIT", timestamp="2026-03-03T11:10:00Z"),
    ]
    _ingest(client, events)
    r = client.get("/stores/STORE_BLR_002/funnel")
    assert r.status_code == 200
    stages = {s["stage"]: s["count"] for s in r.json()["stages"]}
    assert stages["Entry"] >= 1
    assert stages["Purchase"] >= 1


def test_reentry_two_sessions_not_double_visitor(client, sample_event):
    vid = "VIS_REENTRY1"
    seq = [
        _ev(sample_event, visitor_id=vid, event_type="ENTRY", timestamp="2026-03-03T12:00:00Z"),
        _ev(sample_event, visitor_id=vid, event_type="EXIT", timestamp="2026-03-03T12:05:00Z"),
        _ev(sample_event, visitor_id=vid, event_type="REENTRY", timestamp="2026-03-03T12:30:00Z"),
        _ev(sample_event, visitor_id=vid, event_type="EXIT", timestamp="2026-03-03T12:40:00Z"),
    ]
    _ingest(client, seq)
    r = client.get("/stores/STORE_BLR_002/funnel")
    assert r.json()["sessions_total"] == 2
    m = client.get("/stores/STORE_BLR_002/metrics")
    assert m.json()["unique_visitors"] == 1


def test_all_staff_excluded_from_metrics(client, sample_event):
    events = [
        _ev(sample_event, visitor_id="VIS_STAFF1", is_staff=True, event_type="ENTRY",
            timestamp="2026-03-03T13:00:00Z"),
        _ev(sample_event, visitor_id="VIS_STAFF1", is_staff=True, event_type="ZONE_ENTER",
            zone_id="SKINCARE", timestamp="2026-03-03T13:01:00Z"),
    ]
    _ingest(client, events)
    r = client.get("/stores/STORE_BLR_002/metrics")
    assert r.json()["unique_visitors"] == 0


def test_zero_purchases_conversion(client, sample_event):
    vid = "VIS_NOPURCHASE"
    events = [
        _ev(sample_event, visitor_id=vid, event_type="ENTRY", timestamp="2026-03-03T14:00:00Z"),
        _ev(sample_event, visitor_id=vid, event_type="ZONE_ENTER", zone_id="SKINCARE",
            timestamp="2026-03-03T14:01:00Z"),
        _ev(sample_event, visitor_id=vid, event_type="EXIT", timestamp="2026-03-03T14:10:00Z"),
    ]
    _ingest(client, events)
    r = client.get("/stores/STORE_BLR_002/metrics")
    assert r.json()["conversion_rate"] == 0.0


def test_heatmap_low_confidence(client, sample_event):
    _ingest(client, [_ev(sample_event, timestamp="2026-03-03T15:00:00Z")])
    r = client.get("/stores/STORE_BLR_002/heatmap")
    assert r.json()["data_confidence"] == "LOW"


def test_build_sessions_unit(db_session, sample_event):
    ts = datetime(2026, 3, 3, 10, 0, tzinfo=timezone.utc)
    rec = EventRecord(
        event_id=str(uuid.uuid4()),
        store_id="STORE_BLR_002",
        camera_id="CAM_ENTRY_01",
        visitor_id="VIS_X",
        event_type="ENTRY",
        timestamp=ts,
        zone_id=None,
        dwell_ms=0,
        is_staff=False,
        confidence=0.9,
        queue_depth=None,
        sku_zone=None,
        session_seq=1,
        raw_json="{}",
    )
    db_session.add(rec)
    db_session.commit()
    sessions = build_sessions([rec])
    assert len(sessions) == 1
