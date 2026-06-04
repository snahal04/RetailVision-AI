# PROMPT: Write tests for GET /health including STALE_FEED when last event is older than 10 minutes.
# CHANGES MADE: Injected old timestamp event via ingest; asserted warning list contains STALE_FEED.

import uuid
from datetime import datetime, timedelta, timezone

from app.models import EventRecord


def test_health_no_data(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("healthy", "degraded")


def test_health_stale_feed(client, db_session, sample_event):
    old = datetime.now(timezone.utc) - timedelta(minutes=15)
    rec = EventRecord(
        event_id=str(uuid.uuid4()),
        store_id="STORE_BLR_002",
        camera_id="CAM_ENTRY_01",
        visitor_id="VIS_OLD",
        event_type="ENTRY",
        timestamp=old,
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

    r = client.get("/health")
    body = r.json()
    assert body["status"] == "degraded"
    assert any("STALE_FEED" in w for w in body["warnings"])
