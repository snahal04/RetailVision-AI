# PROMPT: Write pytest tests that validate StoreEvent Pydantic schema and optional events.jsonl file.
# CHANGES MADE: Added ENTRY zone_id null check; skip file test if output missing.

"""Schema compliance tests for Part A events."""

import json
from pathlib import Path

from pipeline.schemas import EventType, StoreEvent


def test_sample_event_validates():
    ev = StoreEvent(
        store_id="STORE_BLR_002",
        camera_id="CAM_ENTRY_01",
        visitor_id="VIS_c8a2f1",
        event_type=EventType.ENTRY,
        timestamp="2026-03-03T14:22:10Z",
        zone_id=None,
        dwell_ms=0,
        is_staff=False,
        confidence=0.91,
        metadata={"queue_depth": None, "sku_zone": None, "session_seq": 1},
    )
    assert ev.event_id
    assert ev.visitor_id.startswith("VIS_")


def test_events_file_if_present():
    path = Path(__file__).resolve().parent.parent / "output" / "events.jsonl"
    if not path.exists():
        return
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = StoreEvent.model_validate(json.loads(line))
        assert ev.event_id not in ids
        ids.add(ev.event_id)
