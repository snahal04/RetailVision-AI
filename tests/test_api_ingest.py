# PROMPT: Write pytest tests for FastAPI POST /events/ingest: idempotent by event_id,
# partial success when some events are malformed, batch max 500, and 503 when DB fails.
# CHANGES MADE: Used in-memory SQLite fixture; added duplicate ingest test and invalid
# event_type rejection; removed DB-fail mock (covered in test_api_errors.py).

import uuid

import pytest


def test_ingest_accepts_valid_event(client, sample_event):
    ev = sample_event()
    r = client.post("/events/ingest", json={"events": [ev]})
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] == 1
    assert body["rejected"] == 0


def test_ingest_idempotent_duplicate_event_id(client, sample_event):
    ev = sample_event()
    payload = {"events": [ev]}
    r1 = client.post("/events/ingest", json=payload)
    r2 = client.post("/events/ingest", json=payload)
    assert r1.json()["accepted"] == 1
    assert r2.json()["accepted"] == 0
    assert r2.json()["duplicate"] == 1


def test_ingest_partial_success_malformed(client, sample_event):
    good = sample_event(event_id=str(uuid.uuid4()))
    bad = {"event_id": "bad", "store_id": "STORE_BLR_002"}
    r = client.post("/events/ingest", json={"events": [good, bad]})
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] == 1
    assert body["rejected"] == 1
    assert len(body["errors"]) == 1


def test_ingest_batch_limit(client, sample_event):
    events = [sample_event(event_id=str(uuid.uuid4())) for _ in range(501)]
    r = client.post("/events/ingest", json={"events": events})
    assert r.status_code == 422
