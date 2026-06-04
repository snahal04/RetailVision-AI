# Store Intelligence — Architecture

## Overview

The system has two layers:

1. **Detection pipeline (Part A)** — processes CCTV MP4s with YOLOv8 + ByteTrack, Re-ID, staff heuristics, and zone polygons from `store_layout.json`. Emits newline-delimited JSON events.
2. **Intelligence API (Part B)** — FastAPI service ingests events into SQLite, reconstructs **visit sessions**, and serves metrics, funnel, heatmap, anomalies, and health.

```
videos/ → pipeline/run.py → output/events.jsonl
                ↓
        scripts/feed_events.py → POST /events/ingest
                ↓
         SQLite (events) → analytics services → REST endpoints
                ↓
        scripts/live_dashboard.py (Part E)
```

## Session model

Analytics use **sessions**, not raw events:

- `ENTRY` / `REENTRY` opens a session (REENTRY = same `visitor_id`, new session — no double-count in funnel visitor metrics).
- `EXIT` closes a session.
- **Purchase** is inferred: `ZONE_ENTER` at `POS`, or `EXIT` after `BILLING_QUEUE_JOIN` without `BILLING_QUEUE_ABANDON`.

Staff (`is_staff=true`) are excluded from customer metrics.

## API layer

| Module | Role |
|--------|------|
| `app/models.py` | SQLAlchemy `EventRecord` |
| `app/services/ingest.py` | Validate (Pydantic), dedupe by `event_id` |
| `app/services/sessions.py` | Session reconstruction |
| `app/services/analytics.py` | Metrics, funnel, heatmap, anomalies, health |
| `app/middleware.py` | Structured JSON logs per request |

## Operations (Part C)

- **Docker**: `docker compose up` runs API on port 8000.
- **Idempotency**: unique constraint on `event_id`; duplicates return `duplicate` count.
- **503**: DB errors → structured `ErrorResponse`, no stack traces in JSON.
- **Logging**: `trace_id`, `store_id`, `endpoint`, `latency_ms`, `event_count`, `status_code`.

## AI-Assisted Decisions

1. **Session-based funnel vs event counts** — An LLM suggested counting `ENTRY` events for funnel width; we overrode to **session units** so REENTRY and track fragmentation do not inflate entry counts.
2. **Purchase proxy without POS feed** — AI proposed adding a `PURCHASE` event type; we kept the spec catalogue and inferred purchase from POS zone + billing behaviour instead.
3. **SQLite for challenge delivery** — Suggested Postgres-only; we used SQLite with Docker volume for zero-config `docker compose up`, still swappable via `DATABASE_URL`.
