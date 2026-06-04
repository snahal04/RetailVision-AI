# Store Intelligence

CCTV detection pipeline + REST analytics API for retail store intelligence.

## Quick start (no Docker required)

```powershell
cd store-intelligence
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pipeline.run
```

> Note: real CCTV/video assets are not stored in this repo. Add your own MP4 files locally in the `videos/` folder (for example, `videos/sample_store_video.mp4`) before running the pipeline.

**Terminal 1 — API** (use this if `docker` is not installed):

```powershell
.\scripts\start_api.ps1
```

**Terminal 2 — ingest & query:**

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/feed_events.py --file output/events.jsonl
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stores/STORE_BLR_002/metrics
```

### Docker (after installing Docker Desktop)

1. **Start Docker Desktop** and wait until it is fully running (whale icon steady).
2. **Restart PowerShell** (or Cursor terminal) so `docker` is on your PATH.
3. From `store-intelligence`:

```powershell
.\scripts\docker_up.ps1
# or: docker compose up --build -d
```

4. Feed events and test:

```powershell
python scripts/feed_events.py --file output/events.jsonl
curl http://localhost:8000/health
```

Stop containers: `docker compose down`

If port 8000 is busy (local uvicorn still running), stop it first or change the host port in `docker-compose.yml`.

## Detection pipeline (Part A)

```powershell
python -m pipeline.run
python -m pipeline.validate_events output/events.jsonl
```

Outputs: `output/events.jsonl`

## Intelligence API (Part B)

| Endpoint | Description |
|----------|-------------|
| `POST /events/ingest` | Batch ingest (max 500), idempotent by `event_id` |
| `GET /stores/{id}/metrics` | Visitors, conversion, dwell, queue |
| `GET /stores/{id}/funnel` | Session funnel with drop-off |
| `GET /stores/{id}/heatmap` | Zone intensity 0–100 |
| `GET /stores/{id}/anomalies` | Queue spike, conversion drop, dead zones |
| `GET /health` | Feed status, `STALE_FEED` if >10 min lag |

Local API (no Docker):

```powershell
uvicorn app.main:app --reload --port 8000
```

## Live dashboard (Part E)

```powershell
python scripts/live_dashboard.py --api http://127.0.0.1:8000 --store STORE_BLR_002
```

## Tests

```powershell
pytest
```

Requires **>70%** statement coverage on `app/`.

## Docs

- [DESIGN.md](DESIGN.md) — architecture + AI-assisted decisions  
- [CHOICES.md](CHOICES.md) — model, schema, API choices  
- [docs/part-a-pipeline.md](docs/part-a-pipeline.md) — pipeline calibration  

## Layout

| Path | Role |
|------|------|
| `pipeline/` | YOLO detection + event emission |
| `app/` | FastAPI intelligence API |
| `videos/` | Placeholder media folder; add local MP4s only |
| `store_layout.json` | Store, cameras, zones |
| `scripts/` | Feed events, live dashboard |
| `tests/` | API + schema tests |
