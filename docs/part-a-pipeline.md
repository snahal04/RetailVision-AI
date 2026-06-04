# Part A — Detection Pipeline

## Run

```powershell
cd store-intelligence
.\.venv\Scripts\Activate.ps1
pip install pydantic
python -m pipeline.run --max-frames 450
python -m pipeline.run --camera CAM_ENTRY_01
python -m pipeline.validate_events output/events.jsonl
```

## Output

- `output/events.jsonl` — one JSON object per line (schema below)
- `output/events_summary.json` — counts by event type

## Configure

Edit `store_layout.json`:

- `store_id`, `clip_start_utc`
- Per camera: `camera_id`, `video_file`, `entry_line` (entry cam), `zones[].polygon_norm` (0–1 coords)

Calibrate zones with `tests/Generate_EntryPoint.py` (click for pixel coords, convert to normalized).

## Stack

| Component | Choice |
|-----------|--------|
| Detection | YOLOv8n |
| Tracking | ByteTrack (Ultralytics) |
| Re-ID | MobileNetV3-small embeddings |
| Staff | Dark-uniform HSV heuristic |
| Events | Pydantic `StoreEvent` |

## Event types

`ENTRY`, `EXIT`, `REENTRY`, `ZONE_ENTER`, `ZONE_EXIT`, `ZONE_DWELL` (every 30s), `BILLING_QUEUE_JOIN`, `BILLING_QUEUE_ABANDON`
