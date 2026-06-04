# Engineering Choices

## 1. Detection model: YOLOv8n + ByteTrack

**Options considered**

| Option | Pros | Cons |
|--------|------|------|
| YOLOv8n | Fast, CUDA-friendly, built-in ByteTrack | Smaller boxes on distant subjects |
| RT-DETR | Accuracy | Heavier, more setup |
| MediaPipe | Lightweight | Weaker in crowded retail CCTV |
| VLM (GPT-4V / LLaVA) for detection | Flexible labels | Latency, cost, not frame-real-time |

**What AI suggested:** YOLOv8 + ByteTrack as default; optional OSNet for Re-ID.

**What we chose:** **YOLOv8n** (`yolov8n.pt`) with Ultralytics **ByteTrack** tracker, person class only, `conf=0.35`. Re-ID via **MobileNetV3-small** embeddings (not OSNet) for speed on GTX 1650-class GPUs.

**Why:** Best balance of real-time throughput and integration in one stack; no VLM in the hot path. Staff detection uses **HSV dark-uniform heuristic** (not VLM) — fast and explainable; miscalibration is visible in `verify_staff_output.mp4`.

**VLM note:** We did not use a VLM for zone classification; zones are polygons in `store_layout.json` calibrated manually. A VLM would help ambiguous layouts but adds non-determinism and API cost.

---

## 2. Event schema design

**Options considered**

- Flat CSV rows per detection
- Nested protobuf
- **JSON document per behavioural event** (required spec)

**What AI suggested:** Single Pydantic `StoreEvent` shared between pipeline and API; `metadata` bag for extensibility.

**What we chose:** `pipeline/schemas.py` as source of truth; `event_id` UUID v4; `visitor_id` prefixed `VIS_`; timestamps ISO-8601 UTC; `confidence` never dropped (low-conf events still emitted).

**Why:** Validates at ingest and generation; supports idempotent `POST /events/ingest` and audit via `raw_json` column.

---

## 3. API architecture: session layer over event store

**Options considered**

- Materialized views updated on ingest
- Stream processor (Flink/Kafka)
- **On-read session reconstruction** from append-only events

**What AI suggested:** Pre-aggregate metrics tables updated in ingest handler.

**What we chose:** **Append-only `events` table** + `build_sessions()` at query time for funnel/metrics.

**Why:** Simpler for challenge scope, idempotent ingest, easy to replay; session logic stays testable in `app/services/sessions.py`. Trade-off: heavier reads on large stores — acceptable for demo SQLite volumes.
