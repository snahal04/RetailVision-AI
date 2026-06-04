"""
Part A detection pipeline: YOLOv8 + ByteTrack + Re-ID + zones -> structured events.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from ultralytics import YOLO

from pipeline.config import (
    CONF_THRESH,
    DEFAULT_MODEL,
    DWELL_INTERVAL_MS,
    LAYOUT_PATH,
    OUTPUT_DIR,
    PERSON_CLASS,
    VIDEOS_DIR,
)
from pipeline.entry_exit import check_line_crossing
from pipeline.reid import ReIDManager
from pipeline.schemas import EventMetadata, EventType, StoreEvent, frame_to_timestamp
from pipeline.session import SessionRegistry
from pipeline.staff import StaffDetector
from pipeline.zones import (
    CameraConfig,
    ZoneDef,
    build_camera_config,
    count_in_zone,
    load_camera_specs,
    zones_at_point,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    visitor_id: str
    prev_centroid: tuple[int, int] | None = None
    zones_inside: set[str] = field(default_factory=set)
    zone_enter_ms: dict[str, int] = field(default_factory=dict)
    dwell_last_emit_ms: dict[str, int] = field(default_factory=dict)
    billing_active: bool = False
    billing_had_queue: bool = False
    visited_pos: bool = False


class DetectionPipeline:
    def __init__(
        self,
        layout_path: Path = LAYOUT_PATH,
        model_name: str = DEFAULT_MODEL,
        max_frames: int | None = None,
    ) -> None:
        self.layout_path = layout_path
        self.store_id, self.clip_start_utc, self._camera_specs = load_camera_specs(layout_path)
        self.max_frames = max_frames
        self.model = YOLO(model_name)
        self.reid = ReIDManager()
        self.staff = StaffDetector()
        self.sessions = SessionRegistry()
        self.events: list[StoreEvent] = []
        self._tracks: dict[int, TrackState] = {}

    def run_all(self, video_filter: str | None = None) -> list[StoreEvent]:
        for spec in self._camera_specs:
            vf = spec["video_file"]
            cid = spec["camera_id"]
            if video_filter and video_filter not in (vf, cid):
                continue
            video_path = VIDEOS_DIR / vf
            if not video_path.exists():
                logger.warning("Skipping missing video: %s", video_path)
                continue
            self._tracks.clear()
            self.process_video(spec, video_path)
        return self.events

    def process_video(self, spec: dict, video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(spec.get("fps_override") or cap.get(cv2.CAP_PROP_FPS) or 15.0)
        cam = build_camera_config(spec, width, height)

        logger.info(
            "Processing %s (%s) %dx%d @ %.1f fps",
            cam.camera_id,
            video_path.name,
            width,
            height,
            fps,
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.max_frames is not None and frame_idx >= self.max_frames:
                break

            self._process_frame(frame, frame_idx, fps, cam)
            if frame_idx % 300 == 0:
                self.reid.purge_stale_gallery()
            frame_idx += 1

        cap.release()
        logger.info("Finished %s — %d frames", cam.camera_id, frame_idx)

    def _process_frame(
        self,
        frame,
        frame_idx: int,
        fps: float,
        cam: CameraConfig,
    ) -> None:
        results = self.model.track(
            frame,
            classes=[PERSON_CLASS],
            conf=CONF_THRESH,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )[0]

        if results.boxes.id is None:
            return

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        centroids: list[tuple[int, int]] = []
        rows: list[tuple[int, int, int, int, int, float]] = []

        for (x1, y1, x2, y2), conf, tid in zip(boxes, confs, track_ids):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2) // 2, int(y2)
            centroids.append((cx, cy))
            rows.append((tid, x1, y1, x2, y2, float(conf)))

        billing_zone = _zone_by_id(cam.zones, "BILLING")
        pos_zone = _zone_by_id(cam.zones, "POS")
        queue_depth = 0
        if billing_zone:
            in_billing = count_in_zone(centroids, billing_zone)
            in_pos = count_in_zone(centroids, pos_zone) if pos_zone else 0
            queue_depth = max(0, in_billing - in_pos)

        ts = frame_to_timestamp(self.clip_start_utc, frame_idx, fps)
        now_ms = int(frame_idx * 1000 / max(fps, 1e-6))

        for (tid, x1, y1, x2, y2, det_conf), (cx, cy) in zip(rows, centroids):
            crop = frame[max(0, y1) : y2, max(0, x1) : x2]
            is_staff, staff_score = self.staff.classify(crop)
            confidence = float(det_conf) if not is_staff else max(float(det_conf), staff_score)

            if tid not in self._tracks:
                visitor_id, _is_reentry = self.reid.identify(crop, tid)
                self._tracks[tid] = TrackState(visitor_id=visitor_id)
            else:
                self.reid.identify(crop, tid)

            state = self._tracks[tid]
            visitor_id = state.visitor_id

            if cam.entry_line and state.prev_centroid is not None:
                crossing = check_line_crossing(
                    cam.entry_line,
                    visitor_id,
                    state.prev_centroid,
                    (cx, cy),
                )
                if crossing:
                    if crossing.direction == "ENTRY":
                        if self.reid.consume_reentry_flag(tid):
                            self._emit(
                                cam, visitor_id, EventType.REENTRY, ts,
                                zone_id=None, dwell_ms=0,
                                is_staff=is_staff, confidence=confidence,
                                sku_zone=None, queue_depth=None,
                            )
                        else:
                            self._emit(
                                cam, visitor_id, EventType.ENTRY, ts,
                                zone_id=None, dwell_ms=0,
                                is_staff=is_staff, confidence=confidence,
                                sku_zone=None, queue_depth=None,
                            )
                    else:
                        self._emit(
                            cam, visitor_id, EventType.EXIT, ts,
                            zone_id=None, dwell_ms=0,
                            is_staff=is_staff, confidence=confidence,
                            sku_zone=None, queue_depth=None,
                        )
                        self.reid.on_exit(tid)

            if cam.zones:
                self._update_zones(
                    cam, state, visitor_id, (cx, cy), ts, now_ms,
                    is_staff, confidence, billing_zone, pos_zone,
                    queue_depth,
                )

            state.prev_centroid = (cx, cy)

    def _update_zones(
        self,
        cam: CameraConfig,
        state: TrackState,
        visitor_id: str,
        point: tuple[int, int],
        ts: str,
        now_ms: int,
        is_staff: bool,
        confidence: float,
        billing_zone: ZoneDef | None,
        pos_zone: ZoneDef | None,
        queue_depth: int,
    ) -> None:
        current = {z.zone_id for z in zones_at_point(point, cam.zones)}
        prev = state.zones_inside

        for zid in current - prev:
            zone = _zone_by_id(cam.zones, zid)
            if not zone:
                continue
            state.zones_inside.add(zid)
            state.zone_enter_ms[zid] = now_ms
            state.dwell_last_emit_ms[zid] = now_ms

            if zid == "BILLING":
                state.billing_active = True
                state.billing_had_queue = queue_depth > 0
                state.visited_pos = False
                if queue_depth > 0:
                    self._emit(
                        cam, visitor_id, EventType.BILLING_QUEUE_JOIN, ts,
                        zone_id=zid, dwell_ms=0,
                        is_staff=is_staff, confidence=confidence,
                        sku_zone=zone.sku_zone, queue_depth=queue_depth,
                    )
                else:
                    self._emit(
                        cam, visitor_id, EventType.ZONE_ENTER, ts,
                        zone_id=zid, dwell_ms=0,
                        is_staff=is_staff, confidence=confidence,
                        sku_zone=zone.sku_zone, queue_depth=None,
                    )
            else:
                self._emit(
                    cam, visitor_id, EventType.ZONE_ENTER, ts,
                    zone_id=zid, dwell_ms=0,
                    is_staff=is_staff, confidence=confidence,
                    sku_zone=zone.sku_zone, queue_depth=None,
                )

        for zid in prev - current:
            zone = _zone_by_id(cam.zones, zid)
            if not zone:
                continue
            dwell_ms = now_ms - state.zone_enter_ms.get(zid, now_ms)
            state.zones_inside.discard(zid)

            if zid == "BILLING" and state.billing_had_queue and not state.visited_pos:
                self._emit(
                    cam, visitor_id, EventType.BILLING_QUEUE_ABANDON, ts,
                    zone_id=zid, dwell_ms=dwell_ms,
                    is_staff=is_staff, confidence=confidence,
                    sku_zone=zone.sku_zone, queue_depth=queue_depth,
                )
                state.billing_active = False
                state.billing_had_queue = False
            else:
                self._emit(
                    cam, visitor_id, EventType.ZONE_EXIT, ts,
                    zone_id=zid, dwell_ms=dwell_ms,
                    is_staff=is_staff, confidence=confidence,
                    sku_zone=zone.sku_zone, queue_depth=None,
                )

            state.zone_enter_ms.pop(zid, None)
            state.dwell_last_emit_ms.pop(zid, None)

        if pos_zone and pos_zone.zone_id in current:
            state.visited_pos = True

        for zid in current:
            zone = _zone_by_id(cam.zones, zid)
            if not zone:
                continue
            enter_ms = state.zone_enter_ms.get(zid, now_ms)
            last_emit = state.dwell_last_emit_ms.get(zid, enter_ms)
            elapsed = now_ms - last_emit
            if elapsed >= DWELL_INTERVAL_MS:
                self._emit(
                    cam, visitor_id, EventType.ZONE_DWELL, ts,
                    zone_id=zid, dwell_ms=DWELL_INTERVAL_MS,
                    is_staff=is_staff, confidence=confidence,
                    sku_zone=zone.sku_zone, queue_depth=None,
                )
                state.dwell_last_emit_ms[zid] = now_ms

    def _emit(
        self,
        cam: CameraConfig,
        visitor_id: str,
        event_type: EventType,
        timestamp: str,
        zone_id: str | None,
        dwell_ms: int,
        is_staff: bool,
        confidence: float,
        sku_zone: str | None,
        queue_depth: int | None,
    ) -> None:
        seq = self.sessions.next_seq(visitor_id)
        event = StoreEvent(
            store_id=self.store_id,
            camera_id=cam.camera_id,
            visitor_id=visitor_id,
            event_type=event_type,
            timestamp=timestamp,
            zone_id=zone_id,
            dwell_ms=dwell_ms,
            is_staff=is_staff,
            confidence=round(confidence, 3),
            metadata=EventMetadata(
                queue_depth=queue_depth,
                sku_zone=sku_zone,
                session_seq=seq,
            ),
        )
        self.events.append(event)
        logger.debug("Event %s %s %s", event_type.value, visitor_id, timestamp)

    def save_events(self, path: Path | None = None) -> Path:
        out = path or (OUTPUT_DIR / "events.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev.to_json_dict()) + "\n")
        summary = OUTPUT_DIR / "events_summary.json"
        by_type: dict[str, int] = {}
        for ev in self.events:
            by_type[ev.event_type.value] = by_type.get(ev.event_type.value, 0) + 1
        summary.write_text(
            json.dumps(
                {
                    "total_events": len(self.events),
                    "by_type": by_type,
                    "unique_visitors": len({e.visitor_id for e in self.events}),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote %d events -> %s", len(self.events), out)
        return out


def _zone_by_id(zones: list[ZoneDef], zone_id: str) -> ZoneDef | None:
    for z in zones:
        if z.zone_id == zone_id:
            return z
    return None
