"""Zone polygons from store_layout.json (normalized coordinates)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.config import LAYOUT_PATH
from pipeline.entry_exit import EntryLine


@dataclass(frozen=True)
class ZoneDef:
    zone_id: str
    sku_zone: str | None
    polygon: np.ndarray


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    video_file: str
    role: str
    entry_line: EntryLine | None
    zones: list[ZoneDef]
    fps_override: float | None = None


@dataclass(frozen=True)
class StoreLayout:
    store_id: str
    clip_start_utc: str
    cameras: list[CameraConfig]


def _scale_polygon(norm_pts: list[list[float]], width: int, height: int) -> np.ndarray:
    pts = [[int(x * width), int(y * height)] for x, y in norm_pts]
    return np.array(pts, dtype=np.int32)


def point_in_zone(point: tuple[int, int], zone: ZoneDef) -> bool:
    return cv2.pointPolygonTest(zone.polygon, point, False) >= 0


def zones_at_point(point: tuple[int, int], zones: list[ZoneDef]) -> list[ZoneDef]:
    return [z for z in zones if point_in_zone(point, z)]


def count_in_zone(centroids: list[tuple[int, int]], zone: ZoneDef) -> int:
    return sum(1 for c in centroids if point_in_zone(c, zone))


def load_layout(layout_path: Path = LAYOUT_PATH) -> StoreLayout:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    return StoreLayout(
        store_id=data["store_id"],
        clip_start_utc=data["clip_start_utc"],
        cameras=[],  # built per-video in pipeline
    )


def load_camera_specs(layout_path: Path = LAYOUT_PATH) -> tuple[str, str, list[dict]]:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    return data["store_id"], data["clip_start_utc"], data["cameras"]


def build_camera_config(spec: dict, width: int, height: int) -> CameraConfig:
    entry_line = None
    if spec.get("entry_line"):
        entry_line = EntryLine(
            p1=tuple(spec["entry_line"]["p1"]),
            p2=tuple(spec["entry_line"]["p2"]),
        )

    zones = [
        ZoneDef(
            zone_id=z["zone_id"],
            sku_zone=z.get("sku_zone"),
            polygon=_scale_polygon(z["polygon_norm"], width, height),
        )
        for z in spec.get("zones", [])
    ]

    return CameraConfig(
        camera_id=spec["camera_id"],
        video_file=spec["video_file"],
        role=spec.get("role", "zone"),
        entry_line=entry_line,
        zones=zones,
        fps_override=spec.get("fps_override"),
    )
