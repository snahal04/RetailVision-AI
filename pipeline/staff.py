"""Staff vs customer classification (dark uniform heuristic)."""

from __future__ import annotations

import cv2
import numpy as np

UPPER_RATIO = 0.55
STAFF_THRESH = 0.50
MIN_H, MIN_W = 60, 30

DARK_MASK_RANGES = [
    {
        "lower": np.array([0, 0, 0], dtype=np.uint8),
        "upper": np.array([180, 60, 80], dtype=np.uint8),
    },
    {
        "lower": np.array([100, 30, 20], dtype=np.uint8),
        "upper": np.array([130, 120, 90], dtype=np.uint8),
    },
]


def _dark_pixel_ratio(crop_bgr: np.ndarray) -> float:
    h = crop_bgr.shape[0]
    torso = crop_bgr[: int(h * UPPER_RATIO), :]
    if torso.size == 0:
        return 0.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for rng in DARK_MASK_RANGES:
        mask |= cv2.inRange(hsv, rng["lower"], rng["upper"])
    total = mask.shape[0] * mask.shape[1]
    return int(np.sum(mask > 0)) / max(total, 1)


class StaffDetector:
    def __init__(self, threshold: float = STAFF_THRESH) -> None:
        self.threshold = threshold

    def classify(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return False, 0.0
        h, w = crop_bgr.shape[:2]
        if h < MIN_H or w < MIN_W:
            return False, 0.0
        score = _dark_pixel_ratio(crop_bgr)
        return score >= self.threshold, round(score, 3)
