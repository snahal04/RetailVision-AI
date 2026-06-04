"""
entry_line.py
-------------
Defines the entry/exit threshold line for Entry_CAM and provides
crossing-detection logic used by the detection pipeline.

Entry line endpoints (pixel coordinates on 1920x1080 frame):
  P1 = (1286, 243)   — top-right anchor near glass door
  P2 = (721,  816)   — bottom-left anchor near glass door

Direction convention:
  Cross from RIGHT side → LEFT side  = ENTRY  (person walking into store)
  Cross from LEFT  side → RIGHT side = EXIT   (person walking out)

  "Side" is determined by the sign of the cross product of the line
  vector with the vector from P1 to the point.  Positive = left side,
  Negative = right side (matches standard 2-D winding convention).
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

# ── Entry line definition ────────────────────────────────────────────────────

    # ENTRY_LINE_P1 = (1286, 243)   # top-right anchor
    # ENTRY_LINE_P2 = (721,  816)   # bottom-left anchor
ENTRY_LINE_P1 = (1136, 24)
ENTRY_LINE_P2 = (980, 171)
# ── Side classification ──────────────────────────────────────────────────────

def _cross_z(p1, p2, point) -> float:
    """Z-component of cross product (p2-p1) × (point-p1)."""
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - \
           (p2[1] - p1[1]) * (point[0] - p1[0])


def side_of_line(point: tuple[int, int]) -> Literal["inside", "outside", "on"]:
    """
    Classify which side of the entry line a point is on.

    Returns:
      "inside"  — store interior side (positive cross product)
      "outside" — outside / street side (negative cross product)
      "on"      — exactly on the line (rare in practice)
    """
    z = _cross_z(ENTRY_LINE_P1, ENTRY_LINE_P2, point)
    if z > 0:
        return "inside"
    elif z < 0:
        return "outside"
    return "on"


@dataclass
class CrossingEvent:
    direction: Literal["ENTRY", "EXIT"]
    visitor_id: str
    prev_point: tuple[int, int]
    curr_point: tuple[int, int]


def check_line_crossing(
    visitor_id: str,
    prev_centroid: tuple[int, int],
    curr_centroid: tuple[int, int],
) -> CrossingEvent | None:
    """
    Given a tracked person's centroid in two consecutive frames,
    return a CrossingEvent if they crossed the entry line, else None.

    ENTRY: moved from "outside" → "inside"
    EXIT:  moved from "inside"  → "outside"
    """
    prev_side = side_of_line(prev_centroid)
    curr_side = side_of_line(curr_centroid)

    if prev_side == curr_side or "on" in (prev_side, curr_side):
        return None

    if prev_side == "outside" and curr_side == "inside":
        direction = "ENTRY"
    else:
        direction = "EXIT"

    return CrossingEvent(
        direction=direction,
        visitor_id=visitor_id,
        prev_point=prev_centroid,
        curr_point=curr_centroid,
    )


# ── Visualisation helper ─────────────────────────────────────────────────────

def draw_entry_line(frame: np.ndarray, show_sides: bool = True) -> np.ndarray:
    """
    Draw the entry/exit threshold line on a frame (in-place copy).

    Args:
        frame:      BGR image (e.g. from cv2.imread / VideoCapture)
        show_sides: if True, shade the two sides faintly so direction
                    convention is easy to verify visually

    Returns:
        Annotated copy of the frame.
    """
    out = frame.copy()
    p1, p2 = ENTRY_LINE_P1, ENTRY_LINE_P2

    if show_sides:
        # Build a mask polygon for the "inside" (store interior) region
        h, w = frame.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        poly_inside  = np.array([p1, p2, [0, h], [0, 0]], dtype=np.int32)
        poly_outside = np.array([p1, p2, [w, h], [w, 0]], dtype=np.int32)

        overlay = out.copy()
        cv2.fillPoly(overlay, [poly_inside],  (0, 180, 0))    # green = inside
        cv2.fillPoly(overlay, [poly_outside], (0, 0, 180))    # blue  = outside
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

    # Main threshold line
    cv2.line(out, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)        # red line

    # Endpoint markers
    for pt in (p1, p2):
        cv2.circle(out, pt, 10, (0, 255, 0), -1)              # filled green dot
        cv2.circle(out, pt,  10, (255, 255, 255), 2)          # white ring

    # Labels
    cv2.putText(out, "ENTRY LINE", (p1[0] + 14, p1[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "P1 (1286,243)", (p1[0] + 14, p1[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(out, "P2 (721,816)",  (p2[0] + 14, p2[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Side labels
    if show_sides:
        cv2.putText(out, "INSIDE (store)",  (60,  80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0),   2, cv2.LINE_AA)
        cv2.putText(out, "OUTSIDE (street)", (1050, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255),  2, cv2.LINE_AA)

    return out


# ── Quick visual test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    VIDEO_PATH = PROJECT_ROOT / "videos" / "Entry_CAM.mp4"
    OUTPUT_IMG = PROJECT_ROOT / "output" / "entry_line_verify.jpg"
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {VIDEO_PATH} (cwd={Path.cwd()})")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame")

    annotated = draw_entry_line(frame, show_sides=True)
    cv2.imwrite(str(OUTPUT_IMG), annotated)
    print(f"Saved annotated frame -> {OUTPUT_IMG}")

    # Smoke-test crossing detection
    tests = [
        ("VIS_001", (1000, 600), (600, 400)),   # right→left = ENTRY
        ("VIS_002", (600, 400), (1000, 600)),   # left→right = EXIT
        ("VIS_003", (200, 200), (300, 300)),     # both inside = None
    ]
    print("\nCrossing detection smoke test:")
    for vid, prev, curr in tests:
        result = check_line_crossing(vid, prev, curr)
        print(f"  {vid}: {prev} -> {curr}  ->  {result.direction if result else 'no crossing'}")