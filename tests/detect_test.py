"""
tracker_test.py
---------------
Adds ByteTrack tracking on top of YOLOv8 detection.
Each person gets a persistent track ID across frames.
Crossing events (ENTRY / EXIT) are detected and printed to console.

Run:
    python tracker_test.py

Output:
    tracker_test_output.mp4   — annotated video with track IDs + crossing events
    crossing_events.txt       — raw crossing log
"""

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from EntryExitLine import draw_entry_line, check_line_crossing, side_of_line

# ── Config ────────────────────────────────────────────────────────────────────

VIDEO_PATH   = "videos/Entry_CAM.mp4"
OUTPUT_PATH  = "output/tracker_test_output.mp4"
LOG_PATH     = "logs/crossing_events.txt"
MODEL_NAME   = "yolov8n.pt"
CONF_THRESH  = 0.35
MAX_FRAMES   = 450          # ~30 seconds at 15fps; set None for full video

# Assign a unique colour per track ID (consistent across frames)
def track_color(track_id: int) -> tuple:
    rng = np.random.default_rng(track_id * 137)
    return tuple(int(c) for c in rng.integers(80, 230, 3))

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading {MODEL_NAME} ...")
model = YOLO(MODEL_NAME)
print("Model ready.\n")

# ── Open video ────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {width}x{height} @ {fps:.1f}fps  |  {total} total frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ── State ─────────────────────────────────────────────────────────────────────

# Last known centroid per track_id
prev_centroids: dict[int, tuple[int, int]] = {}

# Which side each track was on last frame (to avoid duplicate crossing events)
prev_sides: dict[int, str] = {}

# Crossing event log
crossing_log: list[str] = []

# Count entries and exits
entry_count = 0
exit_count  = 0

# Trail: last N centroids per track for drawing movement path
TRAIL_LEN = 20
trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

# ── Processing loop ───────────────────────────────────────────────────────────

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if MAX_FRAMES and frame_idx >= MAX_FRAMES:
        break

    # ── YOLO + ByteTrack ──────────────────────────────────────────────────
    results = model.track(
        frame,
        classes=[0],            # persons only
        conf=CONF_THRESH,
        tracker="bytetrack.yaml",
        persist=True,           # CRITICAL: keeps track IDs consistent across frames
        verbose=False,
    )[0]

    # ── Draw entry line ───────────────────────────────────────────────────
    annotated = draw_entry_line(frame, show_sides=False)   # clean bg for tracking

    # Redraw entry line cleanly (thicker for visibility)
    from EntryExitLine import ENTRY_LINE_P1, ENTRY_LINE_P2
    cv2.line(annotated, ENTRY_LINE_P1, ENTRY_LINE_P2, (0, 0, 255), 3, cv2.LINE_AA)

    # ── Process each tracked detection ────────────────────────────────────
    if results.boxes.id is not None:
        boxes    = results.boxes.xyxy.cpu().numpy()
        confs    = results.boxes.conf.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, tid in zip(boxes, confs, track_ids):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx = (x1 + x2) // 2
            cy = int(y2)           # use bottom-centre as ground point (more accurate for line crossing)

            color = track_color(tid)

            # ── Trail ─────────────────────────────────────────────────
            trails[tid].append((cx, cy))
            if len(trails[tid]) > TRAIL_LEN:
                trails[tid].pop(0)

            trail = trails[tid]
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                tc = tuple(int(c * alpha) for c in color)
                cv2.line(annotated, trail[i-1], trail[i], tc, 2, cv2.LINE_AA)

            # ── Crossing detection ─────────────────────────────────────
            curr_side = side_of_line((cx, cy))

            if tid in prev_centroids:
                event = check_line_crossing(
                    visitor_id=f"TRK_{tid:04d}",
                    prev_centroid=prev_centroids[tid],
                    curr_centroid=(cx, cy),
                )
                if event:
                    if event.direction == "ENTRY":
                        entry_count += 1
                        ev_color = (0, 255, 0)
                        marker   = ">> ENTRY"
                    else:
                        exit_count += 1
                        ev_color = (0, 100, 255)
                        marker   = "<< EXIT"

                    log_line = (
                        f"Frame {frame_idx:04d} | {marker} | "
                        f"Track {tid:04d} | conf {conf:.2f} | "
                        f"centroid ({cx},{cy})"
                    )
                    crossing_log.append(log_line)
                    print(log_line)

                    # Flash marker on video
                    cv2.putText(
                        annotated, f"{marker} TRK_{tid:04d}",
                        (cx - 60, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        ev_color, 2, cv2.LINE_AA,
                    )

            prev_centroids[tid] = (cx, cy)
            prev_sides[tid]     = curr_side

            # ── Bounding box ───────────────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # ── Label: TRK_ID + conf ───────────────────────────────────
            label = f"TRK_{tid:04d}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # ── Ground point dot ───────────────────────────────────────
            cv2.circle(annotated, (cx, cy), 5, color, -1)

    # ── HUD ───────────────────────────────────────────────────────────────
    hud = f"Frame {frame_idx:04d}  |  ENTRIES: {entry_count}  EXITS: {exit_count}"
    cv2.rectangle(annotated, (0, 0), (520, 52), (0, 0, 0), -1)
    cv2.putText(annotated, hud, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    writer.write(annotated)

    if frame_idx % 30 == 0:
        print(f"  frame {frame_idx:04d}  — entries: {entry_count}  exits: {exit_count}")

    frame_idx += 1

cap.release()
writer.release()

# ── Save crossing log ─────────────────────────────────────────────────────────

with open(LOG_PATH, "w") as f:
    f.write("\n".join(crossing_log))

print(f"\n{'='*60}")
print(f"Processed {frame_idx} frames")
print(f"Total ENTRIES : {entry_count}")
print(f"Total EXITS   : {exit_count}")
print(f"Crossing log  → {LOG_PATH}")
print(f"Output video  → {OUTPUT_PATH}")
print(f"{'='*60}")
print("\nWhat to check in the output video:")
print("  1. Each person has a consistent TRK_XXXX ID across frames")
print("  2. ENTRY fires when someone walks IN through the door")
print("  3. EXIT fires when someone walks OUT")
print("  4. The trail shows the movement path behind each person")
print("  5. IDs should NOT change mid-track (if they do, lower CONF_THRESH)")