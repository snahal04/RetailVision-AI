# """
# staff_detector.py
# -----------------
# Classifies whether a detected person is store staff or a customer.

# Strategy:
#   Staff wear a consistent uniform colour. We sample a small set of
#   reference crops from the first N frames, build a colour histogram
#   "staff profile", and then score every new crop against it.

#   No external model needed — pure OpenCV HSV histogram comparison.
#   Fast enough to run on every detection.

# Two-phase workflow
# ──────────────────
# Phase 1 — Calibration (run once per camera):
#     python staff_detector.py --calibrate
#     Opens an interactive window. Press S on a staff crop, C to skip
#     (customer), Q when done. Saves staff_profile.npy.

# Phase 2 — Inference (used by the pipeline):
#     from staff_detector import StaffDetector
#     sd = StaffDetector("staff_profile.npy")
#     is_staff, confidence = sd.classify(crop_bgr)
# """

# import argparse
# import sys
# import json
# from pathlib import Path

# import cv2
# import numpy as np
# from ultralytics import YOLO


# # ── Config ────────────────────────────────────────────────────────────────────

# PROFILE_PATH   = "staff_profile.npy"
# STAFF_THRESH   = 0.55        # histogram similarity above this → staff
# H_BINS, S_BINS = 18, 16      # HSV histogram resolution
# UPPER_RATIO    = 0.45        # use top 45% of crop (torso/uniform, not legs)


# # ── Histogram helpers ─────────────────────────────────────────────────────────

# def _crop_upper(crop: np.ndarray) -> np.ndarray:
#     """Return the upper portion of a crop (torso area where uniform is visible)."""
#     h = crop.shape[0]
#     return crop[: int(h * UPPER_RATIO), :]


# def _hist(crop: np.ndarray) -> np.ndarray:
#     """Compute normalised 2-D HS histogram from a BGR crop."""
#     upper = _crop_upper(crop)
#     if upper.size == 0:
#         return np.zeros(H_BINS * S_BINS, dtype=np.float32)
#     hsv  = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist(
#         [hsv], [0, 1], None,
#         [H_BINS, S_BINS],
#         [0, 180, 0, 256],
#     )
#     cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
#     return hist.flatten()


# def _similarity(h1: np.ndarray, h2: np.ndarray) -> float:
#     """Bhattacharyya-based similarity (1 = identical, 0 = no overlap)."""
#     # cv2.HISTCMP_BHATTACHARYYA returns 0 (identical) → 1 (very different)
#     dist = cv2.compareHist(h1.reshape(H_BINS, S_BINS),
#                            h2.reshape(H_BINS, S_BINS),
#                            cv2.HISTCMP_BHATTACHARYYA)
#     return float(1.0 - dist)


# # ── StaffDetector ─────────────────────────────────────────────────────────────

# class StaffDetector:
#     """
#     Classifies a person crop as staff or customer.

#     Args:
#         profile_path: path to .npy file saved during calibration.
#                       If None or file missing, classify() always returns
#                       (False, 0.0) — safe default until calibrated.
#     """

#     def __init__(self, profile_path: str = PROFILE_PATH):
#         self._staff_hists: list[np.ndarray] = []
#         self._mean_hist: np.ndarray | None  = None

#         path = Path(profile_path)
#         if path.exists():
#             data = np.load(path, allow_pickle=True)
#             self._staff_hists = list(data)
#             self._mean_hist   = np.mean(data, axis=0)
#             print(f"[Staff] Loaded profile: {len(self._staff_hists)} reference crops "
#                   f"from '{profile_path}'")
#         else:
#             print(f"[Staff] No profile found at '{profile_path}'. "
#                   "Run with --calibrate first. Defaulting to customer for all.")

#     # ── Public API ────────────────────────────────────────────────────────

#     def classify(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
#         """
#         Args:
#             crop_bgr: BGR image crop of a detected person.

#         Returns:
#             (is_staff, confidence)
#             confidence is cosine similarity to staff profile (0–1).
#         """
#         if self._mean_hist is None or crop_bgr is None or crop_bgr.size == 0:
#             return False, 0.0

#         h, w = crop_bgr.shape[:2]
#         if h < 32 or w < 16:
#             return False, 0.0

#         query_hist = _hist(crop_bgr)
#         score      = _similarity(query_hist, self._mean_hist)
#         is_staff   = score >= STAFF_THRESH

#         return is_staff, round(score, 3)

#     def is_calibrated(self) -> bool:
#         return self._mean_hist is not None


# # ── Interactive calibration ───────────────────────────────────────────────────

# def calibrate(video_path: str, profile_path: str, max_frames: int = 900) -> None:
#     """
#     Interactive calibration tool.
#     Runs YOLO on the video, shows each detected person crop.
#     Press:  S = staff   C = customer (skip)   Q = quit and save
#     """
#     print(f"\nCalibrating staff profile from '{video_path}'")
#     print("Controls: S = mark as staff | C = skip | Q = quit & save\n")

#     model = YOLO("yolov8n.pt")
#     cap   = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         sys.exit(f"Cannot open {video_path}")

#     staff_hists: list[np.ndarray] = []
#     frame_idx = 0

#     while frame_idx < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Only sample every 15 frames to get variety
#         if frame_idx % 15 != 0:
#             frame_idx += 1
#             continue

#         results = model(frame, classes=[0], conf=0.4, verbose=False)[0]
#         boxes   = results.boxes.xyxy.cpu().numpy()

#         for (x1, y1, x2, y2) in boxes:
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             crop = frame[max(0,y1):y2, max(0,x1):x2]

#             if crop.size == 0 or crop.shape[0] < 32:
#                 continue

#             # Show full frame with box + isolated crop side by side
#             display = frame.copy()
#             cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

#             # Resize crop for display
#             ch, cw = crop.shape[:2]
#             scale  = min(300 / ch, 200 / cw)
#             crop_display = cv2.resize(crop, (int(cw*scale), int(ch*scale)))

#             # Overlay crop top-left
#             ph, pw = crop_display.shape[:2]
#             display[20:20+ph, 20:20+pw] = crop_display
#             cv2.rectangle(display, (18, 18), (22+pw, 22+ph), (255, 255, 0), 2)

#             # Instructions
#             cv2.putText(display,
#                         f"Staff crops saved: {len(staff_hists)}  |  "
#                         "S=staff  C=customer  Q=save+quit",
#                         (20, display.shape[0] - 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#                         (255, 255, 255), 2, cv2.LINE_AA)

#             cv2.imshow("Calibrate — Staff Detector", display)
#             key = cv2.waitKey(0) & 0xFF

#             if key == ord("s"):
#                 h = _hist(crop)
#                 staff_hists.append(h)
#                 print(f"  [S] Saved staff crop #{len(staff_hists)}")
#             elif key == ord("q"):
#                 break

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#         frame_idx += 1

#     cap.release()
#     cv2.destroyAllWindows()

#     if not staff_hists:
#         print("No staff crops collected. Profile not saved.")
#         return

#     arr = np.array(staff_hists, dtype=np.float32)
#     np.save(profile_path, arr)
#     print(f"\nSaved {len(staff_hists)} staff histogram(s) → '{profile_path}'")
#     print(f"Mean similarity threshold: {STAFF_THRESH}")
#     print("You can now use StaffDetector in the pipeline.")


# # ── Verification pass ─────────────────────────────────────────────────────────

# def verify(video_path: str, profile_path: str, max_frames: int = 450) -> None:
#     """
#     Run the detector on the video and print staff/customer decisions.
#     Writes verify_staff_output.mp4 for visual review.
#     """
#     model    = YOLO("yolov8n.pt")
#     detector = StaffDetector(profile_path)
#     cap      = cv2.VideoCapture(video_path)

#     fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter("verify_staff_output.mp4", fourcc, fps, (width, height))

#     frame_idx   = 0
#     staff_count = 0
#     cust_count  = 0

#     while frame_idx < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame, classes=[0], conf=0.35, verbose=False)[0]
#         boxes   = results.boxes.xyxy.cpu().numpy()
#         confs   = results.boxes.conf.cpu().numpy()

#         annotated = frame.copy()

#         for (x1, y1, x2, y2), det_conf in zip(boxes, confs):
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             crop = frame[max(0,y1):y2, max(0,x1):x2]

#             is_staff, staff_conf = detector.classify(crop)

#             if is_staff:
#                 color = (0, 165, 255)     # orange = staff
#                 label = f"STAFF  {staff_conf:.2f}"
#                 staff_count += 1
#             else:
#                 color = (0, 220, 0)       # green = customer
#                 label = f"CUST   {staff_conf:.2f}"
#                 cust_count += 1

#             cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
#             (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#             cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
#             cv2.putText(annotated, label, (x1+3, y1-4),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

#         hud = f"Frame {frame_idx:04d}  |  STAFF: {staff_count}  CUSTOMERS: {cust_count}"
#         cv2.rectangle(annotated, (0, 0), (600, 48), (0,0,0), -1)
#         cv2.putText(annotated, hud, (12, 34),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

#         writer.write(annotated)
#         frame_idx += 1

#     cap.release()
#     writer.release()
#     print(f"\nVerification done over {frame_idx} frames.")
#     print(f"  Staff detections   : {staff_count}")
#     print(f"  Customer detections: {cust_count}")
#     print(f"Output → verify_staff_output.mp4")


# # ── CLI ───────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Staff detector calibration & verification")
#     parser.add_argument("--calibrate", action="store_true",
#                         help="Run interactive calibration to build staff profile")
#     parser.add_argument("--verify",    action="store_true",
#                         help="Run verification pass on video")
#     parser.add_argument("--video",     default="videos/Entry_CAM.mp4")
#     parser.add_argument("--profile",   default=PROFILE_PATH)
#     args = parser.parse_args()

#     if args.calibrate:
#         calibrate(args.video, args.profile)
#     elif args.verify:
#         verify(args.video, args.profile)
#     else:
#         print("Usage:")
#         print("  python staff_detector.py --calibrate   # build staff profile (interactive)")
#         print("  python staff_detector.py --verify      # verify on video")

"""
staff_detector.py
-----------------
Classifies whether a detected person is store staff or a customer.

Staff wear dark black formal uniforms.
We use HSV color analysis on the upper torso region — no calibration
file needed. The dark-black profile is hardcoded.

Usage:
    from staff_detector import StaffDetector
    sd = StaffDetector()
    is_staff, confidence = sd.classify(crop_bgr)

Verify:
    python staff_detector.py --verify --video videos/Entry_CAM.mp4
"""

import argparse
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────

UPPER_RATIO  = 0.55
STAFF_THRESH = 0.50
MIN_H, MIN_W = 60, 30

# ── HSV ranges for dark black uniform ─────────────────────────────────────────
# Dark/black clothing in HSV:
#   Hue:        any (black has no dominant hue)
#   Saturation: 0–60  (very low — achromatic)
#   Value:      0–80  (very dark)
# Also catches dark navy/charcoal:
#   Saturation: 0–90, Value: 0–90

DARK_MASK_RANGES = [
    {"lower": np.array([0,   0,   0],  dtype=np.uint8),
     "upper": np.array([180, 60,  80], dtype=np.uint8)},
    {"lower": np.array([100, 30,  20], dtype=np.uint8),
     "upper": np.array([130, 120, 90], dtype=np.uint8)},
]


def _dark_pixel_ratio(crop_bgr: np.ndarray) -> float:
    h = crop_bgr.shape[0]
    torso = crop_bgr[: int(h * UPPER_RATIO), :]
    if torso.size == 0:
        return 0.0
    hsv  = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for rng in DARK_MASK_RANGES:
        mask |= cv2.inRange(hsv, rng["lower"], rng["upper"])
    total  = mask.shape[0] * mask.shape[1]
    return int(np.sum(mask > 0)) / max(total, 1)


class StaffDetector:
    def __init__(self, threshold: float = STAFF_THRESH):
        self.threshold = threshold
        print(f"[Staff] Dark-uniform detector ready. Threshold={threshold:.2f}")

    def classify(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return False, 0.0
        h, w = crop_bgr.shape[:2]
        if h < MIN_H or w < MIN_W:
            return False, 0.0
        score    = _dark_pixel_ratio(crop_bgr)
        is_staff = score >= self.threshold
        return is_staff, round(score, 3)

    def is_calibrated(self) -> bool:
        return True


def verify(video_path: str, max_frames: int = 450) -> None:
    model    = YOLO("yolov8n.pt")
    detector = StaffDetector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open '{video_path}'")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output/verify_staff_output.mp4", fourcc, fps, (width, height))

    frame_idx = staff_det = cust_det = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0], conf=0.35, verbose=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        annotated = frame.copy()

        for (x1, y1, x2, y2), det_conf in zip(boxes, confs):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            is_staff, score = detector.classify(crop)

            if is_staff:
                color = (0, 140, 255)
                label = f"STAFF {score:.2f}"
                staff_det += 1
            else:
                color = (0, 210, 0)
                label = f"CUST  {score:.2f}"
                cust_det += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(annotated, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

            # Yellow box = torso region analysed
            torso_y2 = min(y2, y1 + int((y2-y1) * UPPER_RATIO))
            cv2.rectangle(annotated, (x1, y1), (x2, torso_y2), (255,255,0), 1)

        hud = (f"Frame {frame_idx:04d}  |  "
               f"STAFF (orange): {staff_det}   "
               f"CUSTOMERS (green): {cust_det}   "
               f"thresh={STAFF_THRESH:.2f}")
        cv2.rectangle(annotated, (0,0), (width, 50), (0,0,0), -1)
        cv2.putText(annotated, hud, (12,34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nVerification done — {frame_idx} frames.")
    print(f"  Staff   : {staff_det}")
    print(f"  Customer: {cust_det}")
    print(f"  Output  → verify_staff_output.mp4")
    print(f"\nToo many staff missed?  → lower STAFF_THRESH in script (currently {STAFF_THRESH})")
    print(f"Customers flagged?      → raise  STAFF_THRESH")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--video",  default="videos/Entry_CAM.mp4")
    args = parser.parse_args()

    if args.verify:
        verify(args.video)
    else:
        print("Usage:")
        print("  python staff_detector.py --verify --video videos/Entry_CAM.mp4")
        print("  python staff_detector.py --verify --video videos/Floor_CAM.mp4")