"""
reid.py
-------
Re-identification module using lightweight appearance embeddings.

When a tracked person exits (EXIT event), their appearance embedding is
stored in a gallery. If they re-enter within RE_ENTRY_WINDOW_SEC seconds
and their embedding cosine-similarity to a gallery entry exceeds
REID_THRESHOLD, they are assigned the same visitor_id (REENTRY event).
Otherwise a fresh visitor_id is minted (ENTRY event).

No torchreid / OSNet dependency — uses a MobileNetV3 feature extractor
from torchvision that is already in your requirements. Fast enough to
run per-frame on CPU; even faster on the CUDA env you have.

Usage (imported by detect.py in the full pipeline):
    from reid import ReIDManager
    reid = ReIDManager()
    visitor_id, is_reentry = reid.identify(crop_bgr, track_id)
    reid.on_exit(track_id)
"""

import time
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

REID_THRESHOLD       = 0.82    # cosine similarity above this → same person
RE_ENTRY_WINDOW_SEC  = 600     # only match against exits within last 10 minutes
EMBED_DIM            = 576     # MobileNetV3-small penultimate layer output dim
MIN_CROP_SIZE        = 32      # skip crops smaller than this (px) — too noisy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Embedding extractor ───────────────────────────────────────────────────────

class EmbeddingExtractor(nn.Module):
    """
    MobileNetV3-small with the classification head removed.
    Outputs a 576-d L2-normalised embedding per crop.
    """
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # Keep everything up to (but not including) the classifier
        self.features   = base.features
        self.pool       = base.avgpool
        # base.classifier[0] is a Linear(576, 1024) — we want the 576-d vector
        self.project    = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)          # (B, 576)
        x = nn.functional.normalize(x, dim=1)
        return x


_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),           # standard Re-ID crop size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ── Gallery entry ─────────────────────────────────────────────────────────────

class GalleryEntry:
    def __init__(self, visitor_id: str, embedding: np.ndarray):
        self.visitor_id  = visitor_id
        self.embedding   = embedding      # shape (576,)  L2-normalised
        self.exit_time   = time.time()
        self.entry_count = 1              # how many times this person entered


# ── Re-ID Manager ─────────────────────────────────────────────────────────────

class ReIDManager:
    """
    Central Re-ID controller. One instance per video / per store camera session.

    Lifecycle per tracked person:
      1. Person appears  → call identify(crop, track_id)
                           returns (visitor_id, is_reentry)
      2. Person exits    → call on_exit(track_id)
                           embedding moves to gallery
      3. Person re-enters→ identify() matches gallery → REENTRY
    """

    def __init__(self):
        print(f"[ReID] Loading embedding extractor on {DEVICE} ...")
        self.extractor = EmbeddingExtractor().to(DEVICE).eval()
        print("[ReID] Ready.")

        # track_id → visitor_id  (active tracks)
        self._active: dict[int, str] = {}

        # track_id → latest embedding  (updated every N frames)
        self._track_embeddings: dict[int, np.ndarray] = {}

        # gallery of exited visitors  (visitor_id → GalleryEntry)
        self._gallery: dict[str, GalleryEntry] = {}

        # How often to refresh embedding for an active track (frames)
        self._embed_every = 10
        self._frame_counter: dict[int, int] = defaultdict(int)

    # ── Public API ────────────────────────────────────────────────────────

    def identify(
        self,
        crop_bgr: np.ndarray,
        track_id: int,
    ) -> tuple[str, bool]:
        """
        Assign a visitor_id to a track.

        Returns:
            (visitor_id, is_reentry)
            is_reentry=True  → emit REENTRY event
            is_reentry=False → emit ENTRY event (first time seen)
        """
        # Already known active track — return cached ID
        if track_id in self._active:
            self._maybe_refresh_embedding(crop_bgr, track_id)
            return self._active[track_id], False

        # New track — extract embedding and search gallery
        emb = self._embed(crop_bgr)
        if emb is None:
            # Crop too small / unusable — mint a new ID without gallery search
            vid = self._new_visitor_id()
            self._active[track_id] = vid
            return vid, False

        self._track_embeddings[track_id] = emb

        match, score = self._search_gallery(emb)

        if match is not None:
            # Re-entry detected
            visitor_id = match.visitor_id
            match.entry_count += 1
            # Remove from gallery so it's treated as active again
            del self._gallery[visitor_id]
            self._active[track_id] = visitor_id
            print(f"[ReID] REENTRY  track={track_id}  visitor={visitor_id}  "
                  f"sim={score:.3f}  entries={match.entry_count}")
            return visitor_id, True
        else:
            # Fresh visitor
            visitor_id = self._new_visitor_id()
            self._active[track_id] = visitor_id
            print(f"[ReID] NEW      track={track_id}  visitor={visitor_id}")
            return visitor_id, False

    def on_exit(self, track_id: int) -> None:
        """
        Call when a track emits an EXIT event.
        Moves embedding to gallery for future re-entry matching.
        """
        if track_id not in self._active:
            return

        visitor_id = self._active.pop(track_id)
        emb = self._track_embeddings.pop(track_id, None)

        if emb is not None:
            self._gallery[visitor_id] = GalleryEntry(visitor_id, emb)
            print(f"[ReID] EXIT     track={track_id}  visitor={visitor_id}  "
                  f"gallery size={len(self._gallery)}")

        self._frame_counter.pop(track_id, None)

    def get_visitor_id(self, track_id: int) -> str | None:
        """Return cached visitor_id for an active track, or None."""
        return self._active.get(track_id)

    def purge_stale_gallery(self) -> None:
        """
        Remove gallery entries older than RE_ENTRY_WINDOW_SEC.
        Call once per minute in the main loop.
        """
        now = time.time()
        stale = [
            vid for vid, entry in self._gallery.items()
            if now - entry.exit_time > RE_ENTRY_WINDOW_SEC
        ]
        for vid in stale:
            del self._gallery[vid]
        if stale:
            print(f"[ReID] Purged {len(stale)} stale gallery entries.")

    def stats(self) -> dict:
        return {
            "active_tracks"  : len(self._active),
            "gallery_size"   : len(self._gallery),
            "unique_visitors": len(set(self._active.values()))
                               + len(self._gallery),
        }

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _new_visitor_id() -> str:
        return "VIS_" + uuid.uuid4().hex[:8].upper()

    def _embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Extract a 576-d L2-normalised embedding from a BGR crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
            return None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor   = _TRANSFORM(crop_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = self.extractor(tensor)          # (1, 576)

        return emb.squeeze(0).cpu().numpy()       # (576,)

    def _search_gallery(
        self,
        query_emb: np.ndarray,
    ) -> tuple[GalleryEntry | None, float]:
        """
        Find best cosine-similarity match in gallery within time window.
        Returns (best_entry, score) or (None, 0.0).
        """
        if not self._gallery:
            return None, 0.0

        now = time.time()
        best_entry = None
        best_score = 0.0

        for entry in self._gallery.values():
            # Skip entries outside the re-entry window
            if now - entry.exit_time > RE_ENTRY_WINDOW_SEC:
                continue

            score = float(np.dot(query_emb, entry.embedding))   # both L2-normed

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= REID_THRESHOLD:
            return best_entry, best_score

        return None, best_score

    def _maybe_refresh_embedding(
        self,
        crop_bgr: np.ndarray,
        track_id: int,
    ) -> None:
        """Refresh stored embedding every N frames (running average)."""
        self._frame_counter[track_id] += 1
        if self._frame_counter[track_id] % self._embed_every != 0:
            return

        new_emb = self._embed(crop_bgr)
        if new_emb is None:
            return

        if track_id in self._track_embeddings:
            # Exponential moving average — keeps embedding stable
            old = self._track_embeddings[track_id]
            merged = 0.7 * old + 0.3 * new_emb
            self._track_embeddings[track_id] = (
                merged / np.linalg.norm(merged)   # re-normalise
            )
        else:
            self._track_embeddings[track_id] = new_emb


# ── Quick standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    VIDEO_PATH = "videos/Entry_CAM.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"Cannot open {VIDEO_PATH}")

    from ultralytics import YOLO
    from EntryExitLine import check_line_crossing, side_of_line, ENTRY_LINE_P1, ENTRY_LINE_P2

    model = YOLO("yolov8n.pt")
    reid  = ReIDManager()

    prev_centroids = {}
    frame_idx      = 0
    MAX_FRAMES     = 450

    # print(f"\nRunning Re-ID test on first {MAX_FRAMES} frames ...\n")
    print(f"\nRunning Re-ID test on all frames ...\n")

    # while frame_idx < MAX_FRAMES:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, classes=[0], conf=0.35,
            tracker="bytetrack.yaml", persist=True, verbose=False,
        )[0]

        if results.boxes.id is not None:
            boxes     = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx = (x1 + x2) // 2
                cy = int(y2)

                crop = frame[max(0,y1):y2, max(0,x1):x2]
                visitor_id, is_reentry = reid.identify(crop, tid)

                if tid in prev_centroids:
                    event = check_line_crossing(visitor_id, prev_centroids[tid], (cx, cy))
                    if event:
                        tag = "REENTRY" if is_reentry else event.direction
                        print(f"  Frame {frame_idx:04d} | {tag:8s} | "
                              f"track={tid} visitor={visitor_id}")
                        if event.direction == "EXIT":
                            reid.on_exit(tid)

                prev_centroids[tid] = (cx, cy)

        # Purge stale gallery every 300 frames (~20s)
        if frame_idx % 300 == 0:
            reid.purge_stale_gallery()

        frame_idx += 1

    cap.release()
    print(f"\nRe-ID stats: {reid.stats()}")
    print("If you see REENTRY events above, Re-ID is working correctly.")