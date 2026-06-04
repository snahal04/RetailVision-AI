"""Re-identification across track loss and re-entry (MobileNetV3 embeddings)."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

logger = logging.getLogger(__name__)

REID_THRESHOLD = 0.82
RE_ENTRY_WINDOW_SEC = 600
MIN_CROP_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.pool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return nn.functional.normalize(x, dim=1)


_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GalleryEntry:
    def __init__(self, visitor_id: str, embedding: np.ndarray) -> None:
        self.visitor_id = visitor_id
        self.embedding = embedding
        self.exit_time = time.time()


class ReIDManager:
    def __init__(self) -> None:
        self.extractor = EmbeddingExtractor().to(DEVICE).eval()
        self._active: dict[int, str] = {}
        self._track_embeddings: dict[int, np.ndarray] = {}
        self._gallery: dict[str, GalleryEntry] = {}
        self._pending_reentry: dict[int, bool] = {}
        self._embed_every = 10
        self._frame_counter: dict[int, int] = defaultdict(int)
        logger.info("ReID ready on %s", DEVICE)

    def identify(self, crop_bgr: np.ndarray, track_id: int) -> tuple[str, bool]:
        if track_id in self._active:
            self._maybe_refresh_embedding(crop_bgr, track_id)
            return self._active[track_id], False

        emb = self._embed(crop_bgr)
        if emb is None:
            vid = self._new_visitor_id()
            self._active[track_id] = vid
            return vid, False

        self._track_embeddings[track_id] = emb
        match, score = self._search_gallery(emb)

        if match is not None:
            visitor_id = match.visitor_id
            del self._gallery[visitor_id]
            self._active[track_id] = visitor_id
            self._pending_reentry[track_id] = True
            logger.debug("REENTRY match track=%s visitor=%s sim=%.3f", track_id, visitor_id, score)
            return visitor_id, True

        visitor_id = self._new_visitor_id()
        self._active[track_id] = visitor_id
        return visitor_id, False

    def consume_reentry_flag(self, track_id: int) -> bool:
        return self._pending_reentry.pop(track_id, False)

    def on_exit(self, track_id: int) -> None:
        if track_id not in self._active:
            return
        visitor_id = self._active.pop(track_id)
        emb = self._track_embeddings.pop(track_id, None)
        if emb is not None:
            self._gallery[visitor_id] = GalleryEntry(visitor_id, emb)
        self._frame_counter.pop(track_id, None)
        self._pending_reentry.pop(track_id, None)

    def purge_stale_gallery(self) -> None:
        now = time.time()
        stale = [
            vid for vid, e in self._gallery.items()
            if now - e.exit_time > RE_ENTRY_WINDOW_SEC
        ]
        for vid in stale:
            del self._gallery[vid]

    @staticmethod
    def _new_visitor_id() -> str:
        return "VIS_" + uuid.uuid4().hex[:8].upper()

    def _embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
            return None
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = _TRANSFORM(crop_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.extractor(tensor)
        return emb.squeeze(0).cpu().numpy()

    def _search_gallery(self, query_emb: np.ndarray) -> tuple[GalleryEntry | None, float]:
        if not self._gallery:
            return None, 0.0
        now = time.time()
        best_entry, best_score = None, 0.0
        for entry in self._gallery.values():
            if now - entry.exit_time > RE_ENTRY_WINDOW_SEC:
                continue
            score = float(np.dot(query_emb, entry.embedding))
            if score > best_score:
                best_score, best_entry = score, entry
        if best_score >= REID_THRESHOLD:
            return best_entry, best_score
        return None, best_score

    def _maybe_refresh_embedding(self, crop_bgr: np.ndarray, track_id: int) -> None:
        self._frame_counter[track_id] += 1
        if self._frame_counter[track_id] % self._embed_every != 0:
            return
        new_emb = self._embed(crop_bgr)
        if new_emb is None:
            return
        if track_id in self._track_embeddings:
            old = self._track_embeddings[track_id]
            merged = 0.7 * old + 0.3 * new_emb
            self._track_embeddings[track_id] = merged / np.linalg.norm(merged)
        else:
            self._track_embeddings[track_id] = new_emb
