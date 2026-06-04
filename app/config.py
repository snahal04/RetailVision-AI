import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{PROJECT_ROOT / 'data' / 'store_intelligence.db'}",
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
STALE_FEED_MINUTES = int(os.getenv("STALE_FEED_MINUTES", "10"))
HEATMAP_MIN_SESSIONS = int(os.getenv("HEATMAP_MIN_SESSIONS", "20"))
