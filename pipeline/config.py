from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAYOUT_PATH = PROJECT_ROOT / "store_layout.json"
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_MODEL = "yolov8n.pt"
CONF_THRESH = 0.35
DWELL_INTERVAL_MS = 30_000
PERSON_CLASS = 0
