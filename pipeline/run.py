"""
Run the Part A detection pipeline.

  python -m pipeline.run
  python -m pipeline.run --camera CAM_ENTRY_01
  python -m pipeline.run --max-frames 450
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import PROJECT_ROOT
from pipeline.detection_pipeline import DetectionPipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Store intelligence detection pipeline (Part A)")
    parser.add_argument(
        "--layout",
        type=Path,
        default=PROJECT_ROOT / "store_layout.json",
    )
    parser.add_argument("--camera", type=str, default=None, help="camera_id or video filename filter")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pipeline = DetectionPipeline(
        layout_path=args.layout,
        model_name=args.model,
        max_frames=args.max_frames,
    )
    pipeline.run_all(video_filter=args.camera)
    if not pipeline.events:
        print("No events emitted. Check that videos exist under videos/ and paths in store_layout.json.")
        sys.exit(1)

    out = pipeline.save_events(args.output)
    print(f"Done. {len(pipeline.events)} events -> {out}")


if __name__ == "__main__":
    main()
