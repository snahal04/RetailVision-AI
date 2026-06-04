#!/usr/bin/env python3
"""Ingest pipeline JSONL output into the Intelligence API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

BATCH = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="output/events.jsonl")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--simulate", action="store_true", help="delay between batches")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python -m pipeline.run")

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    url = f"{args.api.rstrip('/')}/events/ingest"

    with httpx.Client(timeout=60.0) as client:
        for i in range(0, len(events), BATCH):
            batch = events[i : i + BATCH]
            r = client.post(url, json={"events": batch})
            r.raise_for_status()
            body = r.json()
            print(f"Batch {i // BATCH + 1}: accepted={body['accepted']} dup={body['duplicate']} rejected={body['rejected']}")
            if args.simulate:
                time.sleep(1)

    print(f"Fed {len(events)} events to {url}")


if __name__ == "__main__":
    main()
