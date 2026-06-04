"""Validate emitted events against the Part A schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from pipeline.schemas import StoreEvent


def validate_file(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    count = 0

    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        try:
            data = json.loads(line)
            ev = StoreEvent.model_validate(data)
            if ev.event_id in seen_ids:
                errors.append(f"Line {i}: duplicate event_id {ev.event_id}")
            seen_ids.add(ev.event_id)
            if ev.event_type.value in ("ENTRY", "EXIT", "REENTRY") and ev.zone_id is not None:
                errors.append(f"Line {i}: {ev.event_type} should have zone_id=null")
        except (json.JSONDecodeError, ValidationError) as e:
            errors.append(f"Line {i}: {e}")

    return count, errors


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/events.jsonl")
    n, errs = validate_file(p)
    if errs:
        print(f"FAILED — {len(errs)} issues in {n} events")
        for e in errs[:20]:
            print(" ", e)
        sys.exit(1)
    print(f"OK — {n} events validated, all event_ids unique")
