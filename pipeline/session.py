"""Per-visitor session sequencing for event metadata."""

from __future__ import annotations

from collections import defaultdict


class SessionRegistry:
    def __init__(self) -> None:
        self._seq: dict[str, int] = defaultdict(int)

    def next_seq(self, visitor_id: str) -> int:
        self._seq[visitor_id] += 1
        return self._seq[visitor_id]
