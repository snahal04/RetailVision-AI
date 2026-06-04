"""Entry/exit line crossing for entry cameras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EntryLine:
    p1: tuple[int, int]
    p2: tuple[int, int]


@dataclass
class CrossingEvent:
    direction: Literal["ENTRY", "EXIT"]
    visitor_id: str
    prev_point: tuple[int, int]
    curr_point: tuple[int, int]


def _cross_z(p1: tuple[int, int], p2: tuple[int, int], point: tuple[int, int]) -> float:
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def side_of_line(
    point: tuple[int, int],
    line: EntryLine,
) -> Literal["inside", "outside", "on"]:
    z = _cross_z(line.p1, line.p2, point)
    if z > 0:
        return "inside"
    if z < 0:
        return "outside"
    return "on"


def check_line_crossing(
    line: EntryLine,
    visitor_id: str,
    prev_centroid: tuple[int, int],
    curr_centroid: tuple[int, int],
) -> CrossingEvent | None:
    prev_side = side_of_line(prev_centroid, line)
    curr_side = side_of_line(curr_centroid, line)

    if prev_side == curr_side or "on" in (prev_side, curr_side):
        return None

    direction = "ENTRY" if prev_side == "outside" and curr_side == "inside" else "EXIT"
    return CrossingEvent(
        direction=direction,
        visitor_id=visitor_id,
        prev_point=prev_centroid,
        curr_point=curr_centroid,
    )
