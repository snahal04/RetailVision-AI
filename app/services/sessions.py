"""Reconstruct customer visit sessions from raw events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.models import EventRecord

SESSION_START = frozenset({"ENTRY", "REENTRY"})
SESSION_END = frozenset({"EXIT"})


@dataclass
class VisitSession:
    session_key: str
    store_id: str
    visitor_id: str
    started_at: datetime
    ended_at: datetime | None = None
    is_reentry: bool = False
    had_zone_visit: bool = False
    had_billing_queue: bool = False
    had_queue_abandon: bool = False
    had_purchase: bool = False
    zones_visited: set[str] = field(default_factory=set)
    zone_dwell_ms: dict[str, int] = field(default_factory=dict)
    queue_depths: list[int] = field(default_factory=list)


def _parse_ts(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return ts


def build_sessions(events: list[EventRecord], customer_only: bool = True) -> list[VisitSession]:
    """Build sessions from ordered events. REENTRY starts a new session without double-counting visitors in funnel."""
    filtered = [
        e for e in sorted(events, key=lambda x: x.timestamp)
        if not customer_only or not e.is_staff
    ]

    sessions: list[VisitSession] = []
    open_by_visitor: dict[str, VisitSession] = {}
    seq = 0

    for ev in filtered:
        ts = _parse_ts(ev.timestamp)
        vid = ev.visitor_id

        if ev.event_type in SESSION_START:
            if vid in open_by_visitor:
                prev = open_by_visitor.pop(vid)
                prev.ended_at = ts
                sessions.append(prev)
            seq += 1
            sess = VisitSession(
                session_key=f"{vid}_{seq}",
                store_id=ev.store_id,
                visitor_id=vid,
                started_at=ts,
                is_reentry=ev.event_type == "REENTRY",
            )
            open_by_visitor[vid] = sess
            _apply_event(sess, ev)
            continue

        sess = open_by_visitor.get(vid)
        if sess is None:
            seq += 1
            sess = VisitSession(
                session_key=f"{vid}_{seq}",
                store_id=ev.store_id,
                visitor_id=vid,
                started_at=ts,
            )
            open_by_visitor[vid] = sess

        _apply_event(sess, ev)

        if ev.event_type in SESSION_END:
            sess.ended_at = ts
            sessions.append(open_by_visitor.pop(vid))

    for sess in open_by_visitor.values():
        sessions.append(sess)

    return sessions


def _apply_event(sess: VisitSession, ev: EventRecord) -> None:
    et = ev.event_type
    if et in ("ZONE_ENTER", "ZONE_DWELL") and ev.zone_id:
        sess.had_zone_visit = True
        sess.zones_visited.add(ev.zone_id)
        if et == "ZONE_DWELL" and ev.dwell_ms:
            sess.zone_dwell_ms[ev.zone_id] = sess.zone_dwell_ms.get(ev.zone_id, 0) + ev.dwell_ms
    if et == "ZONE_EXIT" and ev.zone_id and ev.dwell_ms:
        sess.zone_dwell_ms[ev.zone_id] = sess.zone_dwell_ms.get(ev.zone_id, 0) + ev.dwell_ms
    if et == "BILLING_QUEUE_JOIN":
        sess.had_billing_queue = True
        if ev.queue_depth is not None:
            sess.queue_depths.append(ev.queue_depth)
    if et == "BILLING_QUEUE_ABANDON":
        sess.had_queue_abandon = True
    if et == "ZONE_ENTER" and ev.zone_id == "POS":
        sess.had_purchase = True
    if et == "BILLING_QUEUE_JOIN" and not sess.had_queue_abandon:
        pass
    if et == "EXIT" and sess.had_billing_queue and not sess.had_queue_abandon:
        sess.had_purchase = True
