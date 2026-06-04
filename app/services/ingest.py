import json
from datetime import datetime, timezone

from pydantic import ValidationError
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.exceptions import DatabaseUnavailableError
from app.models import EventRecord
from app.schemas import IngestErrorItem, IngestResponse, parse_store_event


def ingest_events(db: Session, raw_events: list[dict]) -> IngestResponse:
    try:
        return _ingest(db, raw_events)
    except OperationalError as e:
        raise DatabaseUnavailableError(str(e)) from e


def _ingest(db: Session, raw_events: list[dict]) -> IngestResponse:
    accepted = duplicate = rejected = 0
    errors: list[IngestErrorItem] = []

    for idx, raw in enumerate(raw_events):
        event_id = raw.get("event_id") if isinstance(raw, dict) else None
        try:
            ev = parse_store_event(raw)
            event_id = ev.event_id
            existing = db.query(EventRecord).filter(EventRecord.event_id == ev.event_id).first()
            if existing:
                duplicate += 1
                continue

            ts = datetime.fromisoformat(ev.timestamp.replace("Z", "+00:00"))
            record = EventRecord(
                event_id=ev.event_id,
                store_id=ev.store_id,
                camera_id=ev.camera_id,
                visitor_id=ev.visitor_id,
                event_type=ev.event_type.value,
                timestamp=ts,
                zone_id=ev.zone_id,
                dwell_ms=ev.dwell_ms,
                is_staff=ev.is_staff,
                confidence=ev.confidence,
                queue_depth=ev.metadata.queue_depth,
                sku_zone=ev.metadata.sku_zone,
                session_seq=ev.metadata.session_seq,
                raw_json=json.dumps(ev.to_json_dict()),
                ingested_at=datetime.now(timezone.utc),
            )
            db.add(record)
            db.flush()
            accepted += 1
        except ValidationError as e:
            rejected += 1
            errors.append(IngestErrorItem(index=idx, event_id=event_id, message=str(e)))
        except SQLAlchemyError as e:
            raise DatabaseUnavailableError(str(e)) from e

    try:
        db.commit()
    except OperationalError as e:
        db.rollback()
        raise DatabaseUnavailableError(str(e)) from e

    return IngestResponse(
        accepted=accepted,
        duplicate=duplicate,
        rejected=rejected,
        errors=errors,
    )
