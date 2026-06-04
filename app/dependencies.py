from collections.abc import Generator

from fastapi import Request
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.exceptions import DatabaseUnavailableError


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except OperationalError as e:
        raise DatabaseUnavailableError(str(e)) from e
    finally:
        db.close()


def get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "unknown")
