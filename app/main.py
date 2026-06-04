import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import LOG_LEVEL, PROJECT_ROOT
from app.db import init_db
from app.exceptions import DatabaseUnavailableError
from app.middleware import RequestLoggingMiddleware
from app.routers import events, health, stores
from app.schemas import ErrorResponse

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(message)s",
    stream=sys.stdout,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Path(PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="Store Intelligence API",
    version="1.0.0",
    description="Part B — ingest pipeline events and expose store analytics.",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(events.router)
app.include_router(stores.router)
app.include_router(health.router)


@app.exception_handler(DatabaseUnavailableError)
async def db_unavailable_handler(request: Request, exc: DatabaseUnavailableError):
    trace_id = getattr(request.state, "trace_id", None)
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="database_unavailable",
            detail=str(exc) or "Database unavailable",
            trace_id=trace_id,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, _exc: Exception):
    trace_id = getattr(request.state, "trace_id", None)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred.",
            trace_id=trace_id,
        ).model_dump(),
    )
