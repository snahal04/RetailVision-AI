from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_trace_id
from app.exceptions import DatabaseUnavailableError
from app.schemas import (
    AnomaliesResponse,
    ErrorResponse,
    FunnelResponse,
    HeatmapResponse,
    MetricsResponse,
)
from app.services.analytics import get_anomalies, get_funnel, get_heatmap, get_metrics

router = APIRouter(prefix="/stores", tags=["stores"])


def _raise_db_error(trace_id: str) -> None:
    raise HTTPException(
        status_code=503,
        detail=ErrorResponse(
            error="database_unavailable",
            detail="Database is temporarily unavailable.",
            trace_id=trace_id,
        ).model_dump(),
    )


@router.get("/{store_id}/metrics", response_model=MetricsResponse)
def store_metrics(
    store_id: str,
    db: Session = Depends(get_db),
    trace_id: str = Depends(get_trace_id),
):
    try:
        return get_metrics(db, store_id)
    except DatabaseUnavailableError:
        _raise_db_error(trace_id)


@router.get("/{store_id}/funnel", response_model=FunnelResponse)
def store_funnel(
    store_id: str,
    db: Session = Depends(get_db),
    trace_id: str = Depends(get_trace_id),
):
    try:
        return get_funnel(db, store_id)
    except DatabaseUnavailableError:
        _raise_db_error(trace_id)


@router.get("/{store_id}/heatmap", response_model=HeatmapResponse)
def store_heatmap(
    store_id: str,
    db: Session = Depends(get_db),
    trace_id: str = Depends(get_trace_id),
):
    try:
        return get_heatmap(db, store_id)
    except DatabaseUnavailableError:
        _raise_db_error(trace_id)


@router.get("/{store_id}/anomalies", response_model=AnomaliesResponse)
def store_anomalies(
    store_id: str,
    db: Session = Depends(get_db),
    trace_id: str = Depends(get_trace_id),
):
    try:
        return get_anomalies(db, store_id)
    except DatabaseUnavailableError:
        _raise_db_error(trace_id)
