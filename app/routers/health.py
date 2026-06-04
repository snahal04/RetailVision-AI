from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_trace_id
from app.exceptions import DatabaseUnavailableError
from app.schemas import ErrorResponse, HealthResponse
from app.services.analytics import get_health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    db: Session = Depends(get_db),
    trace_id: str = Depends(get_trace_id),
):
    try:
        return get_health(db)
    except DatabaseUnavailableError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="database_unavailable",
                detail="Database is temporarily unavailable.",
                trace_id=trace_id,
            ).model_dump(),
        ) from None
