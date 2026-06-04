from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_trace_id
from app.exceptions import DatabaseUnavailableError
from app.schemas import ErrorResponse, IngestRequest, IngestResponse
from app.services.ingest import ingest_events

router = APIRouter(prefix="/events", tags=["events"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={503: {"model": ErrorResponse}},
)
def post_ingest(
    body: IngestRequest,
    request: Request,
    db: Session = Depends(get_db),
    trace_id: str = Depends(get_trace_id),
) -> IngestResponse:
    request.state.event_count = len(body.events)
    try:
        return ingest_events(db, body.events)
    except DatabaseUnavailableError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="database_unavailable",
                detail="Database is temporarily unavailable. Retry later.",
                trace_id=trace_id,
            ).model_dump(),
        ) from None
