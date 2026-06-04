import json
import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("store_intelligence.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
        request.state.trace_id = trace_id
        request.state.event_count = None
        store_id = request.path_params.get("id") if request.path_params else None
        start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Trace-Id"] = trace_id
            return response
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log_record = {
                "trace_id": trace_id,
                "store_id": store_id,
                "endpoint": request.url.path,
                "method": request.method,
                "latency_ms": latency_ms,
                "event_count": getattr(request.state, "event_count", None),
                "status_code": status_code,
            }
            logger.info(json.dumps(log_record))
