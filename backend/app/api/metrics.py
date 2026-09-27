"""Prometheus scrape endpoint.

Off unless METRICS_ENABLED=true, and then only for callers on this machine or
holding the admin key (``X-Admin-Key``). Metrics carry no request content, but
route names, model names and traffic volume are still worth not publishing.
"""
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.auth import require_admin_key
from app.config import settings
from app.monitoring import REGISTRY

router = APIRouter()

# Behind a reverse proxy on the same machine every request looks local; run
# uvicorn with --proxy-headers (its default trusts 127.0.0.1) so the real
# client address is what gets checked here.
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def require_metrics_access(
    request: Request, x_admin_key: str | None = Header(default=None)
) -> None:
    """Allow loopback callers, or anyone presenting the admin key.

    Raises:
        HTTPException: 404 while metrics are disabled; otherwise whatever the
            admin-key check raises (403, or 503 when no admin key is set).
    """
    if not settings.metrics_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    host = request.client.host if request.client else ""
    if host in _LOOPBACK:
        return
    require_admin_key(x_admin_key)


@router.get("", include_in_schema=False, dependencies=[Depends(require_metrics_access)])
async def metrics() -> Response:
    """Prometheus text exposition of the backend's metrics."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
