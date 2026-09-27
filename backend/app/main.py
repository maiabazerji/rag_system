import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api import (
    admin,
    ask,
    compare,
    eval_routes,
    graph,
    ingest,
    metrics,
    privacy,
    traces,
)
from app.api import advisor as advisor_api
from app.auth import init_db as init_auth_db
from app.config import settings
from app.i18n import AcceptLanguageMiddleware
from app.logging_config import get_structured_logger, setup_logging
from app.middleware.request_id import RequestIDMiddleware, get_request_id
from app.monitoring import MetricsMiddleware
from app.privacy.pii import redact_for_telemetry
from app.privacy.retention import start_retention_task
from app.rag.generate import public_provider_error
from app.rag.providers import MissingKeyError, ProviderError
from app.rag.providers.anthropic_provider import close_client as close_anthropic_client
from app.rag.store import StoreUnavailable, store_unavailable_handler
from app.rag.store import close as close_store
from app.tracing import langfuse_exporter, policy

setup_logging()
logger = get_structured_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration and prepare the auth database on startup.

    Configuration is validated here rather than at import time so that tests,
    linters and tooling can import the app without a configured environment.
    """
    settings.validate_startup()

    telemetry = policy.evaluate(settings)
    policy.log_decision(telemetry)
    policy.apply_wandb_env(telemetry, settings)
    langfuse_exporter.set_redactor(redact_for_telemetry)
    langfuse_exporter.configure(settings)

    if settings.require_api_key or settings.admin_key:
        try:
            init_auth_db()
        except Exception as e:
            logger.error(
                f"Auth database initialization failed: {type(e).__name__}: {e}",
                extra_fields={"error_type": type(e).__name__},
            )
            if settings.require_api_key:
                raise RuntimeError(
                    "REQUIRE_API_KEY is on but the auth database is unreachable. "
                    "Check POSTGRES_URL and that Postgres is running."
                ) from e
    else:
        logger.info("Auth disabled; skipping auth database setup.")

    # Daily purge of records past their retention period (docs/gdpr/README.md).
    retention_task = start_retention_task()
    try:
        yield
    finally:
        retention_task.cancel()
        with suppress(asyncio.CancelledError):
            await retention_task
        await close_anthropic_client()
        await close_store()
        langfuse_exporter.shutdown()


app = FastAPI(title="EvalRAG", version="0.3.0", lifespan=lifespan)
app.add_exception_handler(StoreUnavailable, store_unavailable_handler)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(AcceptLanguageMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])
app.include_router(compare.router, prefix="/compare", tags=["compare"])
app.include_router(graph.router, prefix="/graph", tags=["graph"])
app.include_router(eval_routes.router, prefix="/eval", tags=["eval"])
app.include_router(traces.router, prefix="/traces", tags=["traces"])
app.include_router(advisor_api.router, prefix="/advise", tags=["advisor"])
app.include_router(privacy.router, prefix="/privacy", tags=["privacy"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a differentiated error response for uncaught exceptions.

    - 400: validation errors (bad input)
    - 503: provider errors (API unavailable, missing keys)
    - 500: everything else (server fault)
    """
    request_id = get_request_id()

    if isinstance(exc, ValidationError):
        logger.warning(
            f"Validation error on {request.url.path}: {exc}",
            extra_fields={"error_type": "validation_error", "path": request.url.path},
        )
        return JSONResponse(
            status_code=400,
            content={
                "code": "validation_error",
                "message": "Invalid request parameters.",
                "detail": exc.error_count(),
                "request_id": request_id,
            },
        )

    if isinstance(exc, (MissingKeyError, ProviderError)):
        logger.warning(
            f"Provider error on {request.url.path}: {type(exc).__name__}: {exc}",
            extra_fields={
                "error_type": type(exc).__name__,
                "path": request.url.path,
                "detail": str(exc),
            },
        )
        return JSONResponse(
            status_code=503,
            content={
                "code": "provider_error",
                "message": "External service unavailable.",
                # Never the raw exception text: it can carry upstream bodies.
                "detail": public_provider_error(exc),
                "request_id": request_id,
            },
        )

    logger.exception(
        f"Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}",
        extra_fields={
            "error_type": type(exc).__name__,
            "path": request.url.path,
            "detail": str(exc),
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "An unexpected error occurred. Please try again.",
            "detail": type(exc).__name__,
            "request_id": request_id,
        },
    )


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Report liveness, which providers are configured, and whether auth is on.

    ``langfuse`` and ``wandb`` are true only when the telemetry policy actually
    lets them export, not merely when keys are present.
    """
    telemetry = policy.evaluate(settings)
    return {
        "status": "ok",
        "version": app.version,
        "auth_required": settings.require_api_key,
        # Lets a client offer SSO sign-in; null when only API keys are accepted.
        "oidc_issuer": (settings.oidc_issuer or None) if settings.require_api_key else None,
        "providers": {
            "anthropic": bool(settings.anthropic_api_key),
            "openai": bool(settings.openai_api_key),
            "langfuse": telemetry.langfuse.enabled,
            "wandb": telemetry.wandb.enabled,
        },
        "telemetry": {
            "mode": telemetry.mode,
            "wandb_mode": telemetry.wandb_mode,
            "metrics": settings.metrics_enabled,
        },
    }
