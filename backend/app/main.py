from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api import admin, ask, compare, eval_routes, graph, ingest, traces
from app.auth import init_db as init_auth_db
from app.config import settings
from app.logging_config import get_structured_logger, setup_logging
from app.middleware.request_id import RequestIDMiddleware, get_request_id
from app.rag.providers import MissingKeyError, ProviderError
from app.rag.providers.anthropic_provider import close_client as close_anthropic_client

setup_logging()
logger = get_structured_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration and prepare the auth database on startup.

    Configuration is validated here rather than at import time so that tests,
    linters and tooling can import the app without a configured environment.
    """
    settings.validate_startup()

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

    try:
        yield
    finally:
        await close_anthropic_client()


app = FastAPI(title="EvalRAG", version="0.3.0", lifespan=lifespan)

app.add_middleware(RequestIDMiddleware)
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
                "detail": str(exc),
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
    """Report liveness, which providers are configured, and whether auth is on."""
    return {
        "status": "ok",
        "version": app.version,
        "auth_required": settings.require_api_key,
        "providers": {
            "anthropic": bool(settings.anthropic_api_key),
            "openai": bool(settings.openai_api_key),
            "langfuse": bool(settings.langfuse_public_key and settings.langfuse_secret_key),
            "wandb": bool(settings.wandb_api_key),
        },
    }
