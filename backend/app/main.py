import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api import ask, compare, eval_routes, graph, ingest, traces
from app.config import settings
from app.logging_config import setup_logging, get_structured_logger
from app.middleware.request_id import RequestIDMiddleware, get_request_id
from app.rag.providers import MissingKeyError, ProviderError

setup_logging()
logger = get_structured_logger(__name__)

app = FastAPI(title="EvalRAG", version="0.2.0")

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])
app.include_router(compare.router, prefix="/compare", tags=["compare"])
app.include_router(graph.router, prefix="/graph", tags=["graph"])
app.include_router(eval_routes.router, prefix="/eval", tags=["eval"])
app.include_router(traces.router, prefix="/traces", tags=["traces"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with differentiated error responses.

    Returns:
    - 400: Validation errors (bad input)
    - 503: Provider errors (API unavailable, missing keys)
    - 500: All other errors (server fault)
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


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "providers": {
            "anthropic": bool(settings.anthropic_api_key),
            "openai": bool(settings.openai_api_key),
            "cohere": bool(settings.cohere_api_key),
            "langfuse": bool(settings.langfuse_public_key and settings.langfuse_secret_key),
            "wandb": bool(settings.wandb_api_key),
        },
    }
