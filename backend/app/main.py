import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import ask, compare, eval_routes, graph, ingest, traces
from app.config import settings
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="EvalRAG", version="0.2.0")

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
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred. Please try again.",
            "detail": str(type(exc).__name__),
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
