from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ask, compare, eval_routes, ingest, traces
from app.config import settings

app = FastAPI(title="EvalRAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])
app.include_router(compare.router, prefix="/compare", tags=["compare"])
app.include_router(eval_routes.router, prefix="/eval", tags=["eval"])
app.include_router(traces.router, prefix="/traces", tags=["traces"])


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
