"""Offline retrieval benchmark: dense vs sparse (BM25) vs hybrid (RRF).

Indexes ``data/docs`` into an in-process Qdrant (``QdrantClient(":memory:")``,
no Docker, no server), runs every question of a golden dataset through
:func:`app.rag.retrieve.hybrid_search` in each requested mode, and scores the
retrieved documents against the example's ``expected_sources``.

Documents are ranked by their best-ranked chunk and K counts documents. The
metrics come from :func:`app.eval.retrieval.ranked_retrieval_metrics`, the
function the eval harness uses, so benchmark and eval numbers share one
definition (document level, binary relevance):
    Recall@K     share of expected documents found in the top K documents
    Precision@K  share of the top K documents that were expected
    HitRate@K    share of questions with at least one expected document in the top K
    MRR          mean reciprocal rank of the first expected document
    nDCG@K       normalised discounted cumulative gain at K

Dense and hybrid need the embedding model. When it cannot be loaded (no
network and no local copy) those modes are reported as skipped, with the
reason, and sparse still runs. Numbers are only ever measured, never filled in.

Results go to ``data/benchmarks/retrieval_<dataset>_<timestamp>.{json,md}``.

Usage:
    python scripts/benchmark_retrieval.py
    python scripts/benchmark_retrieval.py --modes sparse --dataset golden_v1
    python scripts/benchmark_retrieval.py --k 1 3 5 10 --rerank
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.http import models as qm  # noqa: E402

from app.config import settings  # noqa: E402
from app.eval.metrics import load_dataset  # noqa: E402
from app.eval.retrieval import ranked_retrieval_metrics  # noqa: E402
from app.rag import parsers, store  # noqa: E402
from app.rag.chunking import chunk_structured  # noqa: E402
from app.rag.retrieve import hybrid_search  # noqa: E402
from app.schemas import Chunk  # noqa: E402

DOCS_DIR = ROOT / "data" / "docs"
OUT_DIR = ROOT / "data" / "benchmarks"
MODES = ("dense", "sparse", "hybrid")
# Chunks retrieved per document cutoff: several chunks often share a document.
CHUNKS_PER_DOC_CUTOFF = 5


def _norm(name: str) -> str:
    return name.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()


def load_corpus(docs_dir: Path) -> list[Chunk]:
    """Parse and chunk every document the way ingestion does (minus PII masking)."""
    chunks: list[Chunk] = []
    for path in sorted(p for p in docs_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(docs_dir).as_posix()
        try:
            parsed = parsers.extract(rel, path.read_bytes())
        except ValueError as e:
            print(f"  skipping {rel}: {e}", file=sys.stderr)
            continue
        doc_id = rel.replace("/", "_")
        for i, c in enumerate(chunk_structured(parsed.text)):
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{i}",
                    doc_id=doc_id,
                    text=c.text,
                    tokens=len(c.text.split()),
                    metadata={"filename": path.name, "heading_path": c.heading_path},
                )
            )
    return chunks


def try_load_embeddings() -> tuple[bool, str]:
    """Whether the embedding model loads here; the reason when it does not."""
    try:
        from app.rag.embed import embedding_dim

        embedding_dim()
        return True, ""
    except Exception as e:  # network blocked, model not cached, ...
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"


async def build_index(chunks: list[Chunk], with_vectors: bool) -> AsyncQdrantClient:
    """Index ``chunks`` into an in-memory Qdrant and point app.rag.store at it."""
    if with_vectors:
        from app.rag.embed import embed_texts, embedding_dim

        dim = embedding_dim()
        vectors = await asyncio.to_thread(embed_texts, [c.text for c in chunks])
    else:
        # Sparse retrieval reads payloads only; a constant 1-d vector satisfies Qdrant.
        dim = 1
        vectors = [[1.0]] * len(chunks)
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        settings.qdrant_collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    # Bypass store.client(): it would check the collection against the model.
    store._client = client
    await store.upsert(chunks, vectors)
    return client


async def run_mode(
    mode: str, examples: list[dict], ks: list[int], rerank: bool
) -> dict[str, Any]:
    """Retrieve for every example in one mode and aggregate the metrics."""
    # Documents are ranked by their best chunk and K counts documents, exactly
    # as in the eval harness (app.eval.retrieval.ranked_retrieval_metrics).
    # Enough chunks are retrieved to fill the deepest cutoff with distinct
    # documents in the common case.
    depth = max(ks) * CHUNKS_PER_DOC_CUTOFF
    names = {"recall": "recall@{k}", "precision": "precision@{k}", "hit_rate": "hit_rate@{k}", "ndcg": "ndcg@{k}"}
    per_k: dict[int, dict[str, list[float]]] = {
        k: {"recall": [], "precision": [], "hit_rate": [], "mrr": [], "ndcg": []} for k in ks
    }
    latencies: list[float] = []
    rerankers: set[str] = set()
    degraded = 0
    for ex in examples:
        started = time.perf_counter()
        result = await hybrid_search(ex["question"], None, mode=mode, final_k=depth, rerank=rerank)
        latencies.append((time.perf_counter() - started) * 1000)
        rerankers.add(result.diagnostics.reranker)
        degraded += bool(result.diagnostics.degraded)
        ranked_docs = [c.metadata.get("filename") or c.doc_id for c in result.chunks]
        scores = ranked_retrieval_metrics(ex["expected_sources"], ranked_docs, ks)
        assert scores is not None  # examples without labels are filtered out earlier
        for k in ks:
            m = per_k[k]
            for metric, template in names.items():
                m[metric].append(scores[template.format(k=k)])
            m["mrr"].append(scores["mrr"])
    return {
        "status": "ok",
        "reranker": sorted(rerankers),
        "degraded_queries": degraded,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2)
            if latencies
            else 0.0,
        },
        "at_k": {
            str(k): {name: round(statistics.fmean(v), 4) for name, v in m.items()}
            for k, m in per_k.items()
        },
    }


def to_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        f"# Retrieval benchmark: {cfg['dataset']}",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Corpus: {cfg['documents']} documents, {cfg['chunks']} chunks "
        f"(CHUNK_SIZE_TOKENS={cfg['chunk_size_tokens']})",
        f"- Questions scored: {cfg['examples_scored']} of {cfg['examples_total']}",
        f"- DENSE_TOP_K={cfg['dense_top_k']}, BM25_TOP_K={cfg['bm25_top_k']}, "
        f"RRF_K={cfg['rrf_k']}, rerank={'on' if cfg['rerank'] else 'off'}",
        f"- Embedding model: {cfg['embedding_model']}",
        "",
        "| Mode | K | Recall@K | Precision@K | HitRate@K | MRR | nDCG@K | mean latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    skipped: list[str] = []
    for mode, res in report["modes"].items():
        if res["status"] != "ok":
            skipped.append(f"- **{mode}**: skipped -- {res['reason']}")
            continue
        for k, m in res["at_k"].items():
            lines.append(
                f"| {mode} | {k} | {m['recall']:.3f} | {m['precision']:.3f} | "
                f"{m['hit_rate']:.3f} | {m['mrr']:.3f} | {m['ndcg']:.3f} | "
                f"{res['latency_ms']['mean']:.1f} |"
            )
    if skipped:
        lines += ["", "Skipped modes:", *skipped]
    if report.get("missing_expected_sources"):
        lines += [
            "",
            "Expected sources absent from the corpus (they can never be retrieved): "
            + ", ".join(report["missing_expected_sources"]),
        ]
    return "\n".join(lines) + "\n"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="golden_v1")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank fused candidates (cross-encoder, BM25 fallback). Off by "
        "default so the modes compare the retrievers themselves.",
    )
    parser.add_argument("--docs", type=Path, default=DOCS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    ks = sorted({k for k in args.k if k > 0})
    if not ks:
        parser.error("--k needs at least one positive value")

    examples_all = load_dataset(args.dataset)
    examples = [ex for ex in examples_all if ex.get("expected_sources")]
    if not examples:
        print(f"No scorable examples (with expected_sources) in '{args.dataset}'.")
        return 1

    print(f"Parsing and chunking {args.docs} ...")
    chunks = load_corpus(args.docs)
    if not chunks:
        print(f"No documents found under {args.docs}.")
        return 1
    filenames = {c.metadata["filename"].lower() for c in chunks}
    missing = sorted(
        {s for ex in examples for s in ex["expected_sources"] if _norm(s) not in filenames}
    )

    modes: dict[str, dict[str, Any]] = {}
    need_vectors = any(m in ("dense", "hybrid") for m in args.modes)
    vectors_ok, why = try_load_embeddings() if need_vectors else (False, "")
    if need_vectors and not vectors_ok:
        print(
            f"\nEmbedding model '{settings.embedding_model}' could not be loaded ({why}).\n"
            "Dense and hybrid modes are SKIPPED; only sparse (BM25) results below are "
            "measured. Run scripts/download_models.py with network access to enable them.\n"
        )
        for m in args.modes:
            if m != "sparse":
                modes[m] = {"status": "skipped", "reason": f"embedding model unavailable: {why}"}

    print(f"Indexing {len(chunks)} chunks into in-memory Qdrant ...")
    client = await build_index(chunks, with_vectors=vectors_ok)
    try:
        for mode in args.modes:
            if mode in modes:
                continue
            print(f"Running {mode} on {len(examples)} questions ...")
            modes[mode] = await run_mode(mode, examples, ks, args.rerank)
    finally:
        await client.close()
        store._client = None

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "dataset": args.dataset,
            "examples_total": len(examples_all),
            "examples_scored": len(examples),
            "documents": len(filenames),
            "chunks": len(chunks),
            "chunk_size_tokens": settings.chunk_size_tokens,
            "dense_top_k": settings.dense_top_k,
            "bm25_top_k": settings.bm25_top_k,
            "rrf_k": settings.rrf_k,
            "k": ks,
            "rerank": args.rerank,
            "embedding_model": settings.embedding_model
            + ("" if vectors_ok else " (not loaded)" if need_vectors else " (not needed)"),
        },
        "missing_expected_sources": missing,
        "modes": {m: modes[m] for m in args.modes},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.out / f"retrieval_{args.dataset}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = to_markdown(report)
    base.with_suffix(".md").write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"Wrote {base.with_suffix('.json')} and {base.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
