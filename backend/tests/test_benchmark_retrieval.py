"""The offline benchmark scores retrieval exactly like the eval harness."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.eval.retrieval import ranked_retrieval_metrics
from app.rag.retrieve import RetrievalResult
from app.schemas import Chunk

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load("benchmark_retrieval")


def _chunk(cid: str, filename: str) -> Chunk:
    return Chunk(id=cid, doc_id=filename, text="t", tokens=1, metadata={"filename": filename})


async def test_k_counts_documents_and_matches_the_harness():
    # Two chunks of a.md come first: at K=2 the documents are a.md and b.md,
    # not a.md twice, which is what the harness would score too.
    chunks = [_chunk("a:0", "a.md"), _chunk("a:1", "a.md"), _chunk("b:0", "b.md"), _chunk("c:0", "c.md")]
    example = {"question": "q", "expected_sources": ["b.md", "c.md"]}
    with patch.object(
        benchmark, "hybrid_search", new=AsyncMock(return_value=RetrievalResult(chunks=chunks))
    ):
        report = await benchmark.run_mode("sparse", [example], [1, 2, 3], rerank=False)

    expected = ranked_retrieval_metrics(["b.md", "c.md"], ["a.md", "a.md", "b.md", "c.md"], [1, 2, 3])
    assert expected is not None
    for k in (1, 2, 3):
        got = report["at_k"][str(k)]
        assert got["recall"] == round(expected[f"recall@{k}"], 4)
        assert got["precision"] == round(expected[f"precision@{k}"], 4)
        assert got["ndcg"] == round(expected[f"ndcg@{k}"], 4)
        assert got["mrr"] == round(expected["mrr"], 4)
    assert report["at_k"]["2"]["recall"] == 0.5  # a.md, b.md -> one of two expected
