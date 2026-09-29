import { describe, expect, it } from "vitest";
import type { RequestMetrics, StrategyComparison } from "../api/eval";
import { bestValues, chunkProvenance, isFailedRow, latencyStages, whyThisResult, type Fact } from "./compareInsights";

function row(over: Partial<StrategyComparison> = {}): StrategyComparison {
  return {
    strategy: "classic",
    question: "q",
    answer: "An answer [S1].",
    sources: [{ chunk_id: "a:0", quote: "x", handle: "S1" }],
    refusal: false,
    confidence: 0.8,
    latency_ms: 1200,
    input_tokens: 100,
    output_tokens: 20,
    iterations: 1,
    trace: [],
    extra: {},
    status: "answered",
    grounded: true,
    citation_count: 1,
    invalid_citations: [],
    ...over,
  };
}

const metrics = (over: Partial<RequestMetrics> = {}): RequestMetrics => ({
  strategy: "classic",
  model: "m",
  provider: "anthropic",
  input_tokens: 100,
  output_tokens: 20,
  estimated_cost_usd: 0.002,
  latency_ms: { total: 1000, retrieval: 300, rerank: 0, generation: 600, citation_validation: 50, other: 50 },
  retrieved_chunks: 20,
  context_chunks: 5,
  llm_calls: 1,
  ...over,
});

const keys = (facts: Fact[]) => facts.map((f) => ("plural" in f ? f.plural : f.key));

const diag = {
  mode: "hybrid" as const,
  reranker: "cross-encoder" as const,
  counts: { dense: 20, sparse: 20, fused: 30, reranked: 30, final: 4 },
  latency_ms: { preprocess: 1, dense: 10, sparse: 5, fusion: 1, rerank: 40 },
  degraded: [],
  chunks: [
    { chunk_id: "a:0", doc_id: "a", dense_rank: 1, sparse_rank: 2, fused_score: 0.03, rerank_score: 5 },
    { chunk_id: "b:0", doc_id: "b", dense_rank: 3, sparse_rank: null, fused_score: 0.02, rerank_score: 3 },
    { chunk_id: "c:0", doc_id: "c", dense_rank: null, sparse_rank: 1, fused_score: 0.02, rerank_score: 2 },
    { chunk_id: "d:0", doc_id: "d", dense_rank: 5, sparse_rank: null, fused_score: 0.01, rerank_score: 1 },
  ],
};

describe("whyThisResult", () => {
  it("describes hybrid retrieval provenance from the measured ranks", () => {
    const r = row({ retrieval: diag });
    expect(chunkProvenance(r)).toEqual({ total: 4, both: 1, denseOnly: 2, sparseOnly: 1 });
    const facts = whyThisResult(r);
    expect(facts[0]).toEqual({ key: "why.retrieval", vars: { mode: "hybrid", reranker: "cross-encoder" } });
    expect(facts).toContainEqual({
      plural: "why.provenance",
      count: 4,
      vars: { both: 1, dense: 2, sparse: 1 },
    });
    // Context size falls back to the diagnostics' final count.
    expect(facts).toContainEqual({ plural: "why.citations", count: 4, vars: { cited: 1 } });
    expect(keys(facts)).toContain("why.status.grounded");
  });

  it("uses a single-source sentence outside hybrid mode", () => {
    const facts = whyThisResult(row({ retrieval: { ...diag, mode: "dense" } }));
    expect(facts).toContainEqual({ plural: "why.provenanceSingle", count: 4, vars: { mode: "dense" } });
  });

  it("prefers the listed context sources for context size, and reports invalid citations", () => {
    const facts = whyThisResult(
      row({ extra: { context_sources: [{}, {}, {}] }, invalid_citations: ["S9", "S7"], citation_count: 2 }),
    );
    expect(facts).toContainEqual({ plural: "why.citations", count: 3, vars: { cited: 2 } });
    expect(facts).toContainEqual({ plural: "why.invalid", count: 2, vars: { handles: "S9, S7" } });
  });

  it("reports grounding status and the model's own notes", () => {
    const facts = whyThisResult(
      row({ status: "insufficient_context", grounded: false, refusal: true, unsupported_notes: "salary data" }),
    );
    expect(keys(facts)).toContain("why.status.insufficient");
    expect(facts).toContainEqual({ key: "why.notCovered", vars: { notes: "salary data" } });
    expect(keys(whyThisResult(row({ status: "answered", grounded: false })))).toContain("why.status.weak");
    expect(keys(whyThisResult(row({ status: "partial" })))).toContain("why.status.partial");
  });

  it("counts agentic searches and iterations", () => {
    const facts = whyThisResult(
      row({ strategy: "agentic", iterations: 4, extra: { retrieval_searches: [{}, {}, {}], stop_reason: "submitted" } }),
    );
    expect(facts).toContainEqual({ key: "why.agent", vars: { searches: 3, iterations: 4 } });
    expect(facts).toContainEqual({ key: "why.stopReason", vars: { reason: "submitted" } });
  });

  it("lists graph entities, or says none were extracted", () => {
    const facts = whyThisResult(
      row({ strategy: "graph", extra: { entities: ["BM25", "RRF"], related_entities: ["x", "y", "z"] } }),
    );
    expect(facts).toContainEqual({ plural: "why.entities", count: 2, vars: { list: "BM25, RRF", related: 3 } });
    expect(keys(whyThisResult(row({ strategy: "graph", extra: { entities: [] } })))).toContain("why.noEntities");
  });

  it("mentions model calls only when per-request metrics exist", () => {
    expect(keys(whyThisResult(row()))).not.toContain("why.llmCalls");
    expect(whyThisResult(row({ metrics: metrics({ llm_calls: 3 }) }))).toContainEqual({ plural: "why.llmCalls", count: 3 });
  });

  it("says nothing but 'failed' for a failed strategy", () => {
    const failed = row({
      answer: "This strategy failed.",
      sources: [{ chunk_id: "none", quote: "" }],
      latency_ms: 0,
      input_tokens: 0,
      output_tokens: 0,
      status: null,
      refusal: true,
    });
    expect(isFailedRow(failed)).toBe(true);
    expect(whyThisResult(failed)).toEqual([{ key: "why.failed" }]);
  });
});

describe("bestValues", () => {
  const ev = (faith: number | null, recall?: number) => ({
    status: "scored" as const,
    scores: { faithfulness: faith, answer_relevance: 0.9, answer_correctness: null },
    reasoning: {},
    retrieval: recall == null ? null : { "recall@5": recall, mrr: 1 },
    retrieved_docs: [],
    relevant_docs: null,
    has_reference_answer: false,
    correct_refusal: null,
    judge_model: null,
    rubric_version: null,
    judge_attempts: 1,
    judge_input_tokens: 0,
    judge_output_tokens: 0,
    error_type: null,
    error: null,
  });

  it("marks the highest score and the lowest cost, ties included", () => {
    const rows = [
      row({ strategy: "classic", latency_ms: 900, evaluation: ev(0.9, 1) }),
      row({ strategy: "graph", latency_ms: 1500, evaluation: ev(0.9, 0.5) }),
      row({ strategy: "agentic", latency_ms: 4000, evaluation: ev(0.7, 0.5) }),
    ];
    const best = bestValues(rows);
    expect([...best.faithfulness].sort()).toEqual(["classic", "graph"]);
    expect([...best["recall@5"]]).toEqual(["classic"]);
    expect([...best.latency]).toEqual(["classic"]);
  });

  it("highlights nothing when all measured values are equal or fewer than two are measured", () => {
    const rows = [
      row({ strategy: "classic", evaluation: ev(0.8) }),
      row({ strategy: "graph", evaluation: ev(0.8) }),
      row({ strategy: "agentic", evaluation: ev(null) }),
    ];
    const best = bestValues(rows);
    expect(best.faithfulness.size).toBe(0);
    expect(best.answer_relevance.size).toBe(0);
    // answer_correctness and cost are never measured here.
    expect(best.answer_correctness.size).toBe(0);
    expect(best.cost.size).toBe(0);
  });

  it("ignores failed rows (their zero latency is not a measurement)", () => {
    const rows = [
      row({ strategy: "classic", latency_ms: 800 }),
      row({ strategy: "graph", latency_ms: 2000 }),
      row({ strategy: "agentic", latency_ms: 0, input_tokens: 0, output_tokens: 0, status: null, sources: [{ chunk_id: "none", quote: "" }] }),
    ];
    expect([...bestValues(rows).latency]).toEqual(["classic"]);
  });

  it("prefers per-request metrics for latency, tokens and cost", () => {
    const rows = [
      row({ strategy: "classic", latency_ms: 100, metrics: metrics({ latency_ms: { ...metrics().latency_ms, total: 5000 }, estimated_cost_usd: 0.01 }) }),
      row({ strategy: "graph", latency_ms: 900, metrics: metrics({ estimated_cost_usd: null }) }),
      row({ strategy: "agentic", latency_ms: 900, metrics: metrics({ estimated_cost_usd: 0.03 }) }),
    ];
    const best = bestValues(rows);
    expect([...best.latency].sort()).toEqual(["agentic", "graph"]);
    expect([...best.cost]).toEqual(["classic"]);
  });
});

describe("latencyStages", () => {
  it("returns non-zero stages in pipeline order, or null without metrics", () => {
    expect(latencyStages(row())).toBeNull();
    expect(latencyStages(row({ metrics: metrics() }))).toEqual([
      { stage: "retrieval", ms: 300 },
      { stage: "generation", ms: 600 },
      { stage: "citation_validation", ms: 50 },
      { stage: "other", ms: 50 },
    ]);
  });
});
