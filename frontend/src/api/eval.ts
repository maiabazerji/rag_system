/**
 * Typed client for strategy comparison and recorded evaluation runs.
 *
 * Every field the backend added after the first release is optional here, so
 * older payloads (and backends without per-request metrics) still type-check
 * and render.
 */
import { LONG_TIMEOUT_MS, get, post } from "./client";
import type { CitedSource, Grounding } from "../components/Citations";

export type StrategyName = "classic" | "graph" | "agentic";
export const STRATEGY_NAMES: readonly StrategyName[] = ["classic", "graph", "agentic"];

/** Per-request cost and latency breakdown (optional: newer backends only). */
export interface RequestMetrics {
  strategy: string;
  model: string | null;
  provider: string | null;
  input_tokens: number;
  output_tokens: number;
  estimated_cost_usd: number | null;
  latency_ms: {
    total: number;
    retrieval: number;
    rerank: number;
    generation: number;
    citation_validation: number;
    other: number;
  };
  retrieved_chunks: number;
  context_chunks: number;
  llm_calls: number;
}

export interface RetrievedChunkDiagnostics {
  chunk_id: string;
  doc_id: string;
  dense_rank: number | null;
  sparse_rank: number | null;
  fused_score: number | null;
  rerank_score: number | null;
}

export interface RetrievalDiagnostics {
  mode: "dense" | "sparse" | "hybrid";
  reranker: "cross-encoder" | "bm25-fallback" | "none";
  query_truncated?: boolean;
  counts: { dense: number; sparse: number; fused: number; reranked: number; final: number };
  latency_ms: { preprocess: number; dense: number; sparse: number; fusion: number; rerank: number };
  degraded?: string[];
  chunks: RetrievedChunkDiagnostics[];
}

export type JudgeDimension =
  | "faithfulness"
  | "answer_relevance"
  | "context_precision"
  | "context_recall"
  | "answer_correctness";

/** Scores attached to a comparison row when it was requested with evaluate=true. */
export interface StrategyEvaluation {
  status: "scored" | "judge_failed" | "refusal_checked" | "not_evaluated";
  scores: Partial<Record<JudgeDimension, number | null>> | null;
  reasoning: Record<string, string>;
  /** mrr, recall@K, precision@K, hit_rate@K, ndcg@K. Null when no relevant documents are known. */
  retrieval: Record<string, number> | null;
  retrieved_docs: string[];
  relevant_docs: string[] | null;
  has_reference_answer: boolean;
  correct_refusal: number | null;
  judge_model: string | null;
  rubric_version: string | null;
  judge_attempts: number;
  judge_input_tokens: number;
  judge_output_tokens: number;
  error_type: string | null;
  error: string | null;
}

export type TraceStep = { step?: string; [k: string]: unknown };

export type StrategyComparison = Grounding & {
  strategy: StrategyName;
  question: string;
  answer: string;
  sources: CitedSource[];
  refusal: boolean;
  confidence: number;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  iterations: number;
  trace: TraceStep[];
  extra: Record<string, unknown>;
  trace_id?: string | null;
  retrieval?: RetrievalDiagnostics | null;
  metrics?: RequestMetrics | null;
  evaluation?: StrategyEvaluation | null;
};

export interface CompareStrategiesResponse {
  question: string;
  results: StrategyComparison[];
}

export interface CompareStrategiesRequest {
  question: string;
  strategies?: StrategyName[];
  model?: string;
  evaluate?: boolean;
  reference?: { ideal_answer?: string; relevant_doc_ids?: string[] };
  golden?: { dataset: string; id: string };
}

export interface GoldenQuestion {
  id: string;
  question: string;
  question_type: string;
  difficulty: string | null;
}

export interface MetricAggregate {
  mean: number | null;
  std: number | null;
  n: number;
}

export interface RunCost {
  mean_latency_ms?: number;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  total_input_tokens?: number;
  total_output_tokens?: number;
  mean_tokens_per_question?: number | null;
  judge_input_tokens?: number;
  judge_output_tokens?: number;
  /** Present only when the backend priced the run. */
  mean_cost_usd_per_question?: number | null;
  total_cost_usd?: number | null;
}

/** PASS, FAIL, SKIPPED (no baseline or too few examples) or ERROR (thresholds unreadable). */
export type RegressionStatus = "PASS" | "FAIL" | "SKIPPED" | "ERROR";

/** One row of GET /eval/runs. */
export interface EvalRunSummary {
  id: string;
  dataset: string | null;
  strategy: string | null;
  provider?: string | null;
  model?: string | null;
  prompt_version?: string | null;
  judge_model?: string | null;
  n: number;
  n_scored: number;
  n_unscored?: number;
  n_judge_failed?: number | null;
  n_generation_failed?: number | null;
  rubric_version?: string | null;
  retrieval_mode?: string | null;
  config_hash?: string | null;
  regression_status?: RegressionStatus | null;
  aggregates?: Record<string, MetricAggregate> | null;
  /** Older runs: flat judge means, before `aggregates` carried std and n. */
  aggregate?: Record<string, number | null> | null;
  /** Older runs: flat retrieval means (mrr, recall@K, ...). */
  retrieval_aggregate?: Record<string, number | null> | null;
  cost?: RunCost | null;
  created_at?: string | null;
}

export interface RunBreakdown {
  n_examples: number;
  n_scored: number;
  aggregates: Record<string, MetricAggregate>;
}

/** GET /eval/runs/{id}: a summary plus breakdowns and per-example rows. */
export interface EvalRun extends EvalRunSummary {
  schema_version?: number;
  by_question_type?: Record<string, RunBreakdown>;
  by_difficulty?: Record<string, RunBreakdown>;
  regression?: { status?: RegressionStatus; error?: string } | null;
  per_example?: Record<string, unknown>[];
}

/** One metric of a regression report. */
export interface RegressionCheck {
  metric: string;
  status: RegressionStatus | string;
  baseline?: number | null;
  current?: number | null;
  delta?: number | null;
  delta_pct?: number | null;
  /** Human-readable bound, e.g. "delta >= -0.03". */
  threshold?: string | null;
  /** Why the check failed or was skipped. */
  reason?: string | null;
}

export interface RegressionRunRef {
  id: string | null;
  created_at?: string | null;
}

/** GET /eval/runs/{id}/regression. */
export interface RegressionReport {
  status: RegressionStatus;
  checks?: RegressionCheck[];
  baseline?: RegressionRunRef | null;
  baseline_source?: "pinned" | "previous" | "explicit" | null;
  comparable?: boolean;
  config_mismatch?: string[];
  markdown?: string;
}

/** Body of POST /eval/run. */
export interface EvalRunRequest {
  dataset: string;
  strategy?: StrategyName;
}

const enc = encodeURIComponent;

/** POST /compare/strategies. Evaluation runs a judge call per strategy, so it takes longer. */
export function compareStrategies(req: CompareStrategiesRequest): Promise<CompareStrategiesResponse> {
  return post<CompareStrategiesResponse>("/compare/strategies", req);
}

/** GET /eval/golden: dataset names. */
export function listGoldenDatasets(): Promise<string[]> {
  return get<string[]>("/eval/golden");
}

/** GET /eval/golden/{dataset}: questions only, never the reference answers. */
export function listGoldenQuestions(dataset: string): Promise<GoldenQuestion[]> {
  return get<GoldenQuestion[]>(`/eval/golden/${enc(dataset)}`);
}

/**
 * POST /eval/run. Answers and judges every golden question inside one request,
 * so it gets the long timeout.
 */
export function runEvaluation(req: EvalRunRequest): Promise<EvalRun> {
  return post<EvalRun>("/eval/run", req, { timeoutMs: LONG_TIMEOUT_MS });
}

/** GET /eval/runs: every recorded run, newest first. */
export function listEvalRuns(): Promise<EvalRunSummary[]> {
  return get<EvalRunSummary[]>("/eval/runs");
}

/** GET /eval/runs/{id}. */
export function getEvalRun(id: string): Promise<EvalRun> {
  return get<EvalRun>(`/eval/runs/${enc(id)}`);
}

/** GET /eval/runs/{id}/regression. */
export function getRunRegression(id: string): Promise<RegressionReport> {
  return get<RegressionReport>(`/eval/runs/${enc(id)}/regression`);
}
