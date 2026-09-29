/**
 * Pure helpers behind the Compare page: the "Why this result?" facts and the
 * per-metric best values. Every fact is derived from a measured field of the
 * comparison row; nothing here judges whether a result is good.
 */
import type { StrategyComparison, StrategyName } from "../api/eval";
import type { MessageKey, PluralKey, Vars } from "../i18n/core";

/** One sentence of the "Why this result?" panel, still to be translated. */
export type Fact =
  | { key: MessageKey; vars?: Vars }
  | { plural: PluralKey; count: number; vars?: Vars };

const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

/** A placeholder row for a strategy that failed to run (no answer to explain). */
export function isFailedRow(r: StrategyComparison): boolean {
  if (r.evaluation?.status === "not_evaluated") return true;
  return (
    r.latency_ms === 0 &&
    r.input_tokens === 0 &&
    r.sources.length <= 1 &&
    (r.sources[0]?.chunk_id ?? "none") === "none" &&
    !r.status
  );
}

/** How the final context chunks were found, from their dense and sparse ranks. */
export function chunkProvenance(r: StrategyComparison): {
  total: number;
  both: number;
  denseOnly: number;
  sparseOnly: number;
} | null {
  const chunks = r.retrieval?.chunks;
  if (!chunks || chunks.length === 0) return null;
  let both = 0;
  let denseOnly = 0;
  let sparseOnly = 0;
  for (const c of chunks) {
    const d = c.dense_rank != null;
    const s = c.sparse_rank != null;
    if (d && s) both++;
    else if (d) denseOnly++;
    else if (s) sparseOnly++;
  }
  return { total: chunks.length, both, denseOnly, sparseOnly };
}

/** Number of chunks the model could cite, or null when the row does not say. */
export function contextSize(r: StrategyComparison): number | null {
  const listed = r.extra?.context_sources;
  if (Array.isArray(listed)) return listed.length;
  if (r.metrics?.context_chunks != null) return r.metrics.context_chunks;
  if (r.retrieval) return r.retrieval.counts.final;
  return null;
}

/** The factual sentences explaining one strategy's result, in display order. */
export function whyThisResult(r: StrategyComparison): Fact[] {
  if (isFailedRow(r)) return [{ key: "why.failed" }];
  const facts: Fact[] = [];

  const diag = r.retrieval;
  if (diag) {
    facts.push({ key: "why.retrieval", vars: { mode: diag.mode, reranker: diag.reranker } });
    if (diag.degraded && diag.degraded.length > 0) {
      facts.push({ key: "why.degraded", vars: { stages: diag.degraded.join(", ") } });
    }
  }

  const prov = chunkProvenance(r);
  if (prov) {
    if (diag?.mode === "hybrid") {
      facts.push({
        plural: "why.provenance",
        count: prov.total,
        vars: { both: prov.both, dense: prov.denseOnly, sparse: prov.sparseOnly },
      });
    } else if (diag) {
      facts.push({ plural: "why.provenanceSingle", count: prov.total, vars: { mode: diag.mode } });
    }
  }

  const ctx = contextSize(r);
  const cited = r.citation_count ?? 0;
  if (ctx != null) {
    facts.push({ plural: "why.citations", count: ctx, vars: { cited } });
  } else if (r.citation_count != null) {
    facts.push({ plural: "why.citedOnly", count: cited });
  }

  const invalid = r.invalid_citations ?? [];
  if (invalid.length > 0) {
    facts.push({ plural: "why.invalid", count: invalid.length, vars: { handles: invalid.join(", ") } });
  }

  if (r.status === "insufficient_context") facts.push({ key: "why.status.insufficient" });
  else if (r.status === "partial") facts.push({ key: "why.status.partial" });
  else if (r.status === "answered") {
    facts.push({ key: r.grounded ? "why.status.grounded" : "why.status.weak" });
  } else if (r.refusal) facts.push({ key: "why.status.refused" });
  if (r.unsupported_notes) facts.push({ key: "why.notCovered", vars: { notes: r.unsupported_notes } });

  if (r.strategy === "agentic") {
    const searches = arr(r.extra?.retrieval_searches).length;
    facts.push({ key: "why.agent", vars: { searches, iterations: r.iterations } });
    const stop = r.extra?.stop_reason;
    if (typeof stop === "string" && stop) facts.push({ key: "why.stopReason", vars: { reason: stop } });
  }

  if (r.strategy === "graph") {
    const entities = arr(r.extra?.entities).map(String);
    const related = arr(r.extra?.related_entities).length;
    if (entities.length === 0) facts.push({ key: "why.noEntities" });
    else {
      facts.push({
        plural: "why.entities",
        count: entities.length,
        vars: { list: entities.slice(0, 5).join(", "), related },
      });
    }
  }

  if (r.metrics && r.metrics.llm_calls > 0) {
    facts.push({ plural: "why.llmCalls", count: r.metrics.llm_calls });
  }
  return facts;
}

// ---------------------------------------------------------------------------
// Per-metric best values
// ---------------------------------------------------------------------------

export type Better = "higher" | "lower";

export interface CompareMetric {
  id: string;
  label: MessageKey;
  better: Better;
  kind: "score" | "latency" | "tokens" | "usd" | "count";
  value: (r: StrategyComparison) => number | null | undefined;
}

const judge = (dim: "faithfulness" | "answer_relevance" | "answer_correctness") =>
  (r: StrategyComparison) => r.evaluation?.scores?.[dim] ?? null;
const ret = (key: string) => (r: StrategyComparison) => r.evaluation?.retrieval?.[key] ?? null;

/** Total tokens (in + out); prefers the per-request metrics when present. */
export function totalTokens(r: StrategyComparison): number {
  const m = r.metrics;
  return m ? m.input_tokens + m.output_tokens : r.input_tokens + r.output_tokens;
}

/** End-to-end latency; prefers the per-request metrics when present. */
export function totalLatency(r: StrategyComparison): number {
  return r.metrics?.latency_ms.total ?? r.latency_ms;
}

export const COMPARE_METRICS: CompareMetric[] = [
  { id: "faithfulness", label: "eval.metric.faithfulness", better: "higher", kind: "score", value: judge("faithfulness") },
  { id: "answer_relevance", label: "eval.metric.answer_relevance", better: "higher", kind: "score", value: judge("answer_relevance") },
  { id: "answer_correctness", label: "metric.correctness", better: "higher", kind: "score", value: judge("answer_correctness") },
  { id: "recall@5", label: "metric.recall5", better: "higher", kind: "score", value: ret("recall@5") },
  { id: "mrr", label: "metric.mrr", better: "higher", kind: "score", value: ret("mrr") },
  { id: "latency", label: "meta.latency", better: "lower", kind: "latency", value: totalLatency },
  { id: "tokens", label: "meta.tokens", better: "lower", kind: "tokens", value: totalTokens },
  { id: "cost", label: "metric.estCost", better: "lower", kind: "usd", value: (r) => r.metrics?.estimated_cost_usd ?? null },
  { id: "llm_calls", label: "metric.llmCalls", better: "lower", kind: "count", value: (r) => r.metrics?.llm_calls ?? null },
];

/**
 * Which strategies hold the best measured value of each metric.
 *
 * Failed rows and unmeasured values are ignored. A metric is highlighted only
 * when at least two strategies measured it and they differ; ties at the best
 * value are all highlighted. This marks per-metric bests, not an overall winner.
 */
export function bestValues(
  rows: StrategyComparison[],
  metrics: CompareMetric[] = COMPARE_METRICS,
): Record<string, Set<StrategyName>> {
  const out: Record<string, Set<StrategyName>> = {};
  const live = rows.filter((r) => !isFailedRow(r));
  for (const m of metrics) {
    const measured = live
      .map((r) => ({ s: r.strategy, v: m.value(r) }))
      .filter((x): x is { s: StrategyName; v: number } => typeof x.v === "number" && Number.isFinite(x.v));
    const best = new Set<StrategyName>();
    if (measured.length >= 2) {
      const values = measured.map((x) => x.v);
      const target = m.better === "higher" ? Math.max(...values) : Math.min(...values);
      if (values.some((v) => v !== target)) {
        for (const x of measured) if (x.v === target) best.add(x.s);
      }
    }
    out[m.id] = best;
  }
  return out;
}

/** Latency stages in display order, for the stacked breakdown bar. */
export const LATENCY_STAGES = ["retrieval", "rerank", "generation", "citation_validation", "other"] as const;
export type LatencyStage = (typeof LATENCY_STAGES)[number];

/** Non-zero stage latencies of a row, or null without per-request metrics. */
export function latencyStages(r: StrategyComparison): { stage: LatencyStage; ms: number }[] | null {
  const lat = r.metrics?.latency_ms;
  if (!lat) return null;
  return LATENCY_STAGES.map((stage) => ({ stage, ms: Math.max(0, lat[stage] ?? 0) })).filter((s) => s.ms > 0);
}
