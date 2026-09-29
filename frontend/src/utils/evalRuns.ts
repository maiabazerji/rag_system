/**
 * Pure helpers for the Evaluation page's run history: which metrics a run
 * measured, in a fixed order, and how to show a regression check's values.
 * Nothing is invented: a metric a run did not measure is simply absent.
 */
import type { EvalRunSummary } from "../api/eval";
import type { MessageKey } from "../i18n/core";
import { formatLatency, formatScore, formatTokens } from "./formatting";

/** Metrics shown for a run, in display order. */
export const RUN_METRICS: { id: string; label: MessageKey }[] = [
  { id: "faithfulness", label: "eval.metric.faithfulness" },
  { id: "answer_relevance", label: "eval.metric.answer_relevance" },
  { id: "context_precision", label: "eval.metric.context_precision" },
  { id: "context_recall", label: "eval.metric.context_recall" },
  { id: "answer_correctness", label: "metric.correctness" },
  { id: "recall@5", label: "metric.recall5" },
  { id: "mrr", label: "metric.mrr" },
  { id: "ndcg@5", label: "metric.ndcg5" },
];

/**
 * Datasets labelled against a corpus other than data/docs, which must be
 * ingested into its own collection first (see data/golden/corpora.toml).
 */
export const DEMO_CORPUS_DATASETS: ReadonlySet<string> = new Set(["golden_fr_business_v1"]);

export interface RunMetric {
  id: string;
  label: MessageKey;
  mean: number;
  std: number | null;
  /** Examples behind the mean; null for older runs that did not record it. */
  n: number | null;
}

/**
 * The metrics a run measured. Current runs carry `aggregates` (mean, std, n);
 * older runs only flat means in `aggregate` / `retrieval_aggregate`.
 */
export function runMetrics(run: EvalRunSummary): RunMetric[] {
  const out: RunMetric[] = [];
  for (const { id, label } of RUN_METRICS) {
    const agg = run.aggregates?.[id];
    if (agg && agg.mean != null) {
      out.push({ id, label, mean: agg.mean, std: agg.std, n: agg.n });
      continue;
    }
    const flat = run.aggregate?.[id] ?? run.retrieval_aggregate?.[id];
    if (typeof flat === "number") out.push({ id, label, mean: flat, std: null, n: null });
  }
  return out;
}

const time = (r: EvalRunSummary) => {
  const t = r.created_at ? Date.parse(r.created_at) : NaN;
  return Number.isNaN(t) ? -Infinity : t;
};

/** Runs sorted newest first; ties (and undated runs) keep the API's order. */
export function newestFirst<T extends EvalRunSummary>(runs: T[]): T[] {
  return runs
    .map((r, i) => [r, i] as const)
    .sort(([a, i], [b, j]) => time(b) - time(a) || i - j)
    .map(([r]) => r);
}

/** A regression check's baseline or current value, in the metric's own unit. */
export function formatCheckValue(metric: string, value: number, locale?: string): string {
  if (metric.endsWith("_ms")) return formatLatency(value, locale);
  if (metric.includes("tokens")) return formatTokens(Math.round(value), locale);
  return formatScore(value, 3, locale);
}

/** A signed delta: "+0.012", "−120 ms". */
export function formatCheckDelta(metric: string, delta: number, locale?: string): string {
  const sign = delta > 0 ? "+" : delta < 0 ? "−" : "";
  return sign + formatCheckValue(metric, Math.abs(delta), locale);
}
