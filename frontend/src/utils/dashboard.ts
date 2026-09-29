/**
 * Pure aggregation for the Evaluation Overview: group recorded eval runs into
 * configurations, pick the latest run of each, and shape them for the
 * quality-vs-cost chart and tables. No value is invented: a metric a run did
 * not measure stays null and is shown as such.
 */
import type { EvalRun, EvalRunSummary, MetricAggregate, RegressionStatus } from "../api/eval";
import type { MessageKey } from "../i18n/core";

export type QualityMetric =
  | "faithfulness"
  | "answer_relevance"
  | "answer_correctness"
  | "recall@5"
  | "mrr"
  | "ndcg@5";

export const QUALITY_METRICS: { id: QualityMetric; label: MessageKey }[] = [
  { id: "faithfulness", label: "eval.metric.faithfulness" },
  { id: "answer_relevance", label: "eval.metric.answer_relevance" },
  { id: "answer_correctness", label: "metric.correctness" },
  { id: "recall@5", label: "metric.recall5" },
  { id: "mrr", label: "metric.mrr" },
  { id: "ndcg@5", label: "metric.ndcg5" },
];

export type CostAxis = "tokens" | "cost" | "latency";

export const COST_AXES: { id: CostAxis; label: MessageKey }[] = [
  { id: "tokens", label: "dash.axis.tokens" },
  { id: "cost", label: "dash.axis.cost" },
  { id: "latency", label: "dash.axis.latency" },
];

/** The latest run of one configuration, with what the charts need. */
export interface ConfigRow {
  key: string;
  label: string;
  strategy: string;
  model: string | null;
  runId: string;
  runCount: number;
  createdAt: string | null;
  n: number;
  nScored: number;
  metrics: Partial<Record<QualityMetric, MetricAggregate>>;
  tokensPerQuestion: number | null;
  costPerQuestion: number | null;
  p50LatencyMs: number | null;
  p95LatencyMs: number | null;
  regression: RegressionStatus | null;
}

/** Datasets that have at least one run, sorted. */
export function datasetsOf(runs: EvalRunSummary[]): string[] {
  return [...new Set(runs.map((r) => r.dataset).filter((d): d is string => !!d))].sort();
}

/**
 * Identity of a configuration: runs sharing it are comparable over time. The
 * same fields the backend's regression grouping uses, plus the retrieval
 * config hash.
 */
export function configKey(r: EvalRunSummary): string {
  return [
    r.strategy ?? "",
    r.provider ?? "",
    r.model ?? "",
    r.prompt_version ?? "",
    r.judge_model ?? "",
    r.config_hash ?? "",
  ].join("|");
}

/** Short human label: "classic · claude-sonnet-5 · prompt v2 · hybrid". */
export function configLabel(r: EvalRunSummary): string {
  return [r.strategy ?? "?", r.model ?? "default", r.prompt_version, r.retrieval_mode]
    .filter((p): p is string => !!p)
    .join(" · ");
}

const time = (r: EvalRunSummary) => {
  const t = r.created_at ? Date.parse(r.created_at) : NaN;
  return Number.isNaN(t) ? -Infinity : t;
};

function costPerQuestion(r: EvalRunSummary): number | null {
  const c = r.cost;
  if (!c) return null;
  if (typeof c.mean_cost_usd_per_question === "number") return c.mean_cost_usd_per_question;
  if (typeof c.total_cost_usd === "number" && r.n > 0) return c.total_cost_usd / r.n;
  return null;
}

/**
 * One row per configuration of `dataset`: its most recent run, plus how many
 * runs it has. Sorted by strategy, then label.
 */
export function latestPerConfig(runs: EvalRunSummary[], dataset: string): ConfigRow[] {
  const groups = new Map<string, EvalRunSummary[]>();
  runs
    .filter((r) => r.dataset === dataset)
    .forEach((r) => {
      const k = configKey(r);
      groups.set(k, [...(groups.get(k) ?? []), r]);
    });

  const rows: ConfigRow[] = [];
  for (const [key, members] of groups) {
    // Stable: equal timestamps keep list order, which the API gives newest first.
    const latest = [...members].sort((a, b) => time(b) - time(a))[0];
    const agg = latest.aggregates ?? {};
    const metrics: ConfigRow["metrics"] = {};
    for (const { id } of QUALITY_METRICS) if (agg[id]) metrics[id] = agg[id];
    rows.push({
      key,
      label: configLabel(latest),
      strategy: latest.strategy ?? "?",
      model: latest.model ?? null,
      runId: latest.id,
      runCount: members.length,
      createdAt: latest.created_at ?? null,
      n: latest.n,
      nScored: latest.n_scored,
      metrics,
      tokensPerQuestion: latest.cost?.mean_tokens_per_question ?? null,
      costPerQuestion: costPerQuestion(latest),
      p50LatencyMs: latest.cost?.p50_latency_ms ?? null,
      p95LatencyMs: latest.cost?.p95_latency_ms ?? null,
      regression: latest.regression_status ?? null,
    });
  }
  return rows.sort((a, b) => a.strategy.localeCompare(b.strategy) || a.label.localeCompare(b.label));
}

/** Whether any configuration reports a USD cost, so the cost axis can be offered. */
export function hasCostData(rows: ConfigRow[]): boolean {
  return rows.some((r) => r.costPerQuestion != null);
}

export function axisValue(r: ConfigRow, axis: CostAxis): number | null {
  if (axis === "tokens") return r.tokensPerQuestion;
  if (axis === "cost") return r.costPerQuestion;
  return r.p50LatencyMs;
}

export interface ScatterPoint {
  key: string;
  label: string;
  strategy: string;
  x: number;
  y: number;
  n: number;
  std: number | null;
}

/** Points with both coordinates measured; configurations missing either are listed separately. */
export function scatterPoints(
  rows: ConfigRow[],
  metric: QualityMetric,
  axis: CostAxis,
): { points: ScatterPoint[]; missing: string[] } {
  const points: ScatterPoint[] = [];
  const missing: string[] = [];
  for (const r of rows) {
    const agg = r.metrics[metric];
    const x = axisValue(r, axis);
    if (agg?.mean == null || x == null) {
      missing.push(r.label);
      continue;
    }
    points.push({ key: r.key, label: r.label, strategy: r.strategy, x, y: agg.mean, n: agg.n, std: agg.std });
  }
  return { points, missing };
}

/** Points grouped into one series per strategy, sorted by strategy name. */
export function seriesByStrategy(points: ScatterPoint[]): [string, ScatterPoint[]][] {
  const by = new Map<string, ScatterPoint[]>();
  points.forEach((p) => by.set(p.strategy, [...(by.get(p.strategy) ?? []), p]));
  return [...by.entries()].sort(([a], [b]) => a.localeCompare(b));
}

export interface BreakdownTable {
  types: string[];
  rows: { key: string; label: string; strategy: string; cells: Record<string, MetricAggregate | null> }[];
}

/**
 * Per-question_type values of one metric for each configuration's latest run.
 * `details` maps run id to the full run (only full runs carry breakdowns).
 */
export function questionTypeBreakdown(
  rows: ConfigRow[],
  details: Record<string, EvalRun | undefined>,
  metric: QualityMetric,
): BreakdownTable {
  const types = new Set<string>();
  const out: BreakdownTable["rows"] = [];
  for (const r of rows) {
    const by = details[r.runId]?.by_question_type;
    if (!by) continue;
    const cells: Record<string, MetricAggregate | null> = {};
    for (const [type, b] of Object.entries(by)) {
      types.add(type);
      cells[type] = b.aggregates?.[metric] ?? null;
    }
    out.push({ key: r.key, label: r.label, strategy: r.strategy, cells });
  }
  return { types: [...types].sort(), rows: out };
}
