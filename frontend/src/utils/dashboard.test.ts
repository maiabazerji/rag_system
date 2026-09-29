import { describe, expect, it } from "vitest";
import type { EvalRun, EvalRunSummary } from "../api/eval";
import {
  configKey,
  datasetsOf,
  hasCostData,
  latestPerConfig,
  questionTypeBreakdown,
  scatterPoints,
  seriesByStrategy,
} from "./dashboard";

function run(over: Partial<EvalRunSummary>): EvalRunSummary {
  return {
    id: "r",
    dataset: "golden_v3",
    strategy: "classic",
    provider: "anthropic",
    model: "claude-sonnet-5",
    prompt_version: null,
    judge_model: "claude-opus-5",
    config_hash: "abc",
    retrieval_mode: "hybrid",
    n: 10,
    n_scored: 9,
    aggregates: { faithfulness: { mean: 0.8, std: 0.1, n: 9 }, "recall@5": { mean: 0.7, std: 0.2, n: 8 } },
    cost: { mean_tokens_per_question: 1500, p50_latency_ms: 900, p95_latency_ms: 2100 },
    regression_status: "PASS",
    created_at: "2026-09-01T10:00:00+00:00",
    ...over,
  };
}

const RUNS: EvalRunSummary[] = [
  run({ id: "c-new", created_at: "2026-09-02T10:00:00+00:00", regression_status: "FAIL" }),
  run({ id: "c-old" }),
  run({ id: "a-1", strategy: "agentic", aggregates: { faithfulness: { mean: 0.9, std: null, n: 1 } }, cost: { mean_tokens_per_question: 9000, total_cost_usd: 0.5 } }),
  run({ id: "g-1", strategy: "graph", aggregates: {}, cost: null, regression_status: null }),
  run({ id: "other", dataset: "golden_v1" }),
];

describe("dashboard aggregation", () => {
  it("lists datasets with runs", () => {
    expect(datasetsOf([...RUNS, run({ dataset: null })])).toEqual(["golden_v1", "golden_v3"]);
  });

  it("keeps the latest run per configuration and counts the others", () => {
    const rows = latestPerConfig(RUNS, "golden_v3");
    expect(rows.map((r) => r.runId)).toEqual(["a-1", "c-new", "g-1"]);
    const classic = rows.find((r) => r.strategy === "classic")!;
    expect(classic.runCount).toBe(2);
    expect(classic.regression).toBe("FAIL");
    expect(classic.tokensPerQuestion).toBe(1500);
    expect(classic.p95LatencyMs).toBe(2100);
    expect(classic.costPerQuestion).toBeNull();
    expect(classic.label).toBe("classic · claude-sonnet-5 · hybrid");
  });

  it("separates configurations that differ in model or retrieval config", () => {
    expect(configKey(run({}))).not.toBe(configKey(run({ model: "other" })));
    expect(configKey(run({}))).not.toBe(configKey(run({ config_hash: "def" })));
    expect(configKey(run({ id: "x" }))).toBe(configKey(run({ id: "y" })));
  });

  it("derives cost per question from the run total when no mean is recorded", () => {
    const rows = latestPerConfig(RUNS, "golden_v3");
    expect(rows.find((r) => r.strategy === "agentic")!.costPerQuestion).toBeCloseTo(0.05);
    expect(hasCostData(rows)).toBe(true);
    expect(hasCostData(latestPerConfig(RUNS, "golden_v1"))).toBe(false);
  });

  it("plots only configurations that measured both coordinates", () => {
    const rows = latestPerConfig(RUNS, "golden_v3");
    const { points, missing } = scatterPoints(rows, "faithfulness", "tokens");
    expect(points.map((p) => [p.strategy, p.x, p.y])).toEqual([
      ["agentic", 9000, 0.9],
      ["classic", 1500, 0.8],
    ]);
    expect(missing).toEqual(["graph · claude-sonnet-5 · hybrid"]);
    // Classic has no USD cost, so it drops out of the cost axis.
    expect(scatterPoints(rows, "faithfulness", "cost").points.map((p) => p.strategy)).toEqual(["agentic"]);
    expect(scatterPoints(rows, "recall@5", "latency").points).toEqual([
      expect.objectContaining({ strategy: "classic", x: 900, y: 0.7, n: 8, std: 0.2 }),
    ]);
  });

  it("groups points into one series per strategy", () => {
    const { points } = scatterPoints(latestPerConfig(RUNS, "golden_v3"), "faithfulness", "tokens");
    expect(seriesByStrategy(points).map(([s, p]) => [s, p.length])).toEqual([
      ["agentic", 1],
      ["classic", 1],
    ]);
  });

  it("builds the question-type breakdown from run details", () => {
    const rows = latestPerConfig(RUNS, "golden_v3");
    const detail: EvalRun = {
      ...RUNS[0],
      by_question_type: {
        single_hop: { n_examples: 5, n_scored: 5, aggregates: { faithfulness: { mean: 0.9, std: 0.1, n: 5 } } },
        multi_hop: { n_examples: 5, n_scored: 4, aggregates: {} },
      },
    };
    const table = questionTypeBreakdown(rows, { "c-new": detail }, "faithfulness");
    expect(table.types).toEqual(["multi_hop", "single_hop"]);
    expect(table.rows).toHaveLength(1);
    expect(table.rows[0].cells).toEqual({ single_hop: { mean: 0.9, std: 0.1, n: 5 }, multi_hop: null });
  });
});
