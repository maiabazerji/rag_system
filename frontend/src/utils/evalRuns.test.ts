import { describe, expect, it } from "vitest";
import type { EvalRunSummary } from "../api/eval";
import { formatCheckDelta, formatCheckValue, newestFirst, runMetrics } from "./evalRuns";

const run = (over: Partial<EvalRunSummary>): EvalRunSummary => ({
  id: "r",
  dataset: "golden_v3",
  strategy: "classic",
  n: 10,
  n_scored: 10,
  ...over,
});

describe("runMetrics", () => {
  it("lists measured aggregates in display order and skips unmeasured ones", () => {
    const metrics = runMetrics(
      run({
        aggregates: {
          mrr: { mean: 0.5, std: 0.1, n: 10 },
          faithfulness: { mean: 0.9, std: null, n: 9 },
          answer_correctness: { mean: null, std: null, n: 0 },
        },
      }),
    );
    expect(metrics.map((m) => [m.id, m.mean, m.n])).toEqual([
      ["faithfulness", 0.9, 9],
      ["mrr", 0.5, 10],
    ]);
  });

  it("reads the flat means of older runs without inventing an n", () => {
    const metrics = runMetrics(run({ aggregate: { faithfulness: 0.7 }, retrieval_aggregate: { "recall@5": 0.4 } }));
    expect(metrics.map((m) => [m.id, m.mean, m.std, m.n])).toEqual([
      ["faithfulness", 0.7, null, null],
      ["recall@5", 0.4, null, null],
    ]);
  });

  it("returns nothing for a run the judge never scored", () => {
    expect(runMetrics(run({ aggregates: null, aggregate: null }))).toEqual([]);
  });
});

describe("newestFirst", () => {
  it("sorts by date and keeps the API order for ties and undated runs", () => {
    const runs = [
      run({ id: "a", created_at: "2026-01-01T00:00:00Z" }),
      run({ id: "b", created_at: "2026-03-01T00:00:00Z" }),
      run({ id: "c" }),
      run({ id: "d", created_at: "2026-03-01T00:00:00Z" }),
    ];
    expect(newestFirst(runs).map((r) => r.id)).toEqual(["b", "d", "a", "c"]);
  });
});

describe("regression check formatting", () => {
  it("formats scores, latencies and token counts in their own units", () => {
    expect(formatCheckValue("faithfulness", 0.9, "en-US")).toBe("0.900");
    expect(formatCheckValue("latency_p50_ms", 1500, "en-US")).toBe("1.50s");
    expect(formatCheckValue("tokens_per_question", 1234, "en-US")).toBe("1.2K");
    expect(formatCheckDelta("faithfulness", -0.05, "en-US")).toBe("−0.050");
    expect(formatCheckDelta("mrr", 0.05, "en-US")).toBe("+0.050");
    expect(formatCheckDelta("mrr", 0, "en-US")).toBe("0.000");
  });
});
