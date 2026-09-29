import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  compareStrategies,
  getEvalRun,
  getRunRegression,
  listEvalRuns,
  listGoldenDatasets,
  listGoldenQuestions,
} from "./eval";

function lastCall(): [string, RequestInit] {
  const call = vi.mocked(globalThis.fetch).mock.calls.at(-1)!;
  return [String(call[0]), (call[1] ?? {}) as RequestInit];
}

describe("eval api client", () => {
  beforeEach(() => {
    localStorage.clear();
    globalThis.fetch = vi.fn(
      async () => new Response(JSON.stringify([]), { status: 200, headers: { "Content-Type": "application/json" } }),
    ) as unknown as typeof fetch;
  });
  afterEach(() => vi.restoreAllMocks());

  it("posts the comparison request with evaluation options", async () => {
    await compareStrategies({
      question: "q",
      strategies: ["classic"],
      evaluate: true,
      golden: { dataset: "golden_v3", id: "v3-001" },
    });
    const [url, init] = lastCall();
    expect(url).toBe("/api/compare/strategies");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({
      question: "q",
      strategies: ["classic"],
      evaluate: true,
      golden: { dataset: "golden_v3", id: "v3-001" },
    });
  });

  it("fetches golden datasets and questions, encoding the name", async () => {
    await listGoldenDatasets();
    expect(lastCall()[0]).toBe("/api/eval/golden");
    await listGoldenQuestions("a b");
    expect(lastCall()[0]).toBe("/api/eval/golden/a%20b");
  });

  it("fetches runs, one run and its regression report", async () => {
    await listEvalRuns();
    expect(lastCall()[0]).toBe("/api/eval/runs");
    await getEvalRun("run/1");
    expect(lastCall()[0]).toBe("/api/eval/runs/run%2F1");
    await getRunRegression("r1");
    expect(lastCall()[0]).toBe("/api/eval/runs/r1/regression");
  });

  it("sends the saved API key", async () => {
    localStorage.setItem("evalrag.apiKey", "k-123");
    await listEvalRuns();
    const headers = lastCall()[1].headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer k-123");
  });
});
