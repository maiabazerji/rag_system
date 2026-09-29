/**
 * Render smoke tests for the Compare and Dashboard pages against a mocked
 * backend: they must render measured data, never a "winner", and explain an
 * empty dashboard.
 */
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../i18n";
import Compare from "./Compare";
import Dashboard from "./Dashboard";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// recharts' ResponsiveContainer needs ResizeObserver, which jsdom lacks.
class NoopResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

let container: HTMLDivElement;
let root: Root;
let routes: Record<string, unknown>;

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

beforeEach(() => {
  (globalThis as { ResizeObserver?: unknown }).ResizeObserver = NoopResizeObserver;
  localStorage.clear();
  routes = {};
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input).replace(/^\/api/, "");
    return path in routes ? json(routes[path]) : new Response("not found", { status: 404 });
  }) as unknown as typeof fetch;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

async function render(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  await act(async () => {
    root.render(
      <I18nProvider initialLanguage="en">
        <QueryClientProvider client={qc}>{node}</QueryClientProvider>
      </I18nProvider>,
    );
  });
  // Let queries resolve and re-render.
  for (let i = 0; i < 5; i++) await act(async () => new Promise((r) => setTimeout(r, 0)));
}

describe("Dashboard", () => {
  it("explains how to record a run when there are none", async () => {
    routes["/eval/runs"] = [];
    await render(<Dashboard />);
    expect(container.textContent).toContain("No evaluation runs yet");
    expect(container.textContent).toContain("python scripts/run_eval.py --dataset golden_v3 --all");
  });

  it("shows the latest run per configuration with the trade-off caption and no winner", async () => {
    const base = {
      dataset: "golden_v3",
      provider: "anthropic",
      model: "claude-sonnet-5",
      judge_model: "claude-opus-5",
      config_hash: "h",
      n: 10,
      n_scored: 10,
      created_at: "2026-09-01T00:00:00+00:00",
    };
    routes["/eval/runs"] = [
      { ...base, id: "c1", strategy: "classic", aggregates: { faithfulness: { mean: 0.8, std: 0.1, n: 10 } }, cost: { mean_tokens_per_question: 1200, p50_latency_ms: 800, p95_latency_ms: 1500 }, regression_status: "PASS" },
      { ...base, id: "a1", strategy: "agentic", aggregates: { faithfulness: { mean: 0.9, std: 0.05, n: 10 } }, cost: { mean_tokens_per_question: 8000 }, regression_status: "FAIL" },
    ];
    routes["/eval/runs/c1"] = { ...(routes["/eval/runs"] as object[])[0], by_question_type: { single_hop: { n_examples: 10, n_scored: 10, aggregates: { faithfulness: { mean: 0.8, std: 0.1, n: 10 } } } } };
    routes["/eval/runs/a1"] = (routes["/eval/runs"] as object[])[1];
    await render(<Dashboard />);
    const text = container.textContent ?? "";
    expect(text).toContain("Measurements from recorded runs; which trade-off is best depends on your constraints.");
    expect(text).toContain("classic · claude-sonnet-5");
    expect(text).toContain("agentic · claude-sonnet-5");
    expect(text).toContain("0.80 ± 0.10 (10)");
    expect(text).toContain("Fail");
    expect(text).toContain("single_hop");
    expect(text.toLowerCase()).not.toContain("winner");
  });
});

describe("Compare", () => {
  it("renders evaluation scores, best-value marks and the why panel", async () => {
    routes["/graph/stats"] = { triples: 0, entities: 0 };
    await render(<Compare />);
    const row = (strategy: string, faith: number, latency: number) => ({
      strategy,
      question: "q",
      answer: "Answer [S1].",
      sources: [{ chunk_id: "a:0", quote: "x", handle: "S1", title: "Guide", page: 3, relevance_score: 0.91 }],
      refusal: false,
      confidence: 0.8,
      latency_ms: latency,
      input_tokens: 100,
      output_tokens: 10,
      iterations: 1,
      trace: [],
      extra: { context_sources: [{}, {}] },
      status: "answered",
      grounded: true,
      citation_count: 1,
      invalid_citations: [],
      evaluation: {
        status: "scored",
        scores: { faithfulness: faith, answer_relevance: 0.9, context_precision: 1, context_recall: 1, answer_correctness: null },
        reasoning: {},
        retrieval: { "recall@5": 1, mrr: 0.5 },
        retrieved_docs: [],
        relevant_docs: ["a.md"],
        has_reference_answer: false,
        correct_refusal: null,
        judge_model: "j",
        rubric_version: "1",
        judge_attempts: 1,
        judge_input_tokens: 1,
        judge_output_tokens: 1,
        error_type: null,
        error: null,
      },
    });
    vi.mocked(globalThis.fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input).replace(/^\/api/, "");
      if (path === "/compare/strategies") {
        return json({ question: "q", results: [row("classic", 0.9, 800), row("graph", 0.7, 1200), row("agentic", 0.7, 3000)] });
      }
      return json({ triples: 0, entities: 0 });
    });
    const textarea = container.querySelector("textarea")!;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!;
      setter.call(textarea, "q");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const runButton = [...container.querySelectorAll("button")].find((b) => b.textContent === "Compare")!;
    await act(async () => runButton.click());
    for (let i = 0; i < 5; i++) await act(async () => new Promise((r) => setTimeout(r, 0)));

    const text = container.textContent ?? "";
    expect(text).toContain("Per-metric best values");
    expect(text).toContain("Why this result?");
    expect(text).toContain("The answer cites 1 distinct source(s); its context held 2 chunks.");
    expect(text).toContain("Guide · p. 3");
    expect(text).toContain("relevance 0.91");
    // Faithfulness: classic is the only best; latency: classic too.
    expect(container.querySelectorAll("table .sr-only").length).toBe(2);
    expect(text.toLowerCase()).not.toContain("winner");
  });
});
