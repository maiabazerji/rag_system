/**
 * Render smoke tests for the pages against a mocked backend: every route
 * renders, the routes merged into the Evaluation page redirect to it, the
 * Evaluation tabs switch by mouse and keyboard, and no page shows a "winner",
 * a "best strategy" or a latency figure nobody measured.
 */
import { act, useEffect, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../i18n";
import AppRoutes, { NAV } from "../AppRoutes";
import Compare from "./Compare";

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
let location: { pathname: string; search: string };

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

async function settle(rounds = 8) {
  for (let i = 0; i < rounds; i++) await act(async () => new Promise((r) => setTimeout(r, 0)));
}

async function render(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  await act(async () => {
    root.render(
      <I18nProvider initialLanguage="en">
        <QueryClientProvider client={qc}>{node}</QueryClientProvider>
      </I18nProvider>,
    );
  });
  // Let queries (and the lazily loaded Overview) resolve and re-render.
  await settle();
}

function LocationProbe() {
  const { pathname, search } = useLocation();
  useEffect(() => {
    location = { pathname, search };
  }, [pathname, search]);
  return null;
}

async function renderAt(path: string) {
  await render(
    <MemoryRouter initialEntries={[path]}>
      <LocationProbe />
      <AppRoutes />
    </MemoryRouter>,
  );
}

/** Wait until `check` passes (lazy chunks can take more than a few ticks). */
async function until(check: () => boolean, tries = 50) {
  for (let i = 0; i < tries && !check(); i++) await act(async () => new Promise((r) => setTimeout(r, 10)));
}

const text = () => container.textContent ?? "";
const tab = (name: string) =>
  [...container.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((b) => b.textContent === name)!;

async function click(el: Element) {
  await act(async () => {
    (el as HTMLElement).click();
  });
  await settle();
}

async function key(el: Element, k: string) {
  await act(async () => {
    el.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true }));
  });
  await settle();
}

const base = {
  dataset: "golden_v3",
  provider: "anthropic",
  model: "claude-sonnet-5",
  judge_model: "claude-opus-5",
  config_hash: "h",
  n: 10,
  n_scored: 10,
};

describe("routes", () => {
  it("the navigation lists only the consolidated pages", () => {
    expect(NAV.map((n) => n.to)).toEqual(["/", "/ingest", "/compare", "/eval", "/advisor"]);
  });

  it.each(["/", "/ingest", "/compare", "/eval", "/advisor"])("%s renders a page heading", async (path) => {
    await renderAt(path);
    expect(container.querySelector("h1")).not.toBeNull();
    expect(location.pathname).toBe(path);
  });

  it.each([
    ["/dashboard", "?tab=overview", "Overview"],
    ["/regressions", "?tab=runs", "Runs"],
    ["/data", "", "Overview"],
  ])("%s redirects to the Evaluation page", async (from, search, selected) => {
    routes["/eval/runs"] = [];
    routes["/eval/golden"] = ["golden_v1"];
    await renderAt(from);
    expect(location).toEqual({ pathname: "/eval", search });
    expect(container.querySelector("h1")?.textContent).toBe("Evaluation");
    expect(tab(selected).getAttribute("aria-selected")).toBe("true");
  });

  it.each(["/", "/ingest", "/compare", "/eval", "/eval?tab=runs", "/advisor"])(
    "%s links to no removed route and invents no figures",
    async (path) => {
      routes["/eval/runs"] = [];
      routes["/eval/golden"] = ["golden_v1"];
      await renderAt(path);
      for (const removed of ["/data", "/dashboard", "/regressions"]) {
        expect(container.querySelector(`a[href="${removed}"]`)).toBeNull();
      }
      const lower = text().toLowerCase();
      expect(lower).not.toContain("winner");
      expect(lower).not.toContain("best strategy");
      // With no recorded run there is no measured latency to show.
      expect(text()).not.toMatch(/\d\s?(ms|s)\b/);
    },
  );
});

describe("Evaluation tabs", () => {
  beforeEach(() => {
    routes["/eval/runs"] = [];
    routes["/eval/golden"] = ["golden_v1", "golden_v3"];
  });

  it("uses the ARIA tabs pattern with the Overview selected by default", async () => {
    await renderAt("/eval");
    const list = container.querySelector('[role="tablist"]')!;
    expect(list.getAttribute("aria-label")).toBe("Evaluation views");
    const overview = tab("Overview");
    const runs = tab("Runs");
    expect(overview.getAttribute("aria-selected")).toBe("true");
    expect(overview.tabIndex).toBe(0);
    expect(runs.getAttribute("aria-selected")).toBe("false");
    expect(runs.tabIndex).toBe(-1);
    const panel = container.querySelector('[role="tabpanel"]')!;
    expect(panel.getAttribute("aria-labelledby")).toBe(overview.id);
    expect(overview.getAttribute("aria-controls")).toBe(panel.id);
  });

  it("switches tabs on click and records the tab in the URL", async () => {
    await renderAt("/eval");
    await click(tab("Runs"));
    expect(location.search).toBe("?tab=runs");
    expect(tab("Runs").getAttribute("aria-selected")).toBe("true");
    expect(text()).toContain("Run history");
    await click(tab("Overview"));
    await until(() => text().includes("No evaluation runs yet"));
    expect(location.search).toBe("?tab=overview");
    expect(text()).not.toContain("Run history");
  });

  it("moves between tabs with the arrow, Home and End keys", async () => {
    await renderAt("/eval");
    await key(tab("Overview"), "ArrowRight");
    expect(location.search).toBe("?tab=runs");
    expect(document.activeElement).toBe(tab("Runs"));
    await key(tab("Runs"), "ArrowRight");
    expect(location.search).toBe("?tab=overview");
    await key(tab("Overview"), "End");
    expect(location.search).toBe("?tab=runs");
    await key(tab("Runs"), "Home");
    expect(location.search).toBe("?tab=overview");
    expect(document.activeElement).toBe(tab("Overview"));
  });

  it("falls back to the Overview for an unknown tab", async () => {
    await renderAt("/eval?tab=nope");
    expect(tab("Overview").getAttribute("aria-selected")).toBe("true");
  });
});

describe("Evaluation overview", () => {
  it("explains how to record a run when there are none", async () => {
    routes["/eval/runs"] = [];
    await renderAt("/eval?tab=overview");
    await until(() => text().includes("No evaluation runs yet"));
    expect(text()).toContain("No evaluation runs yet");
    expect(text()).toContain("python scripts/run_eval.py --dataset golden_v3 --all");
  });

  it("shows the latest run per configuration with the trade-off caption and no winner", async () => {
    const at = { ...base, created_at: "2026-09-01T00:00:00+00:00" };
    routes["/eval/runs"] = [
      { ...at, id: "c1", strategy: "classic", aggregates: { faithfulness: { mean: 0.8, std: 0.1, n: 10 } }, cost: { mean_tokens_per_question: 1200, p50_latency_ms: 800, p95_latency_ms: 1500 }, regression_status: "PASS" },
      { ...at, id: "a1", strategy: "agentic", aggregates: { faithfulness: { mean: 0.9, std: 0.05, n: 10 } }, cost: { mean_tokens_per_question: 8000 }, regression_status: "FAIL" },
    ];
    routes["/eval/runs/c1"] = { ...(routes["/eval/runs"] as object[])[0], by_question_type: { single_hop: { n_examples: 10, n_scored: 10, aggregates: { faithfulness: { mean: 0.8, std: 0.1, n: 10 } } } } };
    routes["/eval/runs/a1"] = (routes["/eval/runs"] as object[])[1];
    await renderAt("/eval");
    await until(() => text().includes("single_hop"));
    const t = text();
    expect(t).toContain("Measurements from recorded runs; which trade-off is best depends on your constraints.");
    expect(t).toContain("classic · claude-sonnet-5");
    expect(t).toContain("agentic · claude-sonnet-5");
    expect(t).toContain("0.80 ± 0.10 (10)");
    expect(t).toContain("Fail");
    expect(t).toContain("single_hop");
    expect(t.toLowerCase()).not.toContain("winner");
    expect(t.toLowerCase()).not.toContain("best strategy");
  });
});

describe("Evaluation runs", () => {
  const select = (label: string) =>
    [...container.querySelectorAll("label")].find((l) => l.textContent?.startsWith(label))!.querySelector("select")!;

  it("lists the golden datasets from the API and notes the demo corpus only when needed", async () => {
    routes["/eval/runs"] = [];
    routes["/eval/golden"] = ["golden_fr_business_v1", "golden_v1", "golden_v3"];
    await renderAt("/eval?tab=runs");
    const dataset = select("Dataset");
    expect([...dataset.options].map((o) => o.value)).toEqual(["golden_fr_business_v1", "golden_v1", "golden_v3"]);
    expect(dataset.value).toBe("golden_v1");
    expect(text()).not.toContain("demo_fr_business");
    await act(async () => {
      dataset.value = "golden_fr_business_v1";
      dataset.dispatchEvent(new Event("change", { bubbles: true }));
    });
    const note = container.querySelector('[role="note"]')!;
    expect(note.textContent).toContain("data/demo_fr_business/README.md");
    expect(dataset.getAttribute("aria-describedby")).toBe(note.id);
    expect([...select("Strategy").options].map((o) => o.value)).toEqual(["classic", "graph", "agentic"]);
  });

  it("launches a run with the chosen dataset and strategy", async () => {
    routes["/eval/runs"] = [];
    routes["/eval/golden"] = ["golden_v1", "golden_v3"];
    routes["/eval/run"] = { ...base, id: "new", strategy: "graph" };
    await renderAt("/eval?tab=runs");
    const strategy = select("Strategy");
    await act(async () => {
      select("Dataset").value = "golden_v3";
      select("Dataset").dispatchEvent(new Event("change", { bubbles: true }));
      strategy.value = "graph";
      strategy.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await click([...container.querySelectorAll("button")].find((b) => b.textContent === "Run Evaluation")!);
    const call = vi.mocked(globalThis.fetch).mock.calls.find(([u]) => String(u).endsWith("/eval/run"))!;
    expect((call[1] as RequestInit).method).toBe("POST");
    expect(JSON.parse(String((call[1] as RequestInit).body))).toEqual({ dataset: "golden_v3", strategy: "graph" });
  });

  it("shows the history newest first with counts, aggregates and the regression report", async () => {
    routes["/eval/golden"] = ["golden_v3"];
    routes["/eval/runs"] = [
      { ...base, id: "old", strategy: "classic", created_at: "2026-08-01T00:00:00+00:00", aggregate: { faithfulness: 0.7 } },
      {
        ...base,
        id: "new",
        strategy: "agentic",
        created_at: "2026-09-01T00:00:00+00:00",
        n_scored: 8,
        n_judge_failed: 1,
        n_unscored: 2,
        n_generation_failed: 1,
        regression_status: "FAIL",
        aggregates: {
          faithfulness: { mean: 0.9, std: 0.05, n: 8 },
          answer_correctness: { mean: 0.6, std: 0.2, n: 5 },
          "recall@5": { mean: 0.75, std: 0.3, n: 8 },
          mrr: { mean: 0.5, std: 0.4, n: 8 },
          "ndcg@5": { mean: 0.55, std: 0.3, n: 8 },
        },
      },
    ];
    routes["/eval/runs/new/regression"] = {
      status: "FAIL",
      baseline: { id: "prev-run" },
      baseline_source: "previous",
      checks: [
        { metric: "faithfulness", status: "FAIL", baseline: 0.95, current: 0.9, delta: -0.05, threshold: "delta >= -0.03", reason: "delta -0.0500 < -0.03" },
        { metric: "recall@5", status: "PASS", baseline: 0.7, current: 0.75, delta: 0.05, threshold: "delta >= -0.03" },
        { metric: "latency_p50_ms", status: "SKIPPED", baseline: null, current: 900, delta: null, threshold: "change <= +20%", reason: "insufficient data" },
      ],
    };
    await renderAt("/eval?tab=runs");
    const cards = [...container.querySelectorAll("article")];
    expect(cards.map((c) => c.querySelector("h3")?.textContent)).toEqual(["golden_v3", "golden_v3"]);
    const [newest, oldest] = cards;
    expect(newest.textContent).toContain("agentic");
    expect(oldest.textContent).toContain("classic");
    const t = newest.textContent ?? "";
    expect(t).toContain("model claude-sonnet-5");
    expect(t).toContain("Scored / n8 / 10");
    expect(t).toContain("Judge failed1");
    expect(t).toContain("Unscored2");
    expect(t).toContain("Correctness0.60 ± 0.20 (5)");
    expect(t).toContain("Recall@50.75");
    expect(t).toContain("MRR0.50");
    expect(t).toContain("nDCG@50.55");
    // A legacy run with flat means still shows them, without an invented n.
    expect(oldest.textContent).toContain("Faithfulness0.70");
    expect(oldest.textContent).not.toContain("(10)");

    const toggle = [...newest.querySelectorAll("button")].find((b) => b.textContent === "Show regression report")!;
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    await click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById(toggle.getAttribute("aria-controls")!)).not.toBeNull();
    const rows = [...newest.querySelectorAll("tbody tr")].map((r) => [...r.children].map((c) => c.textContent));
    expect(rows).toEqual([
      ["✕Fail", "faithfulness", "0.950", "0.900", "−0.050", "delta >= -0.03", "delta -0.0500 < -0.03"],
      ["✓Pass", "recall@5", "0.700", "0.750", "+0.050", "delta >= -0.03", ""],
      ["–Skipped", "latency_p50_ms", "n/a", "900ms", "n/a", "change <= +20%", "insufficient data"],
    ]);
    expect(newest.textContent).toContain("Baseline: prev-run (previous run of this configuration)");
    await click(toggle);
    expect(newest.querySelector("table")).toBeNull();
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
