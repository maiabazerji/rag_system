/**
 * The static (Netlify) demo: no backend. The Advisor must work with every
 * network call failing, the Lab must be hidden, and the optional browser key
 * must send the description to Anthropic only.
 */
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../i18n";
import AppRoutes, { visibleNav } from "../AppRoutes";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.stubEnv("VITE_STATIC_DEMO", "1");
  sessionStorage.clear();
  localStorage.clear();
  globalThis.fetch = vi.fn(async () => {
    throw new TypeError("no network in the static demo");
  }) as unknown as typeof fetch;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

async function settle(rounds = 8) {
  for (let i = 0; i < rounds; i++) await act(async () => new Promise((r) => setTimeout(r, 0)));
}

async function renderAt(path: string, node: ReactNode = <AppRoutes />) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  await act(async () => {
    root.render(
      <I18nProvider initialLanguage="en">
        <QueryClientProvider client={qc}>
          <MemoryRouter initialEntries={[path]}>{node}</MemoryRouter>
        </QueryClientProvider>
      </I18nProvider>,
    );
  });
  await settle();
}

const text = () => container.textContent ?? "";
const button = (label: string) =>
  [...container.querySelectorAll("button")].find((b) => b.textContent === label)!;

async function click(el: Element) {
  await act(async () => {
    (el as HTMLElement).click();
  });
  await settle();
}

async function type(el: HTMLTextAreaElement | HTMLInputElement, value: string) {
  const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  await act(async () => {
    Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(el, value);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

describe("static demo", () => {
  it("shows only the Advisor and sends Lab paths home", async () => {
    expect(visibleNav().map((n) => n.to)).toEqual(["/"]);
    await renderAt("/compare");
    expect(container.querySelector("h1")?.textContent).toBe("Find the right RAG for your project");
  });

  it("runs the whole Advisor flow with no backend", async () => {
    await renderAt("/");
    await type(container.querySelector("textarea")!, "Assistant RH pour nos 300 documents, réponse en 2 secondes ?");
    await click(button("Continue"));
    // Either follow-up questions, or straight to the result: both computed locally.
    if (text().includes("A few questions to get it right")) await click(button("See my recommendation"));
    expect(text()).toContain("We recommend");
    expect(text()).toContain("Testing on your own documents needs the full app, run locally.");
    expect(text()).not.toContain("Test it on your documents");
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("sends the description to Anthropic only, with the visitor's key", async () => {
    const calls: [string, RequestInit][] = [];
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push([String(input), init!]);
      return new Response(
        JSON.stringify({
          content: [
            {
              type: "tool_use",
              name: "record_project_profile",
              input: {
                corpus_size: "xlarge",
                corpus_size_docs: 250000,
                document_types: ["contracts"],
                languages: ["en"],
                question_mix: { single_fact: 20, relational_multi_hop: 70, exploratory_multi_step: 10 },
                cost_sensitivity: "medium",
                data_freshness: "monthly",
                compliance: [],
                entity_richness: "high",
                example_questions: ["Which subsidiaries does supplier X serve?"],
              },
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as unknown as typeof fetch;

    await renderAt("/");
    await type(container.querySelector('input[type="password"]')!, "sk-ant-visitor");
    expect(sessionStorage.getItem("veritas.anthropicKey")).toBe("sk-ant-visitor");
    expect(localStorage.length).toBe(0);
    await type(container.querySelector("textarea")!, "Supplier contracts and subsidiaries.");
    await click(button("Continue"));
    if (text().includes("A few questions to get it right")) await click(button("See my recommendation"));

    expect(calls.length).toBeGreaterThan(0);
    for (const [url, init] of calls) {
      expect(url).toBe("https://api.anthropic.com/v1/messages");
      const headers = init.headers as Record<string, string>;
      expect(headers["x-api-key"]).toBe("sk-ant-visitor");
      expect(headers["anthropic-dangerous-direct-browser-access"]).toBe("true");
      expect(JSON.parse(String(init.body)).model).toBe("claude-haiku-4-5");
    }
    expect(container.querySelector("h2")?.textContent).toBe("Graph RAG");
    expect(text()).toContain("extracted by Claude");
  });

  it("falls back to the keyword rules when the key fails", async () => {
    globalThis.fetch = vi.fn(async () => new Response("{}", { status: 401 })) as unknown as typeof fetch;
    sessionStorage.setItem("veritas.anthropicKey", "sk-ant-bad");
    await renderAt("/");
    await type(container.querySelector("textarea")!, "FAQ chatbot, 150 articles.");
    await click(button("Continue"));
    if (text().includes("A few questions to get it right")) await click(button("See my recommendation"));
    expect(text()).toContain("We recommend");
    expect(text()).toContain("keyword heuristic");
  });
});
