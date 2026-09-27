import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  MAX_VALIDATE_QUESTIONS,
  advise,
  cleanOverrides,
  parseQuestions,
  validateStrategies,
} from "./advisor";
import { setApiKey } from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function lastCall(): { url: string; init: RequestInit } {
  const call = vi.mocked(globalThis.fetch).mock.calls.at(-1)!;
  return { url: String(call[0]), init: call[1] ?? {} };
}

describe("advisor client", () => {
  beforeEach(() => {
    localStorage.clear();
    globalThis.fetch = vi.fn(async () => jsonResponse({ ok: true })) as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("parses one question per line with optional ideal answers", () => {
    const qs = parseQuestions(
      "What is X?\n\n  Who owns Y? || ACME Corp  \nBlank ideal? ||   \n|| orphan answer",
    );
    expect(qs).toEqual([
      { question: "What is X?" },
      { question: "Who owns Y?", ideal_answer: "ACME Corp" },
      { question: "Blank ideal?" },
    ]);
  });

  it("caps parsed questions", () => {
    const text = Array.from({ length: 30 }, (_, i) => `q${i}`).join("\n");
    expect(parseQuestions(text)).toHaveLength(MAX_VALIDATE_QUESTIONS);
  });

  it("drops empty override fields", () => {
    expect(
      cleanOverrides({
        corpus_size_docs: undefined,
        languages: [],
        latency_budget_ms: Number.NaN,
        cost_sensitivity: "high",
        compliance: ["EU only"],
      }),
    ).toEqual({ cost_sensitivity: "high", compliance: ["EU only"] });
  });

  it("posts /advise with cleaned overrides and the auth header", async () => {
    setApiKey("sk_test");
    await advise("Nous avons 5 000 contrats", { languages: ["fr"], compliance: [] });
    const { url, init } = lastCall();
    expect(url).toBe("/api/advise");
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer sk_test");
    expect(JSON.parse(String(init.body))).toEqual({
      description: "Nous avons 5 000 contrats",
      overrides: { languages: ["fr"] },
    });
  });

  it("posts /advise/validate and omits empty strategies", async () => {
    await validateStrategies([{ question: "q?" }], []);
    const { url, init } = lastCall();
    expect(url).toBe("/api/advise/validate");
    expect(JSON.parse(String(init.body))).toEqual({ questions: [{ question: "q?" }] });

    await validateStrategies([{ question: "q?" }], ["classic", "graph"]);
    expect(JSON.parse(String(lastCall().init.body)).strategies).toEqual(["classic", "graph"]);
  });

  it("surfaces backend error messages", async () => {
    globalThis.fetch = vi.fn(async () =>
      jsonResponse({ detail: "questions too long" }, 422),
    ) as unknown as typeof fetch;
    await expect(advise("x")).rejects.toThrow("questions too long");
  });
});
