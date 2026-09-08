/**
 * Tests for the API client.
 *
 * This file exists because of a specific regression: the backend gained auth
 * dependencies while these helpers still sent no Authorization header, so every
 * button in the UI returned 401. The header tests below are the guard.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  UnauthorizedError,
  fetchHealth,
  get,
  getApiKey,
  post,
  setApiKey,
  upload,
} from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function lastRequestHeaders(): Record<string, string> {
  const call = vi.mocked(globalThis.fetch).mock.calls.at(-1);
  return (call?.[1]?.headers ?? {}) as Record<string, string>;
}

describe("api client", () => {
  beforeEach(() => {
    localStorage.clear();
    globalThis.fetch = vi.fn(async () => jsonResponse({ ok: true })) as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("key storage", () => {
    it("returns null when nothing is stored", () => {
      expect(getApiKey()).toBeNull();
    });

    it("round-trips a key", () => {
      setApiKey("sk_abc123");
      expect(getApiKey()).toBe("sk_abc123");
    });

    it("trims surrounding whitespace on save", () => {
      setApiKey("  sk_abc123\n");
      expect(getApiKey()).toBe("sk_abc123");
    });

    it("clears the key when given null", () => {
      setApiKey("sk_abc123");
      setApiKey(null);
      expect(getApiKey()).toBeNull();
    });

    it("treats a blank string as a clear", () => {
      setApiKey("sk_abc123");
      setApiKey("   ");
      expect(getApiKey()).toBeNull();
    });

    it("survives localStorage throwing", () => {
      const spy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
        throw new Error("blocked in private browsing");
      });
      expect(getApiKey()).toBeNull();
      spy.mockRestore();
    });
  });

  describe("authorization header", () => {
    it("post sends the key when one is stored", async () => {
      setApiKey("sk_abc123");
      await post("/ask", { question: "hi" });
      expect(lastRequestHeaders().Authorization).toBe("Bearer sk_abc123");
    });

    it("get sends the key when one is stored", async () => {
      setApiKey("sk_abc123");
      await get("/ingest/stats");
      expect(lastRequestHeaders().Authorization).toBe("Bearer sk_abc123");
    });

    it("upload sends the key when one is stored", async () => {
      setApiKey("sk_abc123");
      await upload("/ingest", new File(["body"], "a.txt"));
      expect(lastRequestHeaders().Authorization).toBe("Bearer sk_abc123");
    });

    it("omits the header entirely when no key is stored", async () => {
      await post("/ask", { question: "hi" });
      expect(lastRequestHeaders().Authorization).toBeUndefined();
    });

    it("never sets Content-Type on an upload, so the boundary survives", async () => {
      setApiKey("sk_abc123");
      await upload("/ingest", new File(["body"], "a.txt"));
      expect(lastRequestHeaders()["Content-Type"]).toBeUndefined();
    });
  });

  describe("error handling", () => {
    it("raises UnauthorizedError on 401", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: "Missing Authorization header." }, 401),
      ) as unknown as typeof fetch;

      await expect(post("/ask", {})).rejects.toBeInstanceOf(UnauthorizedError);
    });

    it("explains a 429 in plain language", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: "Rate limit exceeded" }, 429),
      ) as unknown as typeof fetch;

      await expect(post("/ask", {})).rejects.toThrow(/Rate limit reached/);
    });

    it("surfaces FastAPI's detail string", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: "Unsupported file type '.zip'." }, 400),
      ) as unknown as typeof fetch;

      await expect(post("/ingest", {})).rejects.toThrow(/Unsupported file type/);
    });

    it("flattens FastAPI's validation error array", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: [{ msg: "question cannot be blank" }] }, 422),
      ) as unknown as typeof fetch;

      await expect(post("/ask", {})).rejects.toThrow(/question cannot be blank/);
    });

    it("falls back to the app's own error envelope", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ code: "internal_error", message: "Something broke." }, 500),
      ) as unknown as typeof fetch;

      await expect(post("/ask", {})).rejects.toThrow(/Something broke/);
    });

    it("falls back to the status code for an empty body", async () => {
      globalThis.fetch = vi.fn(
        async () => new Response("", { status: 502 }),
      ) as unknown as typeof fetch;

      await expect(get("/x")).rejects.toThrow(/HTTP 502/);
    });
  });

  describe("fetchHealth", () => {
    it("returns the parsed body", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ status: "ok", version: "0.3.0", auth_required: true, providers: {} }),
      ) as unknown as typeof fetch;

      const health = await fetchHealth();
      expect(health?.auth_required).toBe(true);
    });

    it("returns null instead of throwing when the backend is down", async () => {
      globalThis.fetch = vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }) as unknown as typeof fetch;

      await expect(fetchHealth()).resolves.toBeNull();
    });

    it("returns null on a non-ok response", async () => {
      globalThis.fetch = vi.fn(
        async () => new Response("", { status: 503 }),
      ) as unknown as typeof fetch;

      await expect(fetchHealth()).resolves.toBeNull();
    });
  });
});
