/**
 * Tests for the API client.
 *
 * This file exists because of a specific regression: the backend gained auth
 * dependencies while these helpers still sent no Authorization header, so every
 * button in the UI returned 401. The header tests below are the guard.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  RequestTimeoutError,
  UnauthorizedError,
  fetchHealth,
  get,
  getAdminKey,
  getApiKey,
  onUnauthorized,
  post,
  setAdminKey,
  setApiKey,
  upload,
} from "./client";
import { setCurrentLanguage } from "../i18n/core";

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

  describe("language", () => {
    afterEach(() => setCurrentLanguage("en"));

    it("sends Accept-Language for the active UI language", async () => {
      setCurrentLanguage("fr");
      await post("/ask", { question: "Que mesure BM25 ?" });
      expect(lastRequestHeaders()["Accept-Language"]).toBe("fr,en;q=0.5");
      await get("/ingest/stats");
      expect(lastRequestHeaders()["Accept-Language"]).toBe("fr,en;q=0.5");
      await upload("/ingest", new File(["body"], "a.txt"));
      expect(lastRequestHeaders()["Accept-Language"]).toBe("fr,en;q=0.5");
    });

    it("words its own errors in the active language", async () => {
      setCurrentLanguage("fr");
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: "slow down" }, 429),
      ) as unknown as typeof fetch;
      await expect(post("/ask", {})).rejects.toThrow(/Limite de requêtes atteinte/);
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

  describe("admin key", () => {
    it("round-trips and clears", () => {
      setAdminKey("  admin-secret ");
      expect(getAdminKey()).toBe("admin-secret");
      setAdminKey(null);
      expect(getAdminKey()).toBeNull();
    });

    it("is sent as X-Admin-Key only when a request asks for it", async () => {
      setApiKey("sk_abc123");
      setAdminKey("admin-secret");

      await post("/graph/build", {}, { admin: true });
      expect(lastRequestHeaders()["X-Admin-Key"]).toBe("admin-secret");
      expect(lastRequestHeaders().Authorization).toBe("Bearer sk_abc123");

      await post("/ask", { question: "hi" });
      expect(lastRequestHeaders()["X-Admin-Key"]).toBeUndefined();
    });

    it("sends no X-Admin-Key when none is stored", async () => {
      await get("/graph/build/abc", { admin: true });
      expect(lastRequestHeaders()["X-Admin-Key"]).toBeUndefined();
    });
  });

  describe("response bodies", () => {
    it("resolves a 204 to undefined instead of failing to parse", async () => {
      globalThis.fetch = vi.fn(
        async () => new Response(null, { status: 204 }),
      ) as unknown as typeof fetch;

      await expect(post("/graph/reset", {})).resolves.toBeUndefined();
    });

    it("resolves an empty 200 body to undefined", async () => {
      globalThis.fetch = vi.fn(
        async () => new Response("", { status: 200 }),
      ) as unknown as typeof fetch;

      await expect(get("/x")).resolves.toBeUndefined();
    });

    it("returns a non-JSON body as text", async () => {
      globalThis.fetch = vi.fn(
        async () =>
          new Response("plain ok", { status: 200, headers: { "Content-Type": "text/plain" } }),
      ) as unknown as typeof fetch;

      await expect(get("/x")).resolves.toBe("plain ok");
    });

    it("still parses JSON sent without a JSON content type", async () => {
      globalThis.fetch = vi.fn(
        async () =>
          new Response('{"a":1}', { status: 200, headers: { "Content-Type": "text/plain" } }),
      ) as unknown as typeof fetch;

      await expect(get<{ a: number }>("/x")).resolves.toEqual({ a: 1 });
    });
  });

  describe("timeouts", () => {
    /** A fetch that never answers on its own but rejects when aborted, like the real one. */
    function hangingFetch() {
      return vi.fn(
        (_url: string, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () =>
              reject(new DOMException("The operation was aborted.", "AbortError")),
            );
          }),
      ) as unknown as typeof fetch;
    }

    it("aborts and raises RequestTimeoutError after timeoutMs", async () => {
      globalThis.fetch = hangingFetch();
      await expect(get("/slow", { timeoutMs: 20 })).rejects.toBeInstanceOf(RequestTimeoutError);
    });

    it("passes an abort signal to fetch", async () => {
      await get("/x");
      const init = vi.mocked(globalThis.fetch).mock.calls.at(-1)?.[1];
      expect(init?.signal).toBeInstanceOf(AbortSignal);
    });

    it("reports an unreachable backend in plain language", async () => {
      globalThis.fetch = vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }) as unknown as typeof fetch;

      await expect(get("/x")).rejects.toThrow(/Could not reach the backend/);
    });
  });

  describe("error handling", () => {
    it("raises UnauthorizedError on 401", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: "Missing Authorization header." }, 401),
      ) as unknown as typeof fetch;

      await expect(post("/ask", {})).rejects.toBeInstanceOf(UnauthorizedError);
    });

    it("tells onUnauthorized listeners about a 401, until they unsubscribe", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: "Invalid API key." }, 401),
      ) as unknown as typeof fetch;
      const listener = vi.fn();
      const unsubscribe = onUnauthorized(listener);

      await expect(get("/x")).rejects.toBeInstanceOf(UnauthorizedError);
      expect(listener).toHaveBeenCalledTimes(1);
      expect(listener.mock.calls[0][0].message).toBe("Invalid API key.");

      unsubscribe();
      await expect(get("/x")).rejects.toBeInstanceOf(UnauthorizedError);
      expect(listener).toHaveBeenCalledTimes(1);
    });

    it("does not notify listeners for other errors", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: "nope" }, 500),
      ) as unknown as typeof fetch;
      const listener = vi.fn();
      const unsubscribe = onUnauthorized(listener);

      await expect(get("/x")).rejects.toThrow(/nope/);
      expect(listener).not.toHaveBeenCalled();
      unsubscribe();
    });

    it("explains a 403 as needing the admin key, keeping the detail", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: "Invalid admin key" }, 403),
      ) as unknown as typeof fetch;

      const err = await post("/graph/build", {}).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(403);
      expect(apiErr.detail).toBe("Invalid admin key");
      expect(apiErr.message).toMatch(/Invalid admin key/);
      expect(apiErr.message).toMatch(/ADMIN_KEY/);
    });

    it("explains a 503 as a service outage", async () => {
      globalThis.fetch = vi.fn(async () =>
        jsonResponse({ detail: "Vector store is unavailable." }, 503),
      ) as unknown as typeof fetch;

      const err = await get("/ingest/stats").catch((e: unknown) => e);
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).status).toBe(503);
      expect((err as ApiError).message).toMatch(/Service unavailable: Vector store is unavailable/);
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
