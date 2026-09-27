import { acceptLanguage, getLanguage, translate, type MessageKey, type Vars } from "../i18n/core";

const BASE = "/api";

/** Translate into the active UI language (this module lives outside React). */
function tr(key: MessageKey, vars?: Vars): string {
  return translate(getLanguage(), key, vars);
}
const API_KEY_STORAGE = "evalrag.apiKey";
const ADMIN_KEY_STORAGE = "evalrag.adminKey";

/** Default ceiling for one request. Compare runs three strategies in a row. */
export const DEFAULT_TIMEOUT_MS = 3 * 60_000;
/** Eval runs answer and judge a whole golden dataset in one request. */
export const LONG_TIMEOUT_MS = 30 * 60_000;
/** Uploads are chunked and embedded inside the request. */
export const UPLOAD_TIMEOUT_MS = 10 * 60_000;
const HEALTH_TIMEOUT_MS = 5_000;

/**
 * A non-2xx response from the backend. `status` is the HTTP status code and
 * `detail` the backend's own wording; `message` may add advice for a person.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  constructor(status: number, message: string, detail = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/** Raised when the backend rejects a request for lack of a valid API key. */
export class UnauthorizedError extends ApiError {
  constructor(message = tr("client.unauthorized")) {
    super(401, message, message);
    this.name = "UnauthorizedError";
  }
}

/** Raised when a request is aborted because it took longer than its timeout. */
export class RequestTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(tr("client.timeout", { seconds: Math.round(timeoutMs / 1000) }));
    this.name = "RequestTimeoutError";
  }
}

export type RequestOptions = {
  /** Abort after this many milliseconds. Defaults to DEFAULT_TIMEOUT_MS. */
  timeoutMs?: number;
  /** Also send the saved admin key as X-Admin-Key (graph build and reset). */
  admin?: boolean;
};

type UnauthorizedListener = (error: UnauthorizedError) => void;
const unauthorizedListeners = new Set<UnauthorizedListener>();

/**
 * Be told whenever a request comes back 401, so the UI can ask for a key even
 * if /health said none was needed (or could not be reached). Returns an
 * unsubscribe function.
 */
export function onUnauthorized(listener: UnauthorizedListener): () => void {
  unauthorizedListeners.add(listener);
  return () => {
    unauthorizedListeners.delete(listener);
  };
}

/** Read the saved API key. Returns null when none is stored. */
export function getApiKey(): string | null {
  try {
    return localStorage.getItem(API_KEY_STORAGE);
  } catch {
    return null; // private browsing, or storage blocked
  }
}

/** Save an API key, or clear it when given an empty value. */
export function setApiKey(key: string | null): void {
  try {
    if (key && key.trim()) localStorage.setItem(API_KEY_STORAGE, key.trim());
    else localStorage.removeItem(API_KEY_STORAGE);
  } catch {
    /* storage unavailable; the key simply will not persist */
  }
}

/** Read the saved admin key (ADMIN_KEY on the backend). Null when none is stored. */
export function getAdminKey(): string | null {
  try {
    return localStorage.getItem(ADMIN_KEY_STORAGE);
  } catch {
    return null;
  }
}

/** Save the admin key, or clear it when given an empty value. */
export function setAdminKey(key: string | null): void {
  try {
    if (key && key.trim()) localStorage.setItem(ADMIN_KEY_STORAGE, key.trim());
    else localStorage.removeItem(ADMIN_KEY_STORAGE);
  } catch {
    /* storage unavailable; the key simply will not persist */
  }
}

function requestHeaders(admin = false): Record<string, string> {
  // Lets the backend word its own messages (e.g. refusals) in the UI language.
  const headers: Record<string, string> = { "Accept-Language": acceptLanguage(getLanguage()) };
  const key = getApiKey();
  if (key) headers.Authorization = `Bearer ${key}`;
  if (admin) {
    const adminKey = getAdminKey();
    if (adminKey) headers["X-Admin-Key"] = adminKey;
  }
  return headers;
}

/**
 * Pull a readable message out of a failed response body.
 *
 * FastAPI returns `{detail}` for HTTPException and `{message, detail}` for the
 * app's own handlers; plain text is the last resort. Returns "" for no body.
 */
async function getErrorDetail(r: Response): Promise<string> {
  try {
    const text = await r.text();
    if (!text) return "";
    try {
      const body = JSON.parse(text);
      if (typeof body?.detail === "string") return body.detail;
      if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
        return body.detail.map((d: { msg: string }) => d.msg).join("; ");
      }
      if (typeof body?.message === "string") return body.message;
      if (typeof body?.error === "string") return body.error;
    } catch {
      /* not JSON */
    }
    return text;
  } catch {
    return "";
  }
}

/** Turn an error status into an error a person can act on. */
async function toError(r: Response): Promise<Error> {
  const detail = await getErrorDetail(r);
  switch (r.status) {
    case 401: {
      const err = new UnauthorizedError(detail || undefined);
      unauthorizedListeners.forEach((l) => l(err));
      return err;
    }
    case 403:
      // e.g. graph build/reset, which need the admin key as well as an API key.
      return new ApiError(
        403,
        tr("client.forbidden", { detail: detail ? `: ${detail}` : "" }),
        detail,
      );
    case 429:
      return new ApiError(429, tr("client.rateLimited"), detail);
    case 503:
      return new ApiError(
        503,
        tr("client.unavailable", { detail: detail || `HTTP ${r.status}` }),
        detail,
      );
    default:
      return new ApiError(r.status, detail || `HTTP ${r.status}`, detail);
  }
}

/**
 * Parse a successful response. 204 and empty bodies resolve to undefined;
 * non-JSON bodies resolve to their text rather than throwing a parse error.
 */
async function parseBody<T>(r: Response): Promise<T> {
  if (r.status === 204 || r.status === 205) return undefined as T;
  const text = await r.text();
  if (!text) return undefined as T;
  const type = r.headers.get("Content-Type") ?? "";
  if (type.includes("json")) return JSON.parse(text) as T;
  try {
    return JSON.parse(text) as T;
  } catch {
    return text as T;
  }
}

async function request<T>(path: string, init: RequestInit, opts: RequestOptions = {}): Promise<T> {
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let r: Response;
  try {
    r = await fetch(`${BASE}${path}`, { ...init, signal: controller.signal });
  } catch (e) {
    if (controller.signal.aborted) throw new RequestTimeoutError(timeoutMs);
    throw new Error(
      tr("client.unreachable", { reason: e instanceof Error ? e.message : String(e) }),
      { cause: e },
    );
  } finally {
    clearTimeout(timer);
  }
  if (!r.ok) throw await toError(r);
  return parseBody<T>(r);
}

export async function post<T>(path: string, body: unknown, opts?: RequestOptions): Promise<T> {
  return request<T>(
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", ...requestHeaders(opts?.admin) },
      body: JSON.stringify(body),
    },
    opts,
  );
}

export async function get<T>(path: string, opts?: RequestOptions): Promise<T> {
  return request<T>(path, { headers: requestHeaders(opts?.admin) }, opts);
}

export async function upload<T>(path: string, file: File, opts?: RequestOptions): Promise<T> {
  const fd = new FormData();
  fd.append("file", file);
  // Content-Type is intentionally unset: the browser adds the multipart boundary.
  return request<T>(
    path,
    { method: "POST", headers: requestHeaders(), body: fd },
    { timeoutMs: UPLOAD_TIMEOUT_MS, ...opts },
  );
}

/** Turn anything thrown into a message fit for an ErrorAlert. */
export function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

export interface HealthResponse {
  status: string;
  version: string;
  auth_required: boolean;
  providers: Record<string, boolean>;
}

/** Fetch backend health. Never throws: an unreachable backend returns null. */
export async function fetchHealth(): Promise<HealthResponse | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
  try {
    const r = await fetch(`${BASE}/health`, { signal: controller.signal });
    return r.ok ? ((await r.json()) as HealthResponse) : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}
