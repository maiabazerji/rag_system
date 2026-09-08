const BASE = "/api";
const API_KEY_STORAGE = "evalrag.apiKey";

/** Raised when the backend rejects a request for lack of a valid API key. */
export class UnauthorizedError extends Error {
  constructor(message = "This backend requires an API key.") {
    super(message);
    this.name = "UnauthorizedError";
  }
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

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { Authorization: `Bearer ${key}` } : {};
}

/**
 * Turn a failed response into a readable message.
 *
 * FastAPI returns `{detail}` for HTTPException and `{message, detail}` for the
 * app's own handlers; plain text is the last resort.
 */
async function getErrorMessage(r: Response): Promise<string> {
  try {
    const text = await r.text();
    if (!text) return `HTTP ${r.status}`;
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
    return `HTTP ${r.status}`;
  }
}

async function handle<T>(r: Response): Promise<T> {
  if (r.status === 401) throw new UnauthorizedError(await getErrorMessage(r));
  if (r.status === 429) {
    throw new Error("Rate limit reached. Wait a moment and try again.");
  }
  if (!r.ok) throw new Error(await getErrorMessage(r));
  return r.json() as Promise<T>;
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  return handle<T>(r);
}

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { headers: authHeaders() });
  return handle<T>(r);
}

export async function upload<T>(path: string, file: File): Promise<T> {
  const fd = new FormData();
  fd.append("file", file);
  // Content-Type is intentionally unset: the browser adds the multipart boundary.
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: authHeaders(),
    body: fd,
  });
  return handle<T>(r);
}

export interface HealthResponse {
  status: string;
  version: string;
  auth_required: boolean;
  providers: Record<string, boolean>;
}

/** Fetch backend health. Never throws: an unreachable backend returns null. */
export async function fetchHealth(): Promise<HealthResponse | null> {
  try {
    const r = await fetch(`${BASE}/health`);
    return r.ok ? ((await r.json()) as HealthResponse) : null;
  } catch {
    return null;
  }
}
