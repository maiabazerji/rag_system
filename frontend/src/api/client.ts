const BASE = "/api";

async function getErrorMessage(r: Response): Promise<string> {
  try {
    const text = await r.text();
    return text || `HTTP ${r.status}`;
  } catch {
    return `HTTP ${r.status}`;
  }
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await getErrorMessage(r));
  return r.json();
}

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(await getErrorMessage(r));
  return r.json();
}

export async function upload<T>(path: string, file: File): Promise<T> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}${path}`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await getErrorMessage(r));
  return r.json();
}
