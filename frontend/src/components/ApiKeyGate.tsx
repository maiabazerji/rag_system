import { useEffect, useState } from "react";
import { fetchHealth, getApiKey, setApiKey } from "../api/client";

/**
 * Shows an API key field when, and only when, the backend requires one.
 *
 * The backend reports `auth_required` from GET /health, so a local instance
 * with auth disabled never sees this. When auth is on and no key is stored,
 * the banner explains how to mint one.
 */
export default function ApiKeyGate() {
  const [authRequired, setAuthRequired] = useState<boolean | null>(null);
  const [key, setKeyState] = useState(getApiKey() ?? "");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchHealth().then((h) => {
      if (!cancelled && h) setAuthRequired(h.auth_required);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!saved) return;
    const t = setTimeout(() => setSaved(false), 2000);
    return () => clearTimeout(t);
  }, [saved]);

  if (!authRequired) return null;

  const stored = getApiKey();

  function save(e: React.FormEvent) {
    e.preventDefault();
    setApiKey(key);
    setSaved(true);
  }

  function clear() {
    setApiKey(null);
    setKeyState("");
    setSaved(false);
  }

  return (
    <div className="mb-6 rounded border border-bg-border bg-bg-surface px-4 py-3">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <div className="text-sm font-medium text-zinc-200">
            {stored ? "API key saved" : "This backend requires an API key"}
          </div>
          <p className="text-xs text-zinc-500 mt-1 leading-relaxed max-w-prose">
            {stored ? (
              <>
                Requests are sent with <code className="font-mono">Authorization: Bearer …</code>.
                The key is kept in this browser only.
              </>
            ) : (
              <>
                Create one with{" "}
                <code className="font-mono">
                  python scripts/setup_auth.py --create-key "my-laptop"
                </code>
                , then paste it below.
              </>
            )}
          </p>
        </div>
        {saved && (
          <span className="text-xs text-emerald-400 font-mono shrink-0">saved</span>
        )}
      </div>

      <form onSubmit={save} className="mt-3 flex gap-2 flex-wrap">
        <input
          type="password"
          value={key}
          onChange={(e) => setKeyState(e.target.value)}
          placeholder="sk_…"
          autoComplete="off"
          spellCheck={false}
          aria-label="API key"
          className="flex-1 min-w-[16rem] rounded border border-bg-border bg-bg-base px-3 py-1.5 text-sm font-mono text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={!key.trim()}
          className="rounded border border-bg-border px-3 py-1.5 text-sm text-zinc-200 hover:border-accent disabled:opacity-40 disabled:hover:border-bg-border transition-colors"
        >
          Save key
        </button>
        {stored && (
          <button
            type="button"
            onClick={clear}
            className="rounded border border-transparent px-3 py-1.5 text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            Clear
          </button>
        )}
      </form>
    </div>
  );
}
