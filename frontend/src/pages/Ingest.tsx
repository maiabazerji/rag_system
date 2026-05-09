import { useEffect, useRef, useState } from "react";
import { get, upload } from "../api/client";
import { UploadIcon } from "../components/Icons";

type IngestResult = { doc_id: string; filename: string; chunks: number; indexed_total: number };

export default function Ingest() {
  const [items, setItems] = useState<IngestResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [stats, setStats] = useState<{ indexed_chunks: number } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refreshStats() {
    try {
      setStats(await get<{ indexed_chunks: number }>("/ingest/stats"));
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    refreshStats();
  }, []);

  async function uploadFiles(files: FileList | File[]) {
    setErr(null);
    setBusy(true);
    try {
      for (const f of Array.from(files)) {
        const r = await upload<IngestResult>("/ingest", f);
        setItems((prev) => [r, ...prev]);
      }
      await refreshStats();
    } catch (e: any) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div className="space-y-2">
          <h1 className="display text-4xl font-semibold text-white">Ingest</h1>
          <p className="text-zinc-400 max-w-xl">
            Drop in <span className="font-mono text-zinc-300">.txt</span>,{" "}
            <span className="font-mono text-zinc-300">.md</span>, or{" "}
            <span className="font-mono text-zinc-300">.pdf</span>. They're chunked, embedded,
            and indexed into Qdrant — ready to ground answers on the Ask page.
          </p>
        </div>
        {stats && (
          <div className="chip">
            <span className="text-accent">●</span>
            {stats.indexed_chunks.toLocaleString()} chunks indexed
          </div>
        )}
      </header>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
        }}
        className={`card border-dashed text-center py-14 transition-colors ${
          drag ? "border-accent bg-accent/5" : "hover:border-zinc-600"
        }`}
      >
        <div className="inline-flex w-11 h-11 rounded-lg border border-bg-border text-zinc-400 items-center justify-center mb-4">
          <UploadIcon className="w-5 h-5" />
        </div>
        <h2 className="display text-xl font-semibold text-white">Drop files here</h2>
        <p className="text-zinc-500 text-sm mt-1">or click below to browse</p>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".txt,.md,.pdf"
          className="hidden"
          onChange={(e) => e.target.files && uploadFiles(e.target.files)}
        />
        <button onClick={() => fileRef.current?.click()} disabled={busy} className="btn-primary mt-5">
          {busy ? "Uploading…" : "Choose files"}
        </button>
        {err && <p className="text-rose-400 text-sm mt-4">{err}</p>}
      </div>

      {items.length > 0 && (
        <div>
          <div className="text-xs uppercase tracking-wider text-zinc-500 mb-2">Recently uploaded</div>
          <div className="grid gap-2">
            {items.map((it, i) => (
              <div key={i} className="card !p-4 flex items-center justify-between">
                <div className="min-w-0">
                  <div className="text-zinc-100 font-medium truncate">{it.filename}</div>
                  <div className="text-xs text-zinc-500 font-mono mt-0.5">{it.doc_id}</div>
                </div>
                <div className="chip !text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10 shrink-0">
                  +{it.chunks} chunks
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
