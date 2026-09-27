import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { errorMessage, get, upload } from "../api/client";
import { UploadIcon } from "../components/Icons";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorAlert from "../components/ErrorAlert";
import { useI18n } from "../i18n";

type IngestResult = { doc_id: string; filename: string; chunks: number; indexed_total: number };
type Item =
  | { key: number; ok: true; result: IngestResult }
  | { key: number; ok: false; filename: string; error: string };

/** Must match SUPPORTED_SUFFIXES in backend/app/rag/ingest.py. */
const ACCEPTED_SUFFIXES = [".pdf", ".txt", ".md", ".markdown", ".rst", ".csv", ".json"];

/** "a, b, or c" / "a, b ou c", with each extension styled as code. */
function SuffixList({ locale }: { locale: string }) {
  const parts = new Intl.ListFormat(locale, { type: "disjunction" }).formatToParts(ACCEPTED_SUFFIXES);
  return (
    <>
      {parts.map((p, i) =>
        p.type === "element" ? (
          <span key={i} className="font-mono text-zinc-300">
            {p.value}
          </span>
        ) : (
          <span key={i}>{p.value}</span>
        ),
      )}
    </>
  );
}

export default function Ingest() {
  const { t, tp, rich, locale, formatNumber } = useI18n();
  const [items, setItems] = useState<Item[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const nextKey = useRef(0);
  const { data: stats, refetch: refetchStats } = useQuery({
    queryKey: ["ingest-stats"],
    queryFn: () => get<{ indexed_chunks: number }>("/ingest/stats"),
  });

  /** Upload one at a time; a failed file is reported and the rest still go. */
  async function uploadFiles(files: FileList | File[]) {
    if (busy) return;
    const list = Array.from(files);
    setErr(null);
    setBusy(true);
    let failed = 0;
    try {
      for (const f of list) {
        const key = nextKey.current++;
        try {
          const result = await upload<IngestResult>("/ingest", f);
          setItems((prev) => [{ key, ok: true, result }, ...prev]);
        } catch (e) {
          failed++;
          setItems((prev) => [{ key, ok: false, filename: f.name, error: errorMessage(e) }, ...prev]);
        }
      }
      if (failed > 0) {
        setErr(tp("ingest.failedSome", list.length, { failed }));
      }
      await refetchStats();
    } finally {
      setBusy(false);
      // Let the same file be picked again after a failure.
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div className="space-y-2">
          <h1 className="display text-4xl font-semibold text-white">{t("ingest.title")}</h1>
          <p className="text-zinc-400 max-w-xl">
            {rich("ingest.intro", { types: <SuffixList locale={locale} /> })}
          </p>
        </div>
        {stats && (
          <div className="chip">
            <span className="text-accent">●</span>
            {t("ingest.chunksIndexed", { count: formatNumber(stats.indexed_chunks) })}
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
        {busy ? (
          <LoadingSpinner message={t("ingest.processing")} />
        ) : (
          <>
            <div className="inline-flex w-11 h-11 rounded-lg border border-bg-border text-zinc-400 items-center justify-center mb-4">
              <UploadIcon className="w-5 h-5" aria-hidden="true" />
            </div>
            <h2 className="display text-xl font-semibold text-white">{t("ingest.dropHere")}</h2>
            <p className="text-zinc-500 text-sm mt-1">{t("ingest.orBrowse")}</p>
            <input
              ref={fileRef}
              type="file"
              multiple
              accept={ACCEPTED_SUFFIXES.join(",")}
              className="hidden"
              onChange={(e) => e.target.files && uploadFiles(e.target.files)}
            />
            <button onClick={() => fileRef.current?.click()} disabled={busy} className="btn-primary mt-5 min-h-[48px]">
              {t("ingest.choose")}
            </button>
          </>
        )}
        {err && (
          <div className="mt-4">
            <ErrorAlert
              error={err}
              onRetry={() => {
                setErr(null);
                fileRef.current?.click();
              }}
              onDismiss={() => setErr(null)}
            />
          </div>
        )}
      </div>

      {items.length > 0 && (
        <div>
          <div className="text-xs uppercase tracking-wider text-zinc-500 mb-2">{t("ingest.recent")}</div>
          <div className="grid gap-2">
            {items.map((item) =>
              item.ok ? (
                <UploadedRow key={item.key} it={item.result} />
              ) : (
                <div
                  key={item.key}
                  className="card !p-4 !border-rose-500/40 flex items-center justify-between gap-4 flex-wrap sm:flex-nowrap"
                >
                  <div className="min-w-0 flex-1">
                    <div className="text-zinc-100 font-medium truncate">{item.filename}</div>
                    <div className="text-xs text-rose-300 mt-0.5">{item.error}</div>
                  </div>
                  <div className="chip !text-rose-300 !border-rose-500/40 !bg-rose-500/10 shrink-0">
                    {t("ingest.failed")}
                  </div>
                </div>
              ),
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function UploadedRow({ it }: { it: IngestResult }) {
  const { t, formatNumber } = useI18n();
  return (
    <div className="card !p-4 flex items-center justify-between gap-4 flex-wrap sm:flex-nowrap">
      <div className="min-w-0 flex-1">
        <div className="text-zinc-100 font-medium truncate">{it.filename}</div>
        <div className="text-xs text-zinc-500 font-mono mt-0.5 truncate">{it.doc_id}</div>
      </div>
      <div className="flex items-center gap-2 shrink-0 flex-wrap sm:flex-nowrap">
        <div className="chip !text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10 min-h-[44px] sm:min-h-auto">
          {t("ingest.addedChunks", { count: formatNumber(it.chunks) })}
        </div>
        <div className="chip text-xs min-h-[44px] sm:min-h-auto">
          {t("ingest.total", { count: formatNumber(it.indexed_total) })}
        </div>
      </div>
    </div>
  );
}
