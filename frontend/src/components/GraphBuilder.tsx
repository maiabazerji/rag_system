/**
 * Knowledge-graph panel for the Ingest page: shows how many triples are indexed
 * and builds or rebuilds the graph. A build is a background job on the backend,
 * so this starts it and polls until it finishes. Build needs the admin key.
 */
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, errorMessage, get, getAdminKey, post, setAdminKey } from "../api/client";
import ErrorAlert from "./ErrorAlert";
import Tooltip from "./Tooltip";
import { useI18n, type I18n, type MessageKey } from "../i18n";

export type GraphStats = { triples: number; entities: number };
type JobStatus = "queued" | "running" | "completed" | "failed";
/** POST /graph/build answers 202 with a job; GET /graph/build/{id} reports on it. */
type BuildJob = {
  id?: string;
  job_id?: string;
  status: JobStatus;
  total?: number;
  completed?: number;
  failures?: string[];
  error?: string;
};

const POLL_INTERVAL_MS = 2000;

const JOB_STATUS: Record<JobStatus, MessageKey> = {
  queued: "graph.status.queued",
  running: "graph.status.running",
  completed: "graph.status.completed",
  failed: "graph.status.failed",
};

/** Graph build and reset need the admin key on top of any API key. */
function graphBuildError(e: unknown, t: I18n["t"]): string {
  const detail = e instanceof ApiError && e.detail ? ` (${e.detail})` : "";
  if (e instanceof ApiError && e.status === 403) {
    return t("graph.adminNeeded", { detail });
  }
  if (e instanceof ApiError && e.status === 503) {
    return t("graph.buildUnavailable", { detail });
  }
  return errorMessage(e);
}
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Triple and entity counts, shared by the Ingest panel and the Compare hint. */
export function useGraphStats() {
  return useQuery({
    queryKey: ["graph-stats"],
    queryFn: () => get<GraphStats>("/graph/stats"),
  });
}

export default function GraphBuilder() {
  const { t, tp } = useI18n();
  const { data: stats = null, refetch: refetchStats } = useGraphStats();
  const [building, setBuilding] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showAdminKey, setShowAdminKey] = useState(false);

  // Stops the build poller when the page unmounts.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  /** Start a build, then poll its job until it finishes. It can run for minutes. */
  async function build() {
    if (building) return;
    setErr(null);
    setBuilding(true);
    setProgress(t(JOB_STATUS.queued));
    try {
      const started = await post<BuildJob>("/graph/build", {}, { admin: true });
      const jobId = started.job_id ?? started.id;
      if (!jobId) throw new Error(t("graph.noJobId"));
      let job = started;
      while (mounted.current && job.status !== "completed" && job.status !== "failed") {
        await sleep(POLL_INTERVAL_MS);
        if (!mounted.current) return;
        job = await get<BuildJob>(`/graph/build/${encodeURIComponent(jobId)}`, { admin: true });
        setProgress(
          job.total
            ? t("graph.progressChunks", { done: job.completed ?? 0, total: job.total })
            : t(JOB_STATUS[job.status] ?? "graph.status.running"),
        );
      }
      if (!mounted.current) return;
      if (job.status === "failed") {
        setErr(t("graph.buildFailed", { error: job.error ?? t("graph.unknownError") }));
      } else if (job.failures && job.failures.length > 0) {
        setErr(tp("graph.partialFailures", job.failures.length));
      }
      await refetchStats();
    } catch (e) {
      if (!mounted.current) return;
      setErr(graphBuildError(e, t));
      if (e instanceof ApiError && e.status === 403) setShowAdminKey(true);
    } finally {
      if (mounted.current) {
        setBuilding(false);
        setProgress(null);
      }
    }
  }

  return (
    <section className="card !p-5 flex flex-col gap-3" aria-labelledby="graph-title">
      <div>
        <h2 id="graph-title" className="text-lg font-semibold text-white">{t("graph.title")}</h2>
        <p className="text-sm text-zinc-400 mt-1 max-w-2xl">{t("graph.intro")}</p>
      </div>
      <GraphStatus stats={stats} busy={building} progress={progress} onBuild={build} />
      {err && <ErrorAlert error={err} onRetry={build} onDismiss={() => setErr(null)} />}
      <AdminKeyField open={showAdminKey} onToggle={setShowAdminKey} />
    </section>
  );
}

/** Optional admin key, sent as X-Admin-Key on graph build. Kept in this browser only. */
function AdminKeyField({ open, onToggle }: { open: boolean; onToggle: (open: boolean) => void }) {
  const { t, rich } = useI18n();
  const [value, setValue] = useState(getAdminKey() ?? "");
  const [stored, setStored] = useState(!!getAdminKey());

  function save(e: React.FormEvent) {
    e.preventDefault();
    setAdminKey(value);
    setStored(!!value.trim());
  }

  function clear() {
    setAdminKey(null);
    setValue("");
    setStored(false);
  }

  return (
    <details
      open={open}
      onToggle={(e) => onToggle((e.currentTarget as HTMLDetailsElement).open)}
      className="text-xs text-zinc-500"
    >
      <summary className="cursor-pointer select-none hover:text-zinc-300">
        {t("graph.adminKey")} {stored ? t("graph.adminSaved") : t("graph.adminNeededHint")}
      </summary>
      <form onSubmit={save} className="mt-2 flex gap-2 flex-wrap items-center">
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={t("graph.adminPlaceholder")}
          autoComplete="off"
          spellCheck={false}
          aria-label={t("graph.adminKey")}
          className="flex-1 min-w-[14rem] rounded border border-bg-border bg-bg-base px-3 py-1.5 text-sm font-mono text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={!value.trim()}
          className="rounded border border-bg-border px-3 py-1.5 text-sm text-zinc-200 hover:border-accent disabled:opacity-40 transition-colors"
        >
          {t("common.save")}
        </button>
        {stored && (
          <button
            type="button"
            onClick={clear}
            className="rounded px-3 py-1.5 text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {t("common.clear")}
          </button>
        )}
        <span className="basis-full text-[11px] text-zinc-600">
          {rich("graph.adminHelp", { header: <code className="font-mono">X-Admin-Key</code> })}
        </span>
      </form>
    </details>
  );
}

function GraphStatus({
  stats,
  busy,
  progress,
  onBuild,
}: {
  stats: GraphStats | null;
  busy: boolean;
  progress: string | null;
  onBuild: () => void;
}) {
  const { t, rich, formatNumber } = useI18n();
  const ready = !!stats && stats.triples > 0;
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <Tooltip label={t("graph.graphTip")} side="top">
        <div className={`text-xs font-mono px-2.5 py-1.5 rounded border flex items-center gap-2 cursor-help ${
          ready
            ? "bg-violet-500/10 border-violet-500/40 text-violet-300"
            : "bg-amber-500/10 border-amber-500/40 text-amber-300"
        }`}>
          <span className={`w-2 h-2 rounded-full ${ready ? "bg-violet-400" : "bg-amber-400"}`} aria-hidden="true" />
          {stats ? (
            <span>
              {rich("graph.triples", {
                count: <span className="font-semibold">{formatNumber(stats.triples)}</span>,
              })}
            </span>
          ) : (
            <span>{t("graph.graphNotBuilt")}</span>
          )}
        </div>
      </Tooltip>
      <button
        type="button"
        onClick={onBuild}
        disabled={busy}
        className={`text-xs px-3 py-1.5 rounded font-medium transition-colors min-h-[44px] flex items-center ${
          busy
            ? "bg-zinc-700/40 text-zinc-500 cursor-not-allowed"
            : "bg-violet-500/20 border border-violet-500/40 text-violet-300 hover:bg-violet-500/30 hover:border-violet-500/60"
        }`}
        aria-label={t("graph.buildAria")}
      >
        {busy
          ? t("graph.building", { progress: progress ?? "" }).trim()
          : ready
            ? t("graph.rebuild")
            : t("graph.buildGraph")}
      </button>
    </div>
  );
}
