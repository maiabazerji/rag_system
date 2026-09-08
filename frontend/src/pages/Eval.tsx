import { useQuery } from "@tanstack/react-query";
import { get, post } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import Tooltip from "../components/Tooltip";
import ErrorAlert from "../components/ErrorAlert";

type Run = {
  id: string;
  dataset: string;
  model?: string | null;
  n: number;
  n_scored: number;
  n_unscored: number;
  created_at?: string | null;
  aggregate?: { faithfulness?: number; answer_relevance?: number; context_precision?: number; context_recall?: number } | null;
};

const METRICS: { key: keyof NonNullable<Run["aggregate"]>; label: string; desc: string }[] = [
  { key: "faithfulness", label: "Faithfulness", desc: "Answer only asserts what's in retrieved context" },
  { key: "answer_relevance", label: "Relevance", desc: "Answer addresses the question" },
  { key: "context_precision", label: "Precision", desc: "Retrieved chunks are on-topic" },
  { key: "context_recall", label: "Recall", desc: "Retrieved all relevant chunks" },
];

/** "2026-09-05T10:53:40+00:00" -> "5 Sep 2026, 10:53". */
function formatRunDate(iso?: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? null
    : d.toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export default function EvalPage() {
  const { data, refetch, isFetching, error, isLoading } = useQuery({
    queryKey: ["runs"],
    queryFn: () => get<Run[]>("/eval/runs"),
  });

  async function runNow() {
    try {
      await post("/eval/run", { dataset: "golden_v1" });
      refetch();
    } catch (e) {
      // Error will be handled by ErrorAlert
    }
  }

  // A run whose judge never returned a score has nothing to show but four
  // dashes, so it is kept out of the list and reported as a count instead.
  const scored = (data ?? []).filter((r) => r.aggregate && r.n_scored > 0);
  const unscoredCount = (data?.length ?? 0) - scored.length;

  return (
    <div className="flex flex-col gap-6 max-w-6xl">
      <header>
        <h1 className="display text-4xl font-semibold text-white">Evaluation</h1>
        <p className="text-sm text-zinc-400 mt-2">
          Score a golden dataset on faithfulness, relevance, and retrieval precision and recall.
        </p>
      </header>

      {isLoading ? (
        <div className="card p-12">
          <LoadingSpinner message="Loading evaluation results..." />
        </div>
      ) : error ? (
        <ErrorAlert
          error={`Failed to load evaluations: ${error instanceof Error ? error.message : "Unknown error"}`}
          onRetry={() => refetch()}
        />
      ) : !scored.length ? (
        <div className="card p-12 text-center">
          <h2 className="text-xl font-semibold text-white mb-2">No scored evaluation runs yet</h2>
          <p className="text-sm text-zinc-400 mb-6">
            {unscoredCount > 0
              ? `${unscoredCount} earlier ${unscoredCount === 1 ? "run" : "runs"} produced no scores. Run an evaluation to measure retrieval strategy performance.`
              : "Run an evaluation on the golden dataset to measure retrieval strategy performance."}
          </p>
          <button onClick={runNow} disabled={isFetching} className="btn-primary min-h-[48px]">
            {isFetching ? "Running…" : "Run Evaluation"}
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          {scored.map((r) => (
            <div key={r.id} className="card p-6">
              <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-white">{r.dataset}</h2>
                  <p className="text-xs text-zinc-500 mt-1">
                    {r.n_scored} of {r.n} {r.n === 1 ? "question" : "questions"} scored
                    {r.model ? ` • ${r.model}` : ""}
                    {formatRunDate(r.created_at) ? ` • ${formatRunDate(r.created_at)}` : ""}
                  </p>
                </div>
                <button onClick={runNow} disabled={isFetching} className="btn-primary text-sm px-4 py-2 min-h-[44px] sm:min-h-auto">
                  {isFetching ? "Running…" : "Re-run"}
                </button>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
                {METRICS.map((m) => {
                  const v = r.aggregate?.[m.key];
                  const pct = v == null ? 0 : Math.round(v * 100);

                  return (
                    <Tooltip key={m.key} label={m.desc} side="top">
                      <div className="p-4 rounded-lg bg-bg-surface border border-bg-border cursor-help">
                        <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">{m.label}</div>
                        <div className="text-2xl sm:text-3xl font-bold text-white mt-2 tabular-nums">
                          {v == null ? "n/a" : v.toFixed(2)}
                        </div>
                        <div className="mt-3 h-1.5 bg-bg-elevated rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-accent to-emerald-400 transition-all"
                            style={{ width: `${pct}%` }}
                            role="progressbar"
                            aria-valuenow={pct}
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-label={`${m.label}: ${v?.toFixed(2) || "N/A"}`}
                          />
                        </div>
                        <p className="text-[10px] sm:text-[11px] text-zinc-500 mt-2 hidden sm:block">{m.desc}</p>
                      </div>
                    </Tooltip>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {scored.length > 0 && unscoredCount > 0 && (
        <p className="text-xs text-zinc-500">
          {unscoredCount} earlier {unscoredCount === 1 ? "run is" : "runs are"} hidden because the judge returned no scores.
        </p>
      )}
    </div>
  );
}
