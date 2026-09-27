import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LONG_TIMEOUT_MS, errorMessage, get, post } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import Tooltip from "../components/Tooltip";
import ErrorAlert from "../components/ErrorAlert";
import { formatScore } from "../utils/formatting";
import { useI18n, type MessageKey } from "../i18n";

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

type MetricKey = keyof NonNullable<Run["aggregate"]>;

const METRICS: { key: MetricKey; label: MessageKey; desc: MessageKey }[] = [
  { key: "faithfulness", label: "eval.metric.faithfulness", desc: "eval.metric.faithfulness.desc" },
  { key: "answer_relevance", label: "eval.metric.answer_relevance", desc: "eval.metric.answer_relevance.desc" },
  { key: "context_precision", label: "eval.metric.context_precision", desc: "eval.metric.context_precision.desc" },
  { key: "context_recall", label: "eval.metric.context_recall", desc: "eval.metric.context_recall.desc" },
];

/** Golden datasets shipped in data/golden/. golden_fr_v1 asks French questions of the English corpus. */
const DATASETS: { id: string; label: MessageKey }[] = [
  { id: "golden_v1", label: "eval.dataset.golden_v1" },
  { id: "golden_v2", label: "eval.dataset.golden_v2" },
  { id: "golden_fr_v1", label: "eval.dataset.golden_fr_v1" },
];

const DATE_FORMAT: Intl.DateTimeFormatOptions = {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
};

export default function EvalPage() {
  const { t, tp, lang, locale, formatDate } = useI18n();
  const queryClient = useQueryClient();
  const [dataset, setDataset] = useState(() => (lang === "fr" ? "golden_fr_v1" : "golden_v1"));
  const { data, refetch, error, isLoading } = useQuery({
    queryKey: ["runs"],
    queryFn: () => get<Run[]>("/eval/runs"),
  });

  // An eval answers and judges every golden question inside one request, so it
  // gets the long timeout, and the buttons stay disabled until it returns.
  const run = useMutation({
    mutationFn: (name: string) => post("/eval/run", { dataset: name }, { timeoutMs: LONG_TIMEOUT_MS }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["regressions"] });
    },
  });
  const runNow = (name: string = dataset) => run.mutate(name);

  // A run whose judge never returned a score has nothing to show but four
  // dashes, so it is kept out of the list and reported as a count instead.
  const scored = (data ?? []).filter((r) => r.aggregate && r.n_scored > 0);
  const unscoredCount = (data?.length ?? 0) - scored.length;

  return (
    <div className="flex flex-col gap-6 max-w-6xl">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="display text-4xl font-semibold text-white">{t("eval.title")}</h1>
          <p className="text-sm text-zinc-400 mt-2">{t("eval.subtitle")}</p>
        </div>
        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          {t("eval.dataset")}
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="input text-sm !py-2"
          >
            {DATASETS.map((d) => (
              <option key={d.id} value={d.id}>
                {t(d.label)}
              </option>
            ))}
          </select>
        </label>
      </header>

      {run.isPending && (
        <div className="card p-4">
          <LoadingSpinner message={t("eval.running")} />
        </div>
      )}

      {run.error && (
        <ErrorAlert
          error={t("eval.runFailed", { error: errorMessage(run.error) })}
          onRetry={() => runNow(run.variables ?? dataset)}
          onDismiss={() => run.reset()}
        />
      )}

      {isLoading ? (
        <div className="card p-12">
          <LoadingSpinner message={t("eval.loading")} />
        </div>
      ) : error ? (
        <ErrorAlert
          error={t("eval.loadFailed", { error: errorMessage(error) })}
          onRetry={() => refetch()}
        />
      ) : !scored.length ? (
        <div className="card p-12 text-center">
          <h2 className="text-xl font-semibold text-white mb-2">{t("eval.empty")}</h2>
          <p className="text-sm text-zinc-400 mb-6">
            {unscoredCount > 0 ? tp("eval.emptyUnscored", unscoredCount) : t("eval.emptyHelp")}
          </p>
          <button onClick={() => runNow()} disabled={run.isPending} className="btn-primary min-h-[48px]">
            {run.isPending ? t("common.running") : t("eval.run")}
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex justify-end">
            <button onClick={() => runNow()} disabled={run.isPending} className="btn-primary text-sm px-4 py-2 min-h-[44px] sm:min-h-auto">
              {run.isPending ? t("common.running") : t("eval.run")}
            </button>
          </div>
          {scored.map((r) => {
            const date = r.created_at ? formatDate(r.created_at, DATE_FORMAT) : "";
            return (
              <div key={r.id} className="card p-6">
                <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
                  <div>
                    <h2 className="text-lg font-semibold text-white">{r.dataset}</h2>
                    <p className="text-xs text-zinc-500 mt-1">
                      {tp("eval.scored", r.n, { scored: r.n_scored })}
                      {r.model ? ` • ${r.model}` : ""}
                      {date ? ` • ${date}` : ""}
                    </p>
                  </div>
                  <button onClick={() => runNow(r.dataset)} disabled={run.isPending} className="btn-ghost text-sm px-4 py-2 min-h-[44px] sm:min-h-auto">
                    {run.isPending ? t("common.running") : t("eval.rerun")}
                  </button>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
                  {METRICS.map((m) => {
                    const v = r.aggregate?.[m.key];
                    const pct = v == null ? 0 : Math.round(v * 100);
                    const shown = v == null ? t("common.na") : formatScore(v, 2, locale);

                    return (
                      <Tooltip key={m.key} label={t(m.desc)} side="top">
                        <div className="p-4 rounded-lg bg-bg-surface border border-bg-border cursor-help">
                          <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">{t(m.label)}</div>
                          <div className="text-2xl sm:text-3xl font-bold text-white mt-2 tabular-nums">
                            {shown}
                          </div>
                          <div className="mt-3 h-1.5 bg-bg-elevated rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-accent to-emerald-400 transition-all"
                              style={{ width: `${pct}%` }}
                              role="progressbar"
                              aria-valuenow={pct}
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-label={t("eval.metricAria", { label: t(m.label), value: shown })}
                            />
                          </div>
                          <p className="text-[10px] sm:text-[11px] text-zinc-500 mt-2 hidden sm:block">{t(m.desc)}</p>
                        </div>
                      </Tooltip>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {scored.length > 0 && unscoredCount > 0 && (
        <p className="text-xs text-zinc-500">{tp("eval.hidden", unscoredCount)}</p>
      )}
    </div>
  );
}
