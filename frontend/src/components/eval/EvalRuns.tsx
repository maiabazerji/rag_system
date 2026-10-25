/**
 * Runs tab of the Evaluation page, laid out like a chat history: the runs are
 * listed on the side (newest first) with a "New run" entry on top, and the
 * selected run shows its counts, measured aggregates and, on demand, its
 * regression report against the configuration's baseline.
 */
import { useId, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { errorMessage } from "../../api/client";
import {
  STRATEGY_NAMES,
  getRunRegression,
  listEvalRuns,
  listGoldenDatasets,
  runEvaluation,
  type EvalRunRequest,
  type EvalRunSummary,
  type StrategyName,
} from "../../api/eval";
import ErrorAlert from "../ErrorAlert";
import LoadingSpinner from "../LoadingSpinner";
import { AggCell, RegressionBadge, StrategyKey } from "./shared";
import {
  DEMO_CORPUS_DATASETS,
  formatCheckDelta,
  formatCheckValue,
  newestFirst,
  runMetrics,
} from "../../utils/evalRuns";
import { useI18n } from "../../i18n";

/** Selection value that pins the launch form instead of a run. */
const NEW_RUN = "new";

const DATE_FORMAT: Intl.DateTimeFormatOptions = {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
};

export default function EvalRuns() {
  const { t, lang } = useI18n();
  const queryClient = useQueryClient();
  const ids = { dataset: useId(), strategy: useId(), note: useId() };

  const golden = useQuery({ queryKey: ["golden-datasets"], queryFn: listGoldenDatasets });
  const datasets = golden.data ?? [];
  const [picked, setPicked] = useState("");
  const preferred = lang === "fr" ? "golden_fr_v1" : "golden_v1";
  const dataset =
    picked && datasets.includes(picked) ? picked : datasets.includes(preferred) ? preferred : (datasets[0] ?? "");
  const [strategy, setStrategy] = useState<StrategyName>("classic");
  const needsDemoCorpus = DEMO_CORPUS_DATASETS.has(dataset);

  const runs = useQuery({ queryKey: ["runs"], queryFn: listEvalRuns });

  // null follows the newest run (or the launch form when there is none); NEW_RUN pins the form.
  const [selected, setSelected] = useState<string | null>(null);

  const run = useMutation({
    mutationFn: runEvaluation,
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["regression"] });
      setSelected(created?.id ?? null);
    },
  });
  const launch = (req: EvalRunRequest) => run.mutate(req);

  const history = newestFirst(runs.data ?? []);
  const current =
    selected === NEW_RUN ? null : (history.find((r) => r.id === selected) ?? (selected ? null : history[0]) ?? null);
  const showForm = !current;

  return (
    <div className="grid gap-4 md:grid-cols-[15rem_minmax(0,1fr)] items-start">
      <aside
        className="card !p-2 flex flex-col gap-1 md:sticky md:top-6 md:max-h-[calc(100vh-4rem)] overflow-y-auto"
        aria-labelledby="eval-history-title"
      >
        <button
          type="button"
          onClick={() => setSelected(NEW_RUN)}
          aria-current={showForm ? "page" : undefined}
          className={`text-left text-sm rounded-md px-3 py-2 min-h-[40px] border transition-colors ${
            showForm
              ? "border-accent/60 text-white bg-accent/10"
              : "border-bg-border text-zinc-300 hover:text-white hover:border-accent/40"
          }`}
        >
          + {t("eval.newRun")}
        </button>
        <div className="flex items-baseline justify-between gap-2 px-2 pt-3 pb-1">
          <h2 id="eval-history-title" className="text-xs uppercase tracking-wider text-zinc-500">{t("eval.history")}</h2>
          <span className="text-[10px] text-zinc-600">{t("eval.historyOrder")}</span>
        </div>

        {runs.isLoading ? (
          <LoadingSpinner message={t("eval.loading")} />
        ) : runs.error ? (
          <ErrorAlert
            error={t("eval.loadFailed", { error: errorMessage(runs.error) })}
            onRetry={() => runs.refetch()}
          />
        ) : history.length === 0 ? (
          <p className="text-xs text-zinc-500 px-2 py-2">{t("eval.empty")}</p>
        ) : (
          <ul className="flex flex-col gap-0.5">
            {history.map((r) => (
              <li key={r.id}>
                <HistoryItem run={r} active={current?.id === r.id} onSelect={() => setSelected(r.id)} />
              </li>
            ))}
          </ul>
        )}
      </aside>

      <div className="min-w-0">
        {current ? (
          <RunCard run={current} busy={run.isPending} onRerun={launch} />
        ) : (
          <section className="card !p-5 flex flex-col gap-4" aria-labelledby="eval-launch-title">
            <div>
              <h2 id="eval-launch-title" className="text-lg font-semibold text-white">{t("eval.launch")}</h2>
              <p className="text-sm text-zinc-400 mt-1">{t("eval.launchHelp")}</p>
            </div>

            {golden.error ? (
              <ErrorAlert
                error={t("eval.datasetsFailed", { error: errorMessage(golden.error) })}
                onRetry={() => golden.refetch()}
              />
            ) : golden.isLoading ? (
              <LoadingSpinner message={t("eval.datasetsLoading")} />
            ) : datasets.length === 0 ? (
              <p className="text-sm text-zinc-400">{t("eval.datasetsEmpty")}</p>
            ) : (
              <div className="flex flex-wrap items-end gap-3">
                <label htmlFor={ids.dataset} className="flex flex-col gap-1 text-xs text-zinc-400">
                  {t("eval.dataset")}
                  <select
                    id={ids.dataset}
                    value={dataset}
                    onChange={(e) => setPicked(e.target.value)}
                    aria-describedby={needsDemoCorpus ? ids.note : undefined}
                    className="input !py-2 text-sm"
                  >
                    {datasets.map((d) => (
                      <option key={d} value={d}>{d}</option>
                    ))}
                  </select>
                </label>
                <label htmlFor={ids.strategy} className="flex flex-col gap-1 text-xs text-zinc-400">
                  {t("eval.strategy")}
                  <select
                    id={ids.strategy}
                    value={strategy}
                    onChange={(e) => setStrategy(e.target.value as StrategyName)}
                    className="input !py-2 text-sm"
                  >
                    {STRATEGY_NAMES.map((s) => (
                      <option key={s} value={s}>{t(`strategy.${s}`)}</option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  onClick={() => launch({ dataset, strategy })}
                  disabled={run.isPending || !dataset}
                  className="btn-primary text-sm px-4 py-2 min-h-[44px]"
                >
                  {run.isPending ? t("common.running") : t("eval.run")}
                </button>
              </div>
            )}

            {needsDemoCorpus && (
              <p id={ids.note} role="note" className="text-xs text-amber-200 border-l-2 border-amber-500/50 pl-3">
                {t("eval.demoCorpusNote", { dataset })}
              </p>
            )}

            {history.length === 0 && runs.isSuccess && (
              <p className="text-xs text-zinc-500">{t("eval.emptyHelp")}</p>
            )}
          </section>
        )}

        {run.isPending && <LoadingSpinner message={t("eval.running")} />}
        {run.error && (
          <ErrorAlert
            error={t("eval.runFailed", { error: errorMessage(run.error) })}
            onRetry={() => run.variables && launch(run.variables)}
            onDismiss={() => run.reset()}
          />
        )}
      </div>
    </div>
  );
}

/** One entry of the side list: dataset, strategy and date, like a chat history. */
function HistoryItem({ run, active, onSelect }: { run: EvalRunSummary; active: boolean; onSelect: () => void }) {
  const { formatDate } = useI18n();
  const date = run.created_at ? formatDate(run.created_at, DATE_FORMAT) : "";
  const strategy = run.strategy ?? "?";
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={active ? "page" : undefined}
      className={`w-full text-left rounded-md px-3 py-2 flex flex-col gap-1 transition-colors ${
        active ? "bg-bg-elevated text-white" : "text-zinc-300 hover:bg-bg-elevated/60 hover:text-white"
      }`}
    >
      <span className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium truncate">{run.dataset ?? "?"}</span>
        <RegressionBadge status={run.regression_status} tip={null} />
      </span>
      <span className="flex items-center justify-between gap-2 text-[11px] text-zinc-500">
        <StrategyKey strategy={strategy} label={strategy} />
        {date && <time dateTime={run.created_at ?? undefined}>{date}</time>}
      </span>
    </button>
  );
}

function RunCard({
  run,
  busy,
  onRerun,
}: {
  run: EvalRunSummary;
  busy: boolean;
  onRerun: (req: EvalRunRequest) => void;
}) {
  const { t, formatDate, formatNumber } = useI18n();
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const metrics = runMetrics(run);
  const date = run.created_at ? formatDate(run.created_at, DATE_FORMAT) : "";
  const strategy = run.strategy ?? "?";
  const count = (v: number | null | undefined) => (v == null ? t("common.na") : formatNumber(v));
  const rerunnable = !!run.dataset && (STRATEGY_NAMES as readonly string[]).includes(strategy);

  return (
    <article className="card !p-5 flex flex-col gap-4" aria-label={`${run.dataset ?? "?"} · ${strategy}${date ? ` · ${date}` : ""}`}>
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-white break-words">{run.dataset ?? "?"}</h3>
          <div className="text-xs text-zinc-400 mt-1 flex flex-wrap gap-x-3 gap-y-1">
            <StrategyKey strategy={strategy} label={strategy} />
            <span>{t("eval.model", { model: run.model ?? t("eval.defaultModel") })}</span>
            {date && <time dateTime={run.created_at ?? undefined}>{date}</time>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <RegressionBadge status={run.regression_status} tip="eval.regTip" />
          {rerunnable && (
            <button
              type="button"
              onClick={() => onRerun({ dataset: run.dataset!, strategy: strategy as StrategyName })}
              disabled={busy}
              className="btn-ghost text-sm px-3 py-1.5 min-h-[40px]"
            >
              {busy ? t("common.running") : t("eval.rerun")}
            </button>
          )}
        </div>
      </div>

      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
        <Count label={t("eval.count.scored")} value={`${count(run.n_scored)} / ${count(run.n)}`} />
        <Count label={t("eval.count.judgeFailed")} value={count(run.n_judge_failed)} />
        <Count label={t("eval.count.unscored")} value={count(run.n_unscored)} />
        <Count label={t("eval.count.generationFailed")} value={count(run.n_generation_failed)} />
      </dl>

      {metrics.length === 0 ? (
        <p className="text-xs text-zinc-500">{t("eval.noScores")}</p>
      ) : (
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {metrics.map((m) => (
            <div key={m.id} className="rounded-md bg-bg-surface border border-bg-border px-3 py-2">
              <dt className="text-[11px] text-zinc-500 uppercase tracking-wide">{t(m.label)}</dt>
              <dd className="text-sm text-white mt-0.5">
                <AggCell agg={m} />
              </dd>
            </div>
          ))}
        </dl>
      )}

      <div>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen((o) => !o)}
          className="text-xs text-accent hover:underline"
        >
          {open ? t("eval.hideReport") : t("eval.showReport")}
        </button>
        {open && (
          <div id={panelId} className="mt-3">
            <RegressionReportView runId={run.id} />
          </div>
        )}
      </div>
    </article>
  );
}

function Count({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-bg-border px-3 py-2">
      <dt className="text-zinc-500">{label}</dt>
      <dd className="text-zinc-200 tabular-nums mt-0.5">{value}</dd>
    </div>
  );
}

function RegressionReportView({ runId }: { runId: string }) {
  const { t, locale } = useI18n();
  const report = useQuery({
    queryKey: ["regression", runId],
    queryFn: () => getRunRegression(runId),
  });

  if (report.isLoading) return <LoadingSpinner message={t("eval.reportLoading")} />;
  if (report.error) {
    return (
      <ErrorAlert
        error={t("eval.reportFailed", { error: errorMessage(report.error) })}
        onRetry={() => report.refetch()}
      />
    );
  }
  const data = report.data;
  if (!data) return null;
  const checks = data.checks ?? [];
  const na = <span className="text-zinc-600">{t("common.na")}</span>;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
        <RegressionBadge status={data.status} tip="eval.regTip" />
        {data.baseline?.id ? (
          <span>
            {t("eval.reportBaseline", {
              id: data.baseline.id,
              source: t(data.baseline_source === "pinned" ? "eval.baseline.pinned" : "eval.baseline.previous"),
            })}
          </span>
        ) : (
          <span>{t("eval.reportNoBaseline")}</span>
        )}
      </div>
      {checks.length === 0 ? (
        <p className="text-xs text-zinc-500">{t("eval.reportEmpty")}</p>
      ) : (
        <div className="overflow-x-auto -mx-1">
          <table className="w-full text-xs min-w-[640px]">
            <caption className="sr-only">{t("eval.reportCaption")}</caption>
            <thead>
              <tr className="text-left text-zinc-500">
                <th scope="col" className="py-1.5 px-1 font-medium">{t("eval.col.status")}</th>
                <th scope="col" className="py-1.5 px-1 font-medium">{t("eval.col.metric")}</th>
                <th scope="col" className="py-1.5 px-1 font-medium">{t("eval.col.baseline")}</th>
                <th scope="col" className="py-1.5 px-1 font-medium">{t("eval.col.current")}</th>
                <th scope="col" className="py-1.5 px-1 font-medium">{t("eval.col.delta")}</th>
                <th scope="col" className="py-1.5 px-1 font-medium">{t("eval.col.threshold")}</th>
                <th scope="col" className="py-1.5 px-1 font-medium">{t("eval.col.note")}</th>
              </tr>
            </thead>
            <tbody className="text-zinc-300">
              {checks.map((c) => (
                <tr key={c.metric} className="border-t border-bg-border align-top">
                  <td className="py-1.5 px-1">
                    <RegressionBadge status={c.status} tip={null} />
                  </td>
                  <th scope="row" className="py-1.5 px-1 text-left font-mono font-normal text-zinc-200">{c.metric}</th>
                  <td className="py-1.5 px-1 tabular-nums">
                    {c.baseline != null ? formatCheckValue(c.metric, c.baseline, locale) : na}
                  </td>
                  <td className="py-1.5 px-1 tabular-nums">
                    {c.current != null ? formatCheckValue(c.metric, c.current, locale) : na}
                  </td>
                  <td className="py-1.5 px-1 tabular-nums">
                    {c.delta != null ? formatCheckDelta(c.metric, c.delta, locale) : na}
                  </td>
                  <td className="py-1.5 px-1 font-mono">{c.threshold ?? na}</td>
                  <td className="py-1.5 px-1 text-zinc-500">{c.reason ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
