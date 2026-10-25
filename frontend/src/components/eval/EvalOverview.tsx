import { useId, useMemo, useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { errorMessage } from "../../api/client";
import { getEvalRun, listEvalRuns, type EvalRun } from "../../api/eval";
import ErrorAlert from "../ErrorAlert";
import { AggCell, RegressionBadge, StrategyKey } from "./shared";
import LoadingSpinner from "../LoadingSpinner";
import {
  QUALITY_METRICS,
  datasetsOf,
  latestPerConfig,
  questionTypeBreakdown,
  type ConfigRow,
  type QualityMetric,
} from "../../utils/dashboard";
import { formatLatency, formatTokens, formatUsd } from "../../utils/formatting";
import { useI18n } from "../../i18n";

const EMPTY_COMMAND = "python scripts/run_eval.py --dataset golden_v3 --all";

/**
 * Overview tab of the Evaluation page: the latest run of every configuration of
 * one dataset, as a table and a per-question-type breakdown.
 */
export default function EvalOverview() {
  const { t } = useI18n();
  const ids = { dataset: useId(), metric: useId() };
  const runs = useQuery({ queryKey: ["runs"], queryFn: listEvalRuns });
  const datasets = useMemo(() => datasetsOf(runs.data ?? []), [runs.data]);
  const [picked, setPicked] = useState("");
  const dataset = picked && datasets.includes(picked) ? picked : (datasets[0] ?? "");
  const [metric, setMetric] = useState<QualityMetric>("faithfulness");

  const rows = useMemo(() => latestPerConfig(runs.data ?? [], dataset), [runs.data, dataset]);

  const detailQueries = useQueries({
    queries: rows.map((r) => ({
      queryKey: ["run", r.runId],
      queryFn: () => getEvalRun(r.runId),
      staleTime: Infinity,
    })),
  });
  const details = useMemo(() => {
    const out: Record<string, EvalRun | undefined> = {};
    rows.forEach((r, i) => (out[r.runId] = detailQueries[i]?.data));
    return out;
  }, [rows, detailQueries]);
  const detailsLoading = detailQueries.some((q) => q.isLoading);

  const metricLabel = t(QUALITY_METRICS.find((m) => m.id === metric)!.label);

  return (
    <div className="flex flex-col gap-6">
      {runs.isLoading && <LoadingSpinner message={t("dash.loading")} />}
      {runs.error && (
        <ErrorAlert
          error={t("dash.loadFailed", { error: errorMessage(runs.error) })}
          onRetry={() => runs.refetch()}
        />
      )}

      {runs.isSuccess && datasets.length === 0 && (
        <div className="card text-center py-10 flex flex-col items-center gap-3">
          <h2 className="text-white font-semibold">{t("dash.empty")}</h2>
          <p className="text-zinc-400 text-sm max-w-md">{t("dash.emptyHelp")}</p>
          <code className="font-mono text-xs text-amber-200 bg-bg-elevated border border-bg-border rounded px-3 py-2 break-all">
            {EMPTY_COMMAND}
          </code>
        </div>
      )}

      {datasets.length > 0 && (
        <>
          <div className="flex flex-wrap gap-3 items-end">
            <label htmlFor={ids.dataset} className="flex flex-col gap-1 text-xs text-zinc-400">
              {t("dash.dataset")}
              <select id={ids.dataset} value={dataset} onChange={(e) => setPicked(e.target.value)} className="input !py-2 text-sm">
                {datasets.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </label>
            <label htmlFor={ids.metric} className="flex flex-col gap-1 text-xs text-zinc-400">
              {t("dash.quality")}
              <select
                id={ids.metric}
                value={metric}
                onChange={(e) => setMetric(e.target.value as QualityMetric)}
                className="input !py-2 text-sm"
              >
                {QUALITY_METRICS.map((m) => (
                  <option key={m.id} value={m.id}>{t(m.label)}</option>
                ))}
              </select>
            </label>
          </div>

          {rows.length === 0 ? (
            <div className="card text-zinc-400 text-sm">{t("dash.emptyDataset")}</div>
          ) : (
            <>
              <ConfigTable rows={rows} />
              <BreakdownTable
                rows={rows}
                details={details}
                metric={metric}
                metricLabel={metricLabel}
                loading={detailsLoading}
              />
            </>
          )}
        </>
      )}
    </div>
  );
}

function ConfigTable({ rows }: { rows: ConfigRow[] }) {
  const { t, locale, formatNumber } = useI18n();
  const na = <span className="text-zinc-600">{t("common.na")}</span>;
  return (
    <section className="card !p-4 flex flex-col gap-2" aria-labelledby="dash-table-title">
      <h2 id="dash-table-title" className="text-sm font-semibold text-white">{t("dash.table")}</h2>
      <p className="text-xs text-zinc-500">{t("dash.meanStd")}</p>
      <div className="overflow-x-auto -mx-1">
        <table className="w-full text-xs min-w-[900px]">
          <thead>
            <tr className="text-left text-zinc-500">
              <th scope="col" className="py-1.5 px-1 font-medium">{t("dash.col.config")}</th>
              <th scope="col" className="py-1.5 px-1 font-medium">{t("dash.col.runs")}</th>
              <th scope="col" className="py-1.5 px-1 font-medium">{t("dash.col.n")}</th>
              {QUALITY_METRICS.map((m) => (
                <th key={m.id} scope="col" className="py-1.5 px-1 font-medium">{t(m.label)}</th>
              ))}
              <th scope="col" className="py-1.5 px-1 font-medium">{t("dash.col.tokens")}</th>
              <th scope="col" className="py-1.5 px-1 font-medium">{t("dash.col.cost")}</th>
              <th scope="col" className="py-1.5 px-1 font-medium">{t("dash.col.p50")}</th>
              <th scope="col" className="py-1.5 px-1 font-medium">{t("dash.col.p95")}</th>
              <th scope="col" className="py-1.5 px-1 font-medium">{t("dash.col.regression")}</th>
            </tr>
          </thead>
          <tbody className="text-zinc-300">
            {rows.map((r) => (
              <tr key={r.key} className="border-t border-bg-border align-top">
                <th scope="row" className="py-1.5 px-1 text-left font-normal text-zinc-200 max-w-[16rem]">
                  <StrategyKey strategy={r.strategy} label={r.label} />
                </th>
                <td className="py-1.5 px-1 tabular-nums">{formatNumber(r.runCount)}</td>
                <td className="py-1.5 px-1 tabular-nums">
                  {formatNumber(r.nScored)} / {formatNumber(r.n)}
                </td>
                {QUALITY_METRICS.map((m) => (
                  <td key={m.id} className="py-1.5 px-1">
                    <AggCell agg={r.metrics[m.id]} />
                  </td>
                ))}
                <td className="py-1.5 px-1 tabular-nums">
                  {r.tokensPerQuestion != null ? formatTokens(Math.round(r.tokensPerQuestion), locale) : na}
                </td>
                <td className="py-1.5 px-1 tabular-nums">
                  {r.costPerQuestion != null ? formatUsd(r.costPerQuestion, locale) : na}
                </td>
                <td className="py-1.5 px-1 tabular-nums">
                  {r.p50LatencyMs != null ? formatLatency(r.p50LatencyMs, locale) : na}
                </td>
                <td className="py-1.5 px-1 tabular-nums">
                  {r.p95LatencyMs != null ? formatLatency(r.p95LatencyMs, locale) : na}
                </td>
                <td className="py-1.5 px-1">
                  <RegressionBadge status={r.regression} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function BreakdownTable({
  rows,
  details,
  metric,
  metricLabel,
  loading,
}: {
  rows: ConfigRow[];
  details: Record<string, EvalRun | undefined>;
  metric: QualityMetric;
  metricLabel: string;
  loading: boolean;
}) {
  const { t } = useI18n();
  const table = questionTypeBreakdown(rows, details, metric);
  return (
    <section className="card !p-4 flex flex-col gap-2" aria-labelledby="dash-breakdown-title">
      <h2 id="dash-breakdown-title" className="text-sm font-semibold text-white">
        {t("dash.breakdown", { metric: metricLabel })}
      </h2>
      {table.rows.length === 0 ? (
        <p className="text-xs text-zinc-500">{loading ? t("dash.breakdownLoading") : t("dash.breakdownEmpty")}</p>
      ) : (
        <div className="overflow-x-auto -mx-1">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-zinc-500">
                <th scope="col" className="py-1.5 px-1 font-medium">{t("dash.col.config")}</th>
                {table.types.map((ty) => (
                  <th key={ty} scope="col" className="py-1.5 px-1 font-medium font-mono">{ty}</th>
                ))}
              </tr>
            </thead>
            <tbody className="text-zinc-300">
              {table.rows.map((r) => (
                <tr key={r.key} className="border-t border-bg-border">
                  <th scope="row" className="py-1.5 px-1 text-left font-normal text-zinc-200">
                    <StrategyKey strategy={r.strategy} label={r.label} />
                  </th>
                  {table.types.map((ty) => (
                    <td key={ty} className="py-1.5 px-1">
                      <AggCell agg={r.cells[ty]} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
