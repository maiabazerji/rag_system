/**
 * Small building blocks shared by the Evaluation page's Overview and Runs tabs,
 * so a score or a regression status reads the same wherever it appears.
 */
import type { MetricAggregate, RegressionStatus } from "../../api/eval";
import { strategyColor } from "../../utils/chartColors";
import { formatScore } from "../../utils/formatting";
import { useI18n, type MessageKey } from "../../i18n";

const REG_STYLE: Record<RegressionStatus, { icon: string; cls: string; label: MessageKey }> = {
  PASS: { icon: "✓", cls: "!text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10", label: "dash.reg.PASS" },
  FAIL: { icon: "✕", cls: "!text-rose-300 !border-rose-500/40 !bg-rose-500/10", label: "dash.reg.FAIL" },
  SKIPPED: { icon: "–", cls: "", label: "dash.reg.SKIPPED" },
  ERROR: { icon: "!", cls: "!text-amber-300 !border-amber-500/40 !bg-amber-500/10", label: "dash.reg.ERROR" },
};

/**
 * PASS / FAIL / SKIPPED / ERROR chip; "Not checked" when the run has no status.
 * `tip` explains what the status refers to; pass null for none.
 */
export function RegressionBadge({
  status,
  tip = "dash.regTip",
}: {
  status: RegressionStatus | string | null | undefined;
  tip?: MessageKey | null;
}) {
  const { t } = useI18n();
  const style = status && status in REG_STYLE ? REG_STYLE[status as RegressionStatus] : undefined;
  return (
    <span className={`chip ${style?.cls ?? ""}`} title={tip ? t(tip) : undefined}>
      <span aria-hidden="true">{style?.icon ?? "·"}</span>
      {t(style?.label ?? "dash.reg.none")}
    </span>
  );
}

/** "mean ± std (n)", or n/a when the metric was not measured. */
export function AggCell({ agg }: { agg: Pick<MetricAggregate, "mean" | "std"> & { n: number | null } | null | undefined }) {
  const { t, locale } = useI18n();
  if (!agg || agg.mean == null) return <span className="text-zinc-600">{t("common.na")}</span>;
  return (
    <span className="tabular-nums">
      {formatScore(agg.mean, 2, locale)}
      {agg.std != null && <span className="text-zinc-500"> ± {formatScore(agg.std, 2, locale)}</span>}
      {agg.n != null && <span className="text-zinc-600"> ({agg.n})</span>}
    </span>
  );
}

/** A strategy's colour dot followed by a label. */
export function StrategyKey({ strategy, label }: { strategy: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: strategyColor(strategy) }} />
      <span className="break-words">{label}</span>
    </span>
  );
}
