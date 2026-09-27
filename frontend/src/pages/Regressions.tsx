import { useQuery } from "@tanstack/react-query";
import { errorMessage, get } from "../api/client";
import ErrorAlert from "../components/ErrorAlert";
import { useI18n, type MessageKey } from "../i18n";

type Reg = { metric: string; delta: number; from: string; to: string };

/** Metric names the Evaluation page already has labels for. */
const METRIC_LABELS: Record<string, MessageKey> = {
  faithfulness: "eval.metric.faithfulness",
  answer_relevance: "eval.metric.answer_relevance",
  context_precision: "eval.metric.context_precision",
  context_recall: "eval.metric.context_recall",
};

export default function Regressions() {
  const { t, formatNumber } = useI18n();
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["regressions"],
    queryFn: () => get<Reg[]>("/eval/regressions"),
  });

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-4xl font-semibold text-white">{t("reg.title")}</h1>
        <p className="text-zinc-400 max-w-xl">{t("reg.subtitle")}</p>
      </header>

      {isLoading && <div className="card text-zinc-400">{t("common.loading")}</div>}

      {error && (
        <ErrorAlert
          error={t("reg.loadFailed", { error: errorMessage(error) })}
          onRetry={() => refetch()}
        />
      )}

      {/* Only claim "no regressions" when the check actually ran. */}
      {!isLoading && !error && (!data || data.length === 0) && (
        <div className="card text-center py-10">
          <div className="text-emerald-400 text-3xl">✓</div>
          <h2 className="text-white font-semibold mt-2">{t("reg.none")}</h2>
          <p className="text-zinc-400 text-sm mt-1">{t("reg.stable")}</p>
        </div>
      )}

      {data && data.length > 0 && (
        <div className="grid gap-2">
          {data.map((r, i) => {
            const bad = r.delta < 0;
            const label = METRIC_LABELS[r.metric];
            return (
              <div key={i} className="card !p-4 flex items-center justify-between">
                <div>
                  <div className="font-medium text-white">{label ? t(label) : r.metric}</div>
                  <div className="text-xs text-zinc-500 font-mono mt-0.5">
                    {r.from} → {r.to}
                  </div>
                </div>
                <div
                  className={`chip ${bad ? "!text-rose-300 !border-rose-500/40 !bg-rose-500/10" : "!text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10"}`}
                >
                  Δ{" "}
                  {formatNumber(r.delta, {
                    signDisplay: "exceptZero",
                    minimumFractionDigits: 3,
                    maximumFractionDigits: 3,
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
