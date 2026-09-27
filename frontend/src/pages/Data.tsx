import { useEffect, useState } from "react";
import { get } from "../api/client";
import { formatLatency } from "../utils/formatting";
import { useI18n, type MessageKey } from "../i18n";

interface CorpusStats {
  indexed_chunks: number;
}

type Best = "classic" | "graph" | "agentic";

const QUESTIONS: {
  level: "medium" | "hard";
  q: MessageKey;
  best: Best;
  classic: number;
  graph: number;
  agentic: number;
}[] = [
  { level: "medium", q: "data.q1", best: "graph", classic: 8300, graph: 7100, agentic: 26000 },
  { level: "hard", q: "data.q2", best: "agentic", classic: 7800, graph: 8100, agentic: 18000 },
  { level: "hard", q: "data.q3", best: "agentic", classic: 9100, graph: 8700, agentic: 21000 },
];

const INSIGHTS: { pattern: MessageKey; impact: MessageKey }[] = [
  { pattern: "data.pattern.entity", impact: "data.pattern.entity.impact" },
  { pattern: "data.pattern.factual", impact: "data.pattern.factual.impact" },
  { pattern: "data.pattern.synthesis", impact: "data.pattern.synthesis.impact" },
];

const BEST_LABEL: Record<Best, MessageKey> = {
  classic: "strategy.classic.short",
  graph: "strategy.graph.short",
  agentic: "strategy.agentic.short",
};

export default function DataPage() {
  const { t, locale, formatNumber } = useI18n();
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    get<CorpusStats>("/ingest/stats")
      .then(setStats)
      .catch(e => console.error("Failed to load corpus stats:", e))
      .finally(() => setLoading(false));
  }, []);

  const indexed_chunks = stats?.indexed_chunks ?? 0;
  const compact = (v: number) =>
    formatNumber(v, { notation: "compact", maximumFractionDigits: 1 });

  return (
    <div className="flex flex-col gap-6 max-w-6xl">
      <header>
        <h1 className="display text-4xl font-semibold text-white">{t("data.title")}</h1>
        <p className="text-sm text-zinc-400 mt-2">{t("data.subtitle")}</p>
      </header>

      {/* Stats Row - Compact */}
      {!loading && (
        <div className="grid grid-cols-5 gap-3">
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">{t("data.chunks")}</div>
            <div className="text-2xl font-bold text-white mt-1">{formatNumber(indexed_chunks)}</div>
          </div>
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">{t("data.tokens")}</div>
            <div className="text-2xl font-bold text-white mt-1">{compact(indexed_chunks * 300)}</div>
          </div>
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">{t("data.size")}</div>
            <div className="text-2xl font-bold text-white mt-1">
              {formatNumber(indexed_chunks * 0.012, {
                style: "unit",
                unit: "megabyte",
                unitDisplay: "narrow",
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
            </div>
          </div>
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">{t("data.status")}</div>
            <div className="text-2xl font-bold text-emerald-400 mt-1">{t("data.live")}</div>
          </div>
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">{t("data.dataset")}</div>
            <div className="text-2xl font-bold text-white mt-1">{compact(100000)}</div>
          </div>
        </div>
      )}

      {/* Performance Table */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-white mb-4">{t("data.goldenPerf")}</h2>
        <div className="space-y-2">
          {QUESTIONS.map((q, i) => (
            <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-bg-surface border border-bg-border/50 hover:border-bg-border transition-colors">
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <span className={`px-2 py-1 rounded text-xs font-semibold whitespace-nowrap ${
                  q.level === "hard" ? "bg-amber-500/20 text-amber-200" : "bg-emerald-500/20 text-emerald-200"
                }`}>
                  {t(q.level === "hard" ? "data.level.hard" : "data.level.medium")}
                </span>
                <span className="text-sm text-zinc-300 truncate">{t(q.q)}</span>
              </div>
              <div className="flex gap-4 items-center ml-4">
                <div className="text-right">
                  <div className="text-xs text-zinc-500">{t("strategy.classic.short")}</div>
                  <div className="text-sm font-mono text-zinc-300">{formatLatency(q.classic, locale)}</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-zinc-500">{t("strategy.graph.short")}</div>
                  <div className="text-sm font-mono text-violet-300">{formatLatency(q.graph, locale)}</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-zinc-500">{t("strategy.agentic.short")}</div>
                  <div className="text-sm font-mono text-emerald-300">{formatLatency(q.agentic, locale)}</div>
                </div>
                <span className={`px-2.5 py-1 rounded text-xs font-semibold whitespace-nowrap ${
                  q.best === "graph" ? "bg-violet-500/20 text-violet-200" :
                  q.best === "agentic" ? "bg-emerald-500/20 text-emerald-200" :
                  "bg-sky-500/20 text-sky-200"
                }`}>
                  {t(BEST_LABEL[q.best])}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Patterns Grid */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-white mb-4">{t("data.patterns")}</h2>
        <div className="grid grid-cols-3 gap-4">
          {INSIGHTS.map((insight, i) => (
            <div key={i} className="p-4 rounded-lg bg-bg-surface border border-bg-border">
              <div className="font-semibold text-white text-sm">{t(insight.pattern)}</div>
              <div className="text-xs text-zinc-400 mt-2">{t(insight.impact)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
