import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, errorMessage, get, getAdminKey, post, setAdminKey } from "../api/client";
import {
  STRATEGY_NAMES,
  compareStrategies,
  listGoldenDatasets,
  listGoldenQuestions,
  type CompareStrategiesRequest,
  type CompareStrategiesResponse,
  type StrategyComparison,
  type StrategyName,
  type TraceStep,
} from "../api/eval";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorAlert from "../components/ErrorAlert";
import Tooltip from "../components/Tooltip";
import {
  CitedText,
  GroundingBadge,
  InvalidCitationsWarning,
  SourceLocation,
  type CitedSource,
} from "../components/Citations";
import { handleNumber, sourceAnchor } from "../utils/citations";
import { formatLatency, formatScore, formatTokens, formatUsd } from "../utils/formatting";
import {
  COMPARE_METRICS,
  bestValues,
  isFailedRow,
  latencyStages,
  totalLatency,
  whyThisResult,
  type CompareMetric,
  type Fact,
} from "../utils/compareInsights";
import { STAGE_COLORS } from "../utils/chartColors";
import { useI18n, type I18n, type MessageKey } from "../i18n";

type StrategyOut = StrategyComparison;
type GraphStats = { triples: number; entities: number };
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
type QuestionSource = "own" | "golden";
type Best = Record<string, Set<StrategyName>>;

const POLL_INTERVAL_MS = 2000;
const ALL_TYPES = "";

/** Graph build and reset need the admin key on top of any API key. */
function graphBuildError(e: unknown, t: I18n["t"]): string {
  const detail = e instanceof ApiError && e.detail ? ` (${e.detail})` : "";
  if (e instanceof ApiError && e.status === 403) {
    return t("compare.adminNeeded", { detail });
  }
  if (e instanceof ApiError && e.status === 503) {
    return t("compare.buildUnavailable", { detail });
  }
  return errorMessage(e);
}
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const STRATEGIES: StrategyName[] = [...STRATEGY_NAMES];

const META: Record<
  StrategyName,
  { title: MessageKey; short: MessageKey; tagline: MessageKey; accent: string }
> = {
  classic: {
    title: "strategy.classic",
    short: "strategy.classic.short",
    tagline: "compare.tagline.classic",
    accent: "from-sky-500/20 to-sky-500/0",
  },
  graph: {
    title: "strategy.graph",
    short: "strategy.graph.short",
    tagline: "compare.tagline.graph",
    accent: "from-violet-500/20 to-violet-500/0",
  },
  agentic: {
    title: "strategy.agentic",
    short: "strategy.agentic.short",
    tagline: "compare.tagline.agentic",
    accent: "from-emerald-500/20 to-emerald-500/0",
  },
};

const SAMPLES: MessageKey[] = ["compare.sample1", "compare.sample2", "compare.sample3"];

const JOB_STATUS: Record<JobStatus, MessageKey> = {
  queued: "compare.status.queued",
  running: "compare.status.running",
  completed: "compare.status.completed",
  failed: "compare.status.failed",
};

/** Split "a.md, b.md" into ["a.md", "b.md"]. */
function parseDocList(value: string): string[] {
  return value
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export default function Compare() {
  const { t, tp } = useI18n();
  const [q, setQ] = useState("");
  const [source, setSource] = useState<QuestionSource>("own");
  const [dataset, setDataset] = useState("");
  const [typeFilter, setTypeFilter] = useState(ALL_TYPES);
  const [goldenId, setGoldenId] = useState("");
  const [evaluate, setEvaluate] = useState(false);
  const [idealAnswer, setIdealAnswer] = useState("");
  const [relevantDocs, setRelevantDocs] = useState("");
  const [res, setRes] = useState<CompareStrategiesResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [buildingGraph, setBuildingGraph] = useState(false);
  const [buildProgress, setBuildProgress] = useState<string | null>(null);
  const [showAdminKey, setShowAdminKey] = useState(false);
  const evaluateHelpId = useId();
  const { data: graphStats = null, refetch: refetchStats } = useQuery({
    queryKey: ["graph-stats"],
    queryFn: () => get<GraphStats>("/graph/stats"),
  });

  // Stops the build poller when the page unmounts.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const golden = source === "golden";
  const canRun = !busy && !!q.trim() && (!golden || !!goldenId);

  function buildRequest(): CompareStrategiesRequest {
    const req: CompareStrategiesRequest = { question: q, strategies: STRATEGIES };
    if (!evaluate) return req;
    req.evaluate = true;
    if (golden) {
      req.golden = { dataset, id: goldenId };
    } else {
      const docs = parseDocList(relevantDocs);
      if (idealAnswer.trim() || docs.length > 0) {
        req.reference = {
          ...(idealAnswer.trim() ? { ideal_answer: idealAnswer.trim() } : {}),
          ...(docs.length > 0 ? { relevant_doc_ids: docs } : {}),
        };
      }
    }
    return req;
  }

  async function run() {
    if (!canRun) return;
    setErr(null);
    setBusy(true);
    setRes(null);
    try {
      setRes(await compareStrategies(buildRequest()));
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  /** Start a build, then poll its job until it finishes. It can run for minutes. */
  async function buildGraph() {
    if (buildingGraph) return;
    setErr(null);
    setBuildingGraph(true);
    setBuildProgress(t(JOB_STATUS.queued));
    try {
      const started = await post<BuildJob>("/graph/build", {}, { admin: true });
      const jobId = started.job_id ?? started.id;
      if (!jobId) throw new Error(t("compare.noJobId"));
      let job = started;
      while (mounted.current && job.status !== "completed" && job.status !== "failed") {
        await sleep(POLL_INTERVAL_MS);
        if (!mounted.current) return;
        job = await get<BuildJob>(`/graph/build/${encodeURIComponent(jobId)}`, { admin: true });
        setBuildProgress(
          job.total
            ? t("compare.progressChunks", { done: job.completed ?? 0, total: job.total })
            : t(JOB_STATUS[job.status] ?? "compare.status.running"),
        );
      }
      if (!mounted.current) return;
      if (job.status === "failed") {
        setErr(t("compare.buildFailed", { error: job.error ?? t("compare.unknownError") }));
      } else if (job.failures && job.failures.length > 0) {
        setErr(tp("compare.partialFailures", job.failures.length));
      }
      await refetchStats();
    } catch (e) {
      if (!mounted.current) return;
      setErr(graphBuildError(e, t));
      if (e instanceof ApiError && e.status === 403) setShowAdminKey(true);
    } finally {
      if (mounted.current) {
        setBuildingGraph(false);
        setBuildProgress(null);
      }
    }
  }

  const results = useMemo(() => res?.results ?? [], [res]);
  const best = useMemo(() => bestValues(results), [results]);
  const byName = (n: StrategyName) => results.find((r) => r.strategy === n);

  function switchSource(next: QuestionSource) {
    setSource(next);
    setQ("");
    setGoldenId("");
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-4xl font-semibold text-white">{t("compare.title")}</h1>
        <p className="text-sm text-zinc-400">{t("compare.subtitle")}</p>
      </header>

      <div className="card flex flex-col gap-3 p-3 sm:p-4">
        <fieldset className="flex flex-wrap items-center gap-2">
          <legend className="sr-only">{t("compare.sourceLabel")}</legend>
          {(["own", "golden"] as const).map((s) => (
            <label
              key={s}
              className={`chip cursor-pointer min-h-[36px] focus-within:ring-2 focus-within:ring-accent/50 ${
                source === s ? "!border-accent/60 !text-white" : "hover:text-white"
              }`}
            >
              <input
                type="radio"
                name="compare-source"
                value={s}
                checked={source === s}
                onChange={() => switchSource(s)}
                className="sr-only"
              />
              {t(s === "own" ? "compare.source.own" : "compare.source.golden")}
            </label>
          ))}
        </fieldset>

        {golden && (
          <GoldenPicker
            dataset={dataset}
            onDataset={(d) => {
              setDataset(d);
              setGoldenId("");
              setQ("");
              setTypeFilter(ALL_TYPES);
            }}
            typeFilter={typeFilter}
            onTypeFilter={setTypeFilter}
            selectedId={goldenId}
            onSelect={(id, question) => {
              setGoldenId(id);
              setQ(question);
            }}
          />
        )}

        <label className="flex flex-col gap-1">
          <span className="sr-only">{t("compare.question")}</span>
          <textarea
            value={q}
            readOnly={golden}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                run();
              }
            }}
            rows={2}
            placeholder={golden ? t("compare.goldenPick") : t("compare.placeholder")}
            className={`input resize-none text-sm ${golden ? "opacity-80" : ""}`}
          />
        </label>

        {!golden && (
          <div className="flex flex-wrap gap-1">
            {SAMPLES.map((key) => {
              const s = t(key);
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setQ(s)}
                  className="chip text-xs text-left hover:text-white hover:border-accent/40 min-h-[36px]"
                >
                  {s}
                </button>
              );
            })}
          </div>
        )}

        <div className="flex flex-col gap-2">
          <label className="inline-flex items-center gap-2 text-sm text-zinc-200 cursor-pointer self-start min-h-[36px]">
            <input
              type="checkbox"
              role="switch"
              checked={evaluate}
              onChange={(e) => setEvaluate(e.target.checked)}
              aria-describedby={evaluateHelpId}
              className="h-4 w-4 accent-[#e8a93c]"
            />
            {t("compare.evaluate")}
          </label>
          <p id={evaluateHelpId} className="text-xs text-zinc-500 max-w-2xl">
            {t("compare.evaluateHelp")}
          </p>
          {evaluate && golden && (
            <p className="text-xs text-zinc-400">{t("compare.goldenReference")}</p>
          )}
          {evaluate && !golden && (
            <ReferenceFields
              idealAnswer={idealAnswer}
              onIdealAnswer={setIdealAnswer}
              relevantDocs={relevantDocs}
              onRelevantDocs={setRelevantDocs}
            />
          )}
        </div>

        <div className="flex items-center justify-between gap-2 flex-wrap">
          <GraphStatus
            stats={graphStats}
            busy={buildingGraph}
            progress={buildProgress}
            onBuild={buildGraph}
          />
          <button
            type="button"
            onClick={run}
            disabled={!canRun}
            className="btn-primary text-sm px-4 py-2 min-h-[44px] w-full sm:w-auto"
          >
            {busy ? t("common.running") : t("compare.run")}
          </button>
        </div>
        {err && (
          <ErrorAlert
            error={err}
            onRetry={() => {
              setErr(null);
              run();
            }}
            onDismiss={() => setErr(null)}
          />
        )}
        <AdminKeyField open={showAdminKey} onToggle={setShowAdminKey} />
      </div>

      {busy && evaluate && <p className="text-xs text-zinc-400" role="status">{t("compare.evaluating")}</p>}

      {results.length > 1 && <SummaryTable results={results} best={best} />}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {STRATEGIES.map((s) => (
          <StrategyCard key={s} name={s} result={byName(s)} loading={busy} best={best} />
        ))}
      </div>
    </div>
  );
}

function GoldenPicker({
  dataset,
  onDataset,
  typeFilter,
  onTypeFilter,
  selectedId,
  onSelect,
}: {
  dataset: string;
  onDataset: (d: string) => void;
  typeFilter: string;
  onTypeFilter: (t: string) => void;
  selectedId: string;
  onSelect: (id: string, question: string) => void;
}) {
  const { t } = useI18n();
  const ids = { dataset: useId(), type: useId(), list: useId() };
  const datasets = useQuery({ queryKey: ["golden-datasets"], queryFn: listGoldenDatasets });
  const active = dataset || datasets.data?.[0] || "";
  const questions = useQuery({
    queryKey: ["golden-questions", active],
    queryFn: () => listGoldenQuestions(active),
    enabled: !!active,
  });

  // Adopt the first dataset once the list arrives, so the parent sends it.
  useEffect(() => {
    if (!dataset && active) onDataset(active);
  }, [dataset, active, onDataset]);

  const types = useMemo(
    () => [...new Set((questions.data ?? []).map((q) => q.question_type))].sort(),
    [questions.data],
  );
  const shown = (questions.data ?? []).filter((q) => !typeFilter || q.question_type === typeFilter);
  const error = datasets.error ?? questions.error;

  return (
    <div className="grid gap-2 sm:grid-cols-2">
      <label htmlFor={ids.dataset} className="flex flex-col gap-1 text-xs text-zinc-400">
        {t("compare.dataset")}
        <select
          id={ids.dataset}
          value={active}
          onChange={(e) => onDataset(e.target.value)}
          className="input !py-2 text-sm"
        >
          {(datasets.data ?? []).map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </label>
      <label htmlFor={ids.type} className="flex flex-col gap-1 text-xs text-zinc-400">
        {t("compare.questionType")}
        <select
          id={ids.type}
          value={typeFilter}
          onChange={(e) => onTypeFilter(e.target.value)}
          className="input !py-2 text-sm"
        >
          <option value={ALL_TYPES}>{t("compare.allTypes")}</option>
          {types.map((ty) => (
            <option key={ty} value={ty}>
              {ty}
            </option>
          ))}
        </select>
      </label>
      <label htmlFor={ids.list} className="flex flex-col gap-1 text-xs text-zinc-400 sm:col-span-2">
        {t("compare.question")}
        {(datasets.isLoading || questions.isLoading) && (
          <span className="text-zinc-500">{t("compare.goldenLoading")}</span>
        )}
        {error ? (
          <span className="text-rose-300">{t("compare.goldenFailed", { error: errorMessage(error) })}</span>
        ) : shown.length === 0 && questions.isSuccess ? (
          <span className="text-zinc-500">{t("compare.goldenEmpty")}</span>
        ) : (
          <select
            id={ids.list}
            size={Math.min(6, Math.max(2, shown.length))}
            value={selectedId}
            onChange={(e) => {
              const picked = shown.find((q) => q.id === e.target.value);
              if (picked) onSelect(picked.id, picked.question);
            }}
            className="input !py-1 text-sm"
          >
            {shown.map((q) => (
              <option key={q.id} value={q.id} className="py-1 whitespace-normal">
                {q.id} · {q.question_type}
                {q.difficulty ? ` · ${q.difficulty}` : ""} — {q.question}
              </option>
            ))}
          </select>
        )}
      </label>
    </div>
  );
}

function ReferenceFields({
  idealAnswer,
  onIdealAnswer,
  relevantDocs,
  onRelevantDocs,
}: {
  idealAnswer: string;
  onIdealAnswer: (v: string) => void;
  relevantDocs: string;
  onRelevantDocs: (v: string) => void;
}) {
  const { t } = useI18n();
  const ids = { answer: useId(), docs: useId(), answerHelp: useId(), docsHelp: useId() };
  return (
    <details className="text-xs text-zinc-400">
      <summary className="cursor-pointer select-none hover:text-zinc-200">{t("compare.reference")}</summary>
      <div className="mt-2 grid gap-2 sm:grid-cols-2">
        <div className="flex flex-col gap-1">
          <label htmlFor={ids.answer}>{t("compare.idealAnswer")}</label>
          <textarea
            id={ids.answer}
            rows={2}
            value={idealAnswer}
            onChange={(e) => onIdealAnswer(e.target.value)}
            aria-describedby={ids.answerHelp}
            className="input resize-y !py-2 text-sm"
          />
          <span id={ids.answerHelp} className="text-zinc-500">{t("compare.idealAnswerHelp")}</span>
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor={ids.docs}>{t("compare.relevantDocs")}</label>
          <input
            id={ids.docs}
            value={relevantDocs}
            onChange={(e) => onRelevantDocs(e.target.value)}
            aria-describedby={ids.docsHelp}
            spellCheck={false}
            className="input !py-2 text-sm font-mono"
          />
          <span id={ids.docsHelp} className="text-zinc-500">{t("compare.relevantDocsHelp")}</span>
        </div>
      </div>
    </details>
  );
}

/** Format one comparison metric value for display. */
function useMetricText() {
  const { t, locale, formatNumber } = useI18n();
  return (m: CompareMetric, v: number | null | undefined): string => {
    if (v == null || !Number.isFinite(v)) return t("compare.notMeasured");
    switch (m.kind) {
      case "score":
        return formatScore(v, 2, locale);
      case "latency":
        return formatLatency(v, locale);
      case "tokens":
        return formatTokens(v, locale);
      case "usd":
        return formatUsd(v, locale);
      default:
        return formatNumber(v);
    }
  };
}

function BestMark() {
  const { t } = useI18n();
  return (
    <span className="ml-1 inline-flex items-center gap-0.5 rounded-full border border-accent/50 bg-accent/10 px-1.5 text-[10px] font-semibold text-accent">
      <span aria-hidden="true">★</span>
      <span aria-hidden="true">{t("compare.best")}</span>
      <span className="sr-only">{t("compare.bestAria")}</span>
    </span>
  );
}

/** Metrics as rows, strategies as columns; best measured value per metric marked. */
function SummaryTable({ results, best }: { results: StrategyOut[]; best: Best }) {
  const { t } = useI18n();
  const text = useMetricText();
  const shown = COMPARE_METRICS.filter((m) => results.some((r) => m.value(r) != null && !isFailedRow(r)));
  if (shown.length === 0) return null;
  const cols = STRATEGIES.filter((s) => results.some((r) => r.strategy === s));
  return (
    <section className="card !p-3 sm:!p-4 flex flex-col gap-2" aria-labelledby="compare-summary">
      <h2 id="compare-summary" className="text-sm font-semibold text-white">{t("compare.summary")}</h2>
      <p className="text-xs text-zinc-500">{t("compare.summaryNote")}</p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-zinc-500">
              <th scope="col" className="py-1.5 pr-3 font-medium">{t("compare.metric")}</th>
              {cols.map((s) => (
                <th key={s} scope="col" className="py-1.5 pr-3 font-medium text-zinc-300">
                  {t(META[s].short)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((m) => (
              <tr key={m.id} className="border-t border-bg-border">
                <th scope="row" className="py-1.5 pr-3 text-left font-normal text-zinc-400">{t(m.label)}</th>
                {cols.map((s) => {
                  const row = results.find((r) => r.strategy === s)!;
                  const isBest = best[m.id]?.has(s);
                  return (
                    <td key={s} className={`py-1.5 pr-3 font-mono tabular-nums ${isBest ? "text-white font-semibold" : "text-zinc-300"}`}>
                      {isFailedRow(row) ? "–" : text(m, m.value(row))}
                      {isBest && <BestMark />}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function StrategyCard({
  name,
  result,
  loading,
  best,
}: {
  name: StrategyName;
  result?: StrategyOut;
  loading: boolean;
  best: Best;
}) {
  const { t, locale } = useI18n();
  const m = META[name];
  const [expanded, setExpanded] = useState(false);
  const anchorPrefix = useId().replace(/:/g, "");
  const failed = result ? isFailedRow(result) : false;

  return (
    <article
      className="card relative overflow-hidden flex flex-col gap-3 min-h-[280px] !p-3 sm:!p-4"
      aria-labelledby={`${anchorPrefix}-title`}
    >
      <div className={`absolute inset-x-0 top-0 h-12 bg-gradient-to-b ${m.accent} pointer-events-none`} />

      <header className="relative flex flex-col gap-0.5">
        <h2 id={`${anchorPrefix}-title`} className="text-sm font-bold text-white">{t(m.title)}</h2>
        <div className="text-[11px] text-zinc-500 font-mono">{t(m.tagline)}</div>
      </header>

      {loading && !result && <LoadingSpinner />}

      {result && (
        <>
          <div className="flex items-center gap-1.5 flex-wrap text-xs">
            <span
              className={`chip ${result.refusal ? "!text-amber-300 !border-amber-500/40 !bg-amber-500/10" : ""}`}
            >
              <span aria-hidden="true">{result.refusal ? "⚠" : "✓"}</span>
              {result.refusal ? t("common.refused") : t("common.answered")}
            </span>
            <GroundingBadge g={result} />
          </div>

          {result.refusal || failed ? (
            <div className="text-amber-200 text-xs p-2 bg-amber-500/10 rounded border border-amber-500/30">
              {result.answer}
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              <div className={`text-zinc-200 text-sm leading-relaxed answer-content ${!expanded ? "line-clamp-6" : ""}`}>
                <FormattedAnswer text={result.answer} cite={{ anchorPrefix, sources: result.sources }} />
              </div>
              <button
                type="button"
                onClick={() => setExpanded(!expanded)}
                aria-expanded={expanded}
                className="text-xs text-accent hover:text-accent-hover font-semibold self-start"
              >
                {expanded ? t("compare.showLess") : t("compare.readMore")}
              </button>
            </div>
          )}

          <InvalidCitationsWarning invalid={result.invalid_citations} />

          {!failed && <EvaluationBlock result={result} best={best} />}
          {!failed && <CostBlock result={result} best={best} />}

          {!failed && result.sources.length > 0 && result.sources[0].chunk_id !== "none" && (
            <details className="text-xs text-zinc-400">
              <summary className="cursor-pointer font-semibold hover:text-zinc-200">
                {t("common.sources", { count: result.sources.length })}
              </summary>
              <ol className="mt-1.5 space-y-1">
                {result.sources.map((s, i) => (
                  <li
                    key={`${s.chunk_id}-${i}`}
                    id={s.handle ? sourceAnchor(anchorPrefix, s.handle) : undefined}
                    className="text-[11px] text-zinc-400 target:bg-accent/10 rounded break-words"
                    title={s.quote}
                  >
                    <span className="font-mono">[{s.handle ? handleNumber(s.handle) : i + 1}]</span>{" "}
                    <SourceLocation s={s} />
                    {s.relevance_score != null && (
                      <span className="text-zinc-500">
                        {" "}· {t("compare.relevanceScore", { score: formatScore(s.relevance_score, 2, locale) })}
                      </span>
                    )}
                  </li>
                ))}
              </ol>
            </details>
          )}

          <WhyPanel result={result} />

          {result.extra && Object.keys(result.extra).length > 0 && (
            <ExtraDetails name={name} extra={result.extra} />
          )}
          {result.trace && result.trace.length > 0 && <TraceDetails trace={result.trace} />}
        </>
      )}
    </article>
  );
}

function MetricTile({
  label,
  value,
  isBest,
  note,
}: {
  label: string;
  value: string;
  isBest?: boolean;
  note?: string;
}) {
  return (
    <div className="rounded-md border border-bg-border bg-bg-elevated px-2 py-1.5" title={note}>
      <div className="text-[11px] text-zinc-500">{label}</div>
      <div className={`font-mono text-sm tabular-nums ${isBest ? "text-white font-semibold" : "text-zinc-200"}`}>
        {value}
        {isBest && <BestMark />}
      </div>
    </div>
  );
}

const byId = (id: string) => COMPARE_METRICS.find((m) => m.id === id)!;

function EvaluationBlock({ result, best }: { result: StrategyOut; best: Best }) {
  const { t } = useI18n();
  const text = useMetricText();
  const ev = result.evaluation;
  if (!ev) return null;

  if (ev.status === "refusal_checked") {
    return (
      <p className="text-xs text-zinc-300">
        {ev.correct_refusal === 1 ? t("compare.refusalCorrect") : t("compare.refusalWrong")}
      </p>
    );
  }

  const tiles: { id: string; note?: string }[] = [
    { id: "faithfulness" },
    { id: "answer_relevance" },
    { id: "answer_correctness", note: ev.has_reference_answer ? undefined : t("compare.noReference") },
    { id: "recall@5", note: ev.retrieval ? undefined : t("compare.noRelevantDocs") },
    { id: "mrr", note: ev.retrieval ? undefined : t("compare.noRelevantDocs") },
  ];
  return (
    <section className="flex flex-col gap-1.5" aria-label={t("compare.scores")}>
      {ev.status === "judge_failed" && (
        <p role="status" className="text-[11px] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded px-2 py-1">
          {t("compare.judgeFailed", { error: ev.error ?? ev.error_type ?? t("compare.unknownError") })}
        </p>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
        {tiles.map(({ id, note }) => {
          const m = byId(id);
          return (
            <MetricTile
              key={id}
              label={t(m.label)}
              value={text(m, m.value(result))}
              isBest={best[id]?.has(result.strategy)}
              note={note}
            />
          );
        })}
      </div>
      {!ev.retrieval && <p className="text-[11px] text-zinc-500">{t("compare.noRelevantDocs")}</p>}
    </section>
  );
}

function CostBlock({ result, best }: { result: StrategyOut; best: Best }) {
  const { t, locale } = useI18n();
  const text = useMetricText();
  const metrics = result.metrics;
  const inTok = metrics?.input_tokens ?? result.input_tokens;
  const outTok = metrics?.output_tokens ?? result.output_tokens;
  const latency = byId("latency");
  const cost = byId("cost");
  const calls = byId("llm_calls");
  return (
    <section className="flex flex-col gap-1.5" aria-label={t("compare.cost")}>
      <div className="grid grid-cols-2 gap-1.5">
        <MetricTile
          label={t("meta.latency")}
          value={text(latency, totalLatency(result))}
          isBest={best.latency?.has(result.strategy)}
        />
        <MetricTile
          label={t("meta.tokens")}
          value={t("compare.tokensInOut", { input: formatTokens(inTok, locale), output: formatTokens(outTok, locale) })}
          isBest={best.tokens?.has(result.strategy)}
        />
        {metrics && (
          <MetricTile
            label={t(cost.label)}
            value={text(cost, cost.value(result))}
            isBest={best.cost?.has(result.strategy)}
          />
        )}
        {metrics && (
          <MetricTile
            label={t(calls.label)}
            value={text(calls, calls.value(result))}
            isBest={best.llm_calls?.has(result.strategy)}
          />
        )}
      </div>
      <LatencyBar result={result} />
    </section>
  );
}

/**
 * Stacked bar of the per-request latency stages. The legend beneath lists every
 * value, so nothing depends on hovering or on colour alone.
 */
function LatencyBar({ result }: { result: StrategyOut }) {
  const { t, locale } = useI18n();
  const stages = latencyStages(result);
  if (!stages || stages.length === 0) return null;
  const total = stages.reduce((s, x) => s + x.ms, 0);
  const label = (stage: string) => t(`compare.stage.${stage}` as MessageKey);
  return (
    <figure className="flex flex-col gap-1.5">
      <figcaption className="text-[11px] text-zinc-500">{t("compare.latencyBreakdown")}</figcaption>
      <div className="flex h-2.5 w-full gap-[2px]" aria-hidden="true">
        {stages.map(({ stage, ms }, i) => (
          <div
            key={stage}
            title={`${label(stage)}: ${formatLatency(ms, locale)}`}
            className={`h-full ${i === 0 ? "rounded-l-[4px]" : ""} ${i === stages.length - 1 ? "rounded-r-[4px]" : ""}`}
            style={{ width: `${(ms / total) * 100}%`, minWidth: 2, backgroundColor: STAGE_COLORS[stage] }}
          />
        ))}
      </div>
      <ul className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-zinc-400">
        {stages.map(({ stage, ms }) => (
          <li key={stage} className="inline-flex items-center gap-1">
            <span aria-hidden="true" className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: STAGE_COLORS[stage] }} />
            {label(stage)} <span className="font-mono tabular-nums text-zinc-300">{formatLatency(ms, locale)}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

function WhyPanel({ result }: { result: StrategyOut }) {
  const { t, tp } = useI18n();
  const facts = whyThisResult(result);
  const render = (f: Fact) => ("plural" in f ? tp(f.plural, f.count, f.vars) : t(f.key, f.vars));
  return (
    <details className="text-xs rounded-md border border-bg-border bg-bg-elevated/60 px-2.5 py-2">
      <summary className="cursor-pointer select-none font-semibold text-zinc-300 hover:text-white">
        {t("why.title")}
      </summary>
      <p className="mt-1.5 text-[11px] text-zinc-500">{t("why.intro")}</p>
      <ul className="mt-1.5 space-y-1 list-disc pl-4 text-zinc-300">
        {facts.map((f, i) => (
          <li key={i}>{render(f)}</li>
        ))}
      </ul>
    </details>
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
        {t("compare.adminKey")} {stored ? t("compare.adminSaved") : t("compare.adminNeededHint")}
      </summary>
      <form onSubmit={save} className="mt-2 flex gap-2 flex-wrap items-center">
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={t("compare.adminPlaceholder")}
          autoComplete="off"
          spellCheck={false}
          aria-label={t("compare.adminKey")}
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
          {rich("compare.adminHelp", { header: <code className="font-mono">X-Admin-Key</code> })}
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
    <div className="flex items-center gap-2">
      <Tooltip label={t("compare.graphTip")} side="top">
        <div className={`text-xs font-mono px-2.5 py-1.5 rounded border flex items-center gap-2 cursor-help ${
          ready
            ? "bg-violet-500/10 border-violet-500/40 text-violet-300"
            : "bg-amber-500/10 border-amber-500/40 text-amber-300"
        }`}>
          <span className={`w-2 h-2 rounded-full ${ready ? "bg-violet-400" : "bg-amber-400"}`} aria-hidden="true" />
          {stats ? (
            <span>
              {rich("compare.triples", {
                count: <span className="font-semibold">{formatNumber(stats.triples)}</span>,
              })}
            </span>
          ) : (
            <span>{t("compare.graphNotBuilt")}</span>
          )}
        </div>
      </Tooltip>
      <button
        onClick={onBuild}
        disabled={busy}
        className={`text-xs px-3 py-1.5 rounded font-medium transition-colors min-h-[44px] flex items-center ${
          busy
            ? "bg-zinc-700/40 text-zinc-500 cursor-not-allowed"
            : "bg-violet-500/20 border border-violet-500/40 text-violet-300 hover:bg-violet-500/30 hover:border-violet-500/60"
        }`}
        aria-label={t("compare.buildAria")}
      >
        {busy
          ? t("compare.building", { progress: progress ?? "" }).trim()
          : ready
            ? t("compare.rebuild")
            : t("compare.buildGraph")}
      </button>
    </div>
  );
}

function FormattedAnswer({
  text,
  cite,
}: {
  text: string;
  /** When set, `[S#]` markers become superscript links to these sources. */
  cite?: { anchorPrefix: string; sources: CitedSource[] };
}) {
  const plain = (str: string, key: string): ReactNode =>
    cite ? <CitedText key={key} text={str} anchorPrefix={cite.anchorPrefix} sources={cite.sources} /> : str;

  const renderInline = (str: string): ReactNode => {
    const parts: ReactNode[] = [];
    let lastIndex = 0;

    // Match **bold**, *italic*, and `code`
    const regex = /\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`/g;
    let match;

    while ((match = regex.exec(str)) !== null) {
      if (match.index > lastIndex) {
        parts.push(plain(str.substring(lastIndex, match.index), `t-${lastIndex}`));
      }

      if (match[1]) {
        parts.push(
          <strong key={`b-${match.index}`} className="font-semibold text-zinc-100">
            {match[1]}
          </strong>
        );
      } else if (match[2]) {
        parts.push(
          <em key={`i-${match.index}`} className="italic text-zinc-200">
            {match[2]}
          </em>
        );
      } else if (match[3]) {
        parts.push(
          <code
            key={`c-${match.index}`}
            className="bg-zinc-800 px-1.5 py-0.5 rounded text-[11px] font-mono text-amber-300"
          >
            {match[3]}
          </code>
        );
      }

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < str.length) {
      parts.push(plain(str.substring(lastIndex), `t-${lastIndex}`));
    }

    return parts.length > 0 ? parts : plain(str, "t");
  };

  const lines = text.split("\n");
  const elements = [];
  let inList = false;
  let tableLines: string[] = [];

  const flushTable = () => {
    if (tableLines.length > 0) {
      const rows = tableLines.map(l => l.split("|").map(c => c.trim()).filter(c => c));
      if (rows.length > 1) {
        elements.push(
          <table key={`table-${elements.length}`} className="text-xs border-collapse border border-zinc-600 my-2">
            <thead>
              <tr className="bg-zinc-800">
                {rows[0].map((cell, i) => (
                  <th key={i} className="border border-zinc-600 px-2 py-1 text-zinc-200 font-semibold">
                    {cell}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(2).map((row, ri) => (
                <tr key={ri} className="hover:bg-zinc-800/50">
                  {row.map((cell, ci) => (
                    <td key={ci} className="border border-zinc-600 px-2 py-1 text-zinc-300">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        );
      }
      tableLines = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      flushTable();
      elements.push(<div key={`br-${i}`} className="h-2" />);
      inList = false;
      continue;
    }

    if (trimmed.startsWith("|")) {
      inList = false;
      tableLines.push(trimmed);
      continue;
    } else {
      flushTable();
    }

    if (trimmed.startsWith("##")) {
      inList = false;
      const heading = trimmed.replace(/^##\s*/, "");
      elements.push(
        <h3 key={`h3-${i}`} className="text-zinc-200 font-semibold text-sm mt-3 mb-2">
          {renderInline(heading)}
        </h3>
      );
    } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      if (!inList) {
        inList = true;
      }
      const item = trimmed.substring(2);
      elements.push(
        <li key={`li-${i}`} className="text-zinc-200 text-sm list-disc ml-4 mt-1">
          {renderInline(item)}
        </li>
      );
    } else {
      inList = false;
      elements.push(
        <p key={`p-${i}`} className="text-zinc-200 text-sm">
          {renderInline(trimmed)}
        </p>
      );
    }
  }

  flushTable();
  return <div className="space-y-2">{elements}</div>;
}

function ExtraDetails({
  name,
  extra,
}: {
  name: StrategyOut["strategy"];
  extra: Record<string, unknown>;
}) {
  const { t } = useI18n();
  if (name === "graph") {
    return (
      <details className="text-xs">
        <summary className="cursor-pointer text-zinc-500 hover:text-zinc-300 select-none">
          {t("compare.kgDetails")}
        </summary>
        <div className="mt-3 space-y-3">
          {Array.isArray(extra.entities) && extra.entities.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5 font-semibold">
                {t("compare.extractedEntities")}
              </div>
              <div className="flex flex-wrap gap-1">
                {extra.entities.map((e, i) => (
                    <span
                      key={i}
                      className="px-2 py-1 rounded-full bg-violet-500/20 border border-violet-500/40 text-violet-300 text-[10px] font-mono"
                    >
                      {String(e)}
                    </span>
                  ))}
              </div>
            </div>
          )}
          {Array.isArray(extra.related_entities) && extra.related_entities.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5 font-semibold">
                {t("compare.relatedEntities")}
              </div>
              <div className="flex flex-wrap gap-1">
                {extra.related_entities.map((e, i) => (
                    <span
                      key={i}
                      className="px-2 py-1 rounded-full bg-violet-500/10 border border-violet-500/30 text-violet-200 text-[10px]"
                    >
                      {String(e)}
                    </span>
                  ))}
              </div>
            </div>
          )}
          {typeof extra.subgraph === "string" && extra.subgraph.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5 font-semibold">
                {t("compare.kgStructure")}
              </div>
              <pre className="bg-bg-elevated rounded-md p-2 overflow-x-auto text-[10px] text-zinc-300 whitespace-pre-wrap leading-relaxed">
                {String(extra.subgraph)}
              </pre>
            </div>
          )}
        </div>
      </details>
    );
  }

  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-zinc-500 hover:text-zinc-300 select-none">
        {t("compare.strategyDetails")}
      </summary>
      <pre className="mt-2 bg-bg-elevated rounded-md p-2 overflow-x-auto text-[11px] text-zinc-300 whitespace-pre-wrap">
        {JSON.stringify(extra, null, 2)}
      </pre>
    </details>
  );
}

function TraceDetails({ trace }: { trace: TraceStep[] }) {
  const { t, tp } = useI18n();
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-zinc-500 hover:text-zinc-300 select-none font-semibold">
        🔍 {tp("compare.trace", trace.length)}
      </summary>
      <div className="mt-2 space-y-1.5 text-[10px]">
        {trace.map((step, i) => (
          <div key={i} className="flex gap-2 items-start">
            <div className="flex-shrink-0 text-zinc-600 font-mono">{String(i + 1).padStart(2, "0")}.</div>
            <div className="flex-1 bg-zinc-900/50 border border-zinc-700/50 rounded px-2 py-1">
              <div className="font-mono text-accent">{step.step ?? t("common.step")}</div>
              <div className="text-zinc-400 mt-0.5">{JSON.stringify(rest(step))}</div>
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}

function rest(t: TraceStep) {
  const r: Record<string, unknown> = {};
  for (const k of Object.keys(t)) if (k !== "step") r[k] = (t as Record<string, unknown>)[k];
  return r;
}
