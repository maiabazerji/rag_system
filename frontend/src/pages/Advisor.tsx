import { useState, type ReactNode } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  AdviseOverrides,
  AdviseResponse,
  Freshness,
  Level,
  MAX_VALIDATE_QUESTIONS,
  ProjectProfile,
  QuestionMix,
  StrategyName,
  StrategyRecommendation,
  ValidateResponse,
  advise,
  parseQuestions,
  validateStrategies,
} from "../api/advisor";
import { errorMessage } from "../api/client";
import ErrorAlert from "../components/ErrorAlert";
import LoadingSpinner from "../components/LoadingSpinner";
import { formatLatency, formatScore, formatTokens } from "../utils/formatting";
import { useI18n, type MessageKey } from "../i18n";

const META: Record<StrategyName, { title: MessageKey; text: string; bar: string; border: string }> = {
  classic: {
    title: "strategy.classic",
    text: "text-sky-300",
    bar: "bg-sky-400",
    border: "border-sky-500/40",
  },
  graph: {
    title: "strategy.graph",
    text: "text-violet-300",
    bar: "bg-violet-400",
    border: "border-violet-500/40",
  },
  agentic: {
    title: "strategy.agentic",
    text: "text-emerald-300",
    bar: "bg-emerald-400",
    border: "border-emerald-500/40",
  },
};

const QUESTION_TYPE_LABEL: Record<keyof QuestionMix, MessageKey> = {
  single_fact: "advisor.qtype.single_fact",
  relational_multi_hop: "advisor.qtype.relational_multi_hop",
  exploratory_multi_step: "advisor.qtype.exploratory_multi_step",
};

const LEVEL_LABEL: Record<Level, MessageKey> = {
  low: "advisor.level.low",
  medium: "advisor.level.medium",
  high: "advisor.level.high",
};

const FRESHNESS_LABEL: Record<Freshness, MessageKey> = {
  static: "advisor.freshness.static",
  monthly: "advisor.freshness.monthly",
  weekly: "advisor.freshness.weekly",
  daily: "advisor.freshness.daily",
  realtime: "advisor.freshness.realtime",
};

const SIZE_LABEL: Record<ProjectProfile["corpus_size"], MessageKey> = {
  tiny: "advisor.size.tiny",
  small: "advisor.size.small",
  medium: "advisor.size.medium",
  large: "advisor.size.large",
  xlarge: "advisor.size.xlarge",
};

type OverrideForm = {
  corpus_size_docs: string;
  languages: string;
  latency_budget_ms: string;
  cost_sensitivity: "" | Level;
  data_freshness: "" | Freshness;
  entity_richness: "" | Level;
  compliance: string;
  question_examples: string;
};

const EMPTY_OVERRIDES: OverrideForm = {
  corpus_size_docs: "",
  languages: "",
  latency_budget_ms: "",
  cost_sensitivity: "",
  data_freshness: "",
  entity_richness: "",
  compliance: "",
  question_examples: "",
};

function splitList(s: string, sep: RegExp = /[,;]/): string[] {
  return s
    .split(sep)
    .map((x) => x.trim())
    .filter(Boolean);
}

function toOverrides(f: OverrideForm): AdviseOverrides {
  return {
    corpus_size_docs: f.corpus_size_docs ? Number(f.corpus_size_docs) : undefined,
    languages: splitList(f.languages),
    latency_budget_ms: f.latency_budget_ms ? Number(f.latency_budget_ms) : undefined,
    cost_sensitivity: f.cost_sensitivity || undefined,
    data_freshness: f.data_freshness || undefined,
    entity_richness: f.entity_richness || undefined,
    compliance: splitList(f.compliance),
    question_examples: splitList(f.question_examples, /\r?\n/),
  };
}

export default function Advisor() {
  const { t, rich } = useI18n();
  const [description, setDescription] = useState("");
  const [showOverrides, setShowOverrides] = useState(false);
  const [form, setForm] = useState<OverrideForm>(EMPTY_OVERRIDES);

  const adviseMut = useMutation<AdviseResponse, Error, void>({
    mutationFn: () => advise(description.trim(), toOverrides(form)),
  });

  const set = <K extends keyof OverrideForm>(k: K, v: OverrideForm[K]) =>
    setForm((f) => ({ ...f, [k]: v }));

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-4xl font-semibold text-white">{t("advisor.title")}</h1>
        <p className="text-sm text-zinc-400">
          {rich("advisor.intro", {
            classic: <span className="text-sky-300">{t("strategy.classic.short")}</span>,
            graph: <span className="text-violet-300">{t("strategy.graph.short")}</span>,
            agentic: <span className="text-emerald-300">{t("strategy.agentic.short")}</span>,
          })}
        </p>
      </header>

      <form
        className="card flex flex-col gap-3 p-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (description.trim()) adviseMut.mutate();
        }}
      >
        <label htmlFor="advisor-description" className="text-sm font-medium text-zinc-200">
          {t("advisor.yourProject")}
        </label>
        <textarea
          id="advisor-description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={6}
          placeholder={t("advisor.placeholder")}
          className="input resize-y text-sm"
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-zinc-500">
            {t("advisor.anyLanguage")}{" "}
            <button
              type="button"
              onClick={() => setDescription(t("advisor.example"))}
              className="text-accent hover:text-accent-hover underline-offset-2 hover:underline"
            >
              {t("advisor.useExample")}
            </button>
          </p>
          <button
            type="button"
            onClick={() => setShowOverrides((v) => !v)}
            className="text-xs text-zinc-400 hover:text-white"
            aria-expanded={showOverrides}
          >
            {showOverrides ? "▾" : "▸"} {t("advisor.knownFacts")}
          </button>
        </div>

        {showOverrides && (
          <div className="grid gap-3 sm:grid-cols-2 border-t border-bg-border pt-3">
            <Field label={t("advisor.field.corpusSize")}>
              <input
                type="number"
                min={0}
                value={form.corpus_size_docs}
                onChange={(e) => set("corpus_size_docs", e.target.value)}
                className="input text-sm !py-2"
                placeholder={t("advisor.eg", { value: "20000" })}
              />
            </Field>
            <Field label={t("advisor.field.languages")}>
              <input
                value={form.languages}
                onChange={(e) => set("languages", e.target.value)}
                className="input text-sm !py-2"
                placeholder={t("advisor.eg", { value: "fr, en" })}
              />
            </Field>
            <Field label={t("advisor.field.latency")}>
              <input
                type="number"
                min={100}
                value={form.latency_budget_ms}
                onChange={(e) => set("latency_budget_ms", e.target.value)}
                className="input text-sm !py-2"
                placeholder={t("advisor.eg", { value: "5000" })}
              />
            </Field>
            <Field label={t("advisor.field.cost")}>
              <Select
                value={form.cost_sensitivity}
                onChange={(v) => set("cost_sensitivity", v as OverrideForm["cost_sensitivity"])}
                options={LEVEL_LABEL}
              />
            </Field>
            <Field label={t("advisor.field.freshness")}>
              <Select
                value={form.data_freshness}
                onChange={(v) => set("data_freshness", v as OverrideForm["data_freshness"])}
                options={FRESHNESS_LABEL}
              />
            </Field>
            <Field label={t("advisor.field.entities")}>
              <Select
                value={form.entity_richness}
                onChange={(v) => set("entity_richness", v as OverrideForm["entity_richness"])}
                options={LEVEL_LABEL}
              />
            </Field>
            <Field label={t("advisor.field.compliance")}>
              <input
                value={form.compliance}
                onChange={(e) => set("compliance", e.target.value)}
                className="input text-sm !py-2"
                placeholder={t("advisor.egCompliance")}
              />
            </Field>
            <Field label={t("advisor.field.examples")}>
              <textarea
                value={form.question_examples}
                onChange={(e) => set("question_examples", e.target.value)}
                rows={2}
                className="input text-sm !py-2 resize-y"
              />
            </Field>
          </div>
        )}

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={!description.trim() || adviseMut.isPending}
            className="btn-primary text-sm"
          >
            {adviseMut.isPending ? t("advisor.analysing") : t("advisor.recommend")}
          </button>
        </div>
        {adviseMut.isError && (
          <ErrorAlert
            error={errorMessage(adviseMut.error)}
            onRetry={() => adviseMut.mutate()}
            onDismiss={() => adviseMut.reset()}
          />
        )}
      </form>

      {adviseMut.isPending && <LoadingSpinner message={t("advisor.loading")} />}
      {adviseMut.data && <AdviceResult result={adviseMut.data} />}
      {adviseMut.data && <ValidateSection advice={adviseMut.data} />}
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-zinc-400">
      {label}
      {children}
    </label>
  );
}

/** A select whose option values stay the API's, labelled in the UI language. */
function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: Record<string, MessageKey>;
}) {
  const { t } = useI18n();
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="input text-sm !py-2">
      <option value="">{t("advisor.fromDescription")}</option>
      {Object.entries(options).map(([o, label]) => (
        <option key={o} value={o}>
          {t(label)}
        </option>
      ))}
    </select>
  );
}

function AdviceResult({ result }: { result: AdviseResponse }) {
  const { t, locale, formatNumber } = useI18n();
  const p = result.profile;
  const mix = p.question_mix;
  const label = <T extends string>(map: Record<T, MessageKey>, v: T) => (map[v] ? t(map[v]) : v);
  return (
    <section className="flex flex-col gap-4">
      <div className="card p-4 flex flex-col gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold text-white">{t("advisor.profile")}</h2>
          <span className="chip" title={t("advisor.profileSourceTitle")}>
            {p.source === "llm" ? t("advisor.sourceLlm") : t("advisor.sourceHeuristic")}
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5 text-xs">
          <span className="chip">
            {t("advisor.chip.corpus", { size: label(SIZE_LABEL, p.corpus_size) })}
            {p.corpus_size_docs != null
              ? t("advisor.chip.docs", { count: formatNumber(p.corpus_size_docs) })
              : ""}
          </span>
          <span className="chip">
            {t("advisor.chip.languages", { list: p.languages.join(", ") || "?" })}
          </span>
          <span className="chip">
            {t("advisor.chip.latency", {
              value:
                p.latency_budget_ms != null
                  ? formatLatency(p.latency_budget_ms, locale)
                  : t("advisor.notStated"),
            })}
          </span>
          <span className="chip">
            {t("advisor.chip.cost", { value: label(LEVEL_LABEL, p.cost_sensitivity) })}
          </span>
          <span className="chip">
            {t("advisor.chip.freshness", { value: label(FRESHNESS_LABEL, p.data_freshness) })}
          </span>
          <span className="chip">
            {t("advisor.chip.entities", { value: label(LEVEL_LABEL, p.entity_richness) })}
          </span>
          {p.document_types.map((d) => (
            <span key={d} className="chip">
              {d}
            </span>
          ))}
          {p.compliance.map((c) => (
            <span key={c} className="chip !border-amber-500/40 !text-amber-200">
              {c}
            </span>
          ))}
        </div>
        <div className="text-xs text-zinc-400">
          {t("advisor.mix", {
            single: formatNumber(mix.single_fact),
            relational: formatNumber(mix.relational_multi_hop),
            exploratory: formatNumber(mix.exploratory_multi_step),
          })}
          {p.overridden_fields.length > 0 && (
            <span className="text-zinc-500">
              {t("advisor.overridden", { fields: p.overridden_fields.join(", ") })}
            </span>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-3">
        {result.recommendations.map((r) => (
          <RecommendationCard key={r.strategy} rec={r} top={r.strategy === result.top_strategy} />
        ))}
      </div>

      {result.hybrid_routing.recommended && (
        <div className="card p-4 flex flex-col gap-2 !border-accent/40">
          <h3 className="text-sm font-semibold text-white">{t("advisor.hybrid")}</h3>
          <p className="text-sm text-zinc-300">{result.hybrid_routing.rationale}</p>
          <ul className="text-sm text-zinc-300 space-y-0.5">
            {result.hybrid_routing.routes.map((route) => (
              <li key={route.question_type}>
                {label(QUESTION_TYPE_LABEL, route.question_type)} (
                {formatNumber(route.share_pct / 100, { style: "percent" })}) →{" "}
                <span className={META[route.strategy].text}>{t(META[route.strategy].title)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.compliance_notes.length > 0 && (
        <div className="card p-4 flex flex-col gap-2 !border-amber-500/30">
          <h3 className="text-sm font-semibold text-white">{t("advisor.notes")}</h3>
          <ul className="list-disc pl-5 text-sm text-zinc-300 space-y-1">
            {result.compliance_notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="card p-4 text-sm text-zinc-300 !bg-accent-soft/40 !border-accent/30">
        <span className="font-semibold text-accent">{t("advisor.nextStep")}</span>
        {result.next_step.summary}
      </div>
    </section>
  );
}

function RecommendationCard({ rec, top }: { rec: StrategyRecommendation; top: boolean }) {
  const { t, formatNumber } = useI18n();
  const m = META[rec.strategy];
  const c = rec.suggested_config;
  const title = t(m.title);
  return (
    <article className={`card p-4 flex flex-col gap-3 ${top ? m.border : ""}`}>
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono text-zinc-500">#{rec.rank}</span>
        <h3 className={`font-semibold ${m.text}`}>{title}</h3>
        {top && <span className="chip !text-accent !border-accent/40">{t("advisor.recommended")}</span>}
        <span className="ml-auto font-mono text-sm text-white">{formatNumber(rec.score)}/100</span>
      </div>
      <div
        className="h-2 w-full rounded-full bg-bg-elevated overflow-hidden"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={rec.score}
        aria-label={t("advisor.scoreAria", { title })}
      >
        <div className={`h-full ${m.bar} transition-all`} style={{ width: `${rec.score}%` }} />
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <h4 className="text-xs uppercase tracking-wider text-zinc-500 mb-1">{t("advisor.why")}</h4>
          <ul className="list-disc pl-5 text-sm text-zinc-300 space-y-1">
            {rec.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="text-xs uppercase tracking-wider text-zinc-500 mb-1">{t("advisor.tradeoffs")}</h4>
          <ul className="list-disc pl-5 text-sm text-zinc-400 space-y-1">
            {rec.tradeoffs.map((tradeoff) => (
              <li key={tradeoff}>{tradeoff}</li>
            ))}
          </ul>
        </div>
      </div>

      <details className="text-sm">
        <summary className="cursor-pointer text-xs text-zinc-400 hover:text-white">
          {t("advisor.suggestedConfig")}
        </summary>
        <div className="mt-2 grid gap-x-6 gap-y-1 sm:grid-cols-2 font-mono text-xs text-zinc-300">
          <div>RETRIEVAL_TOP_K = {c.retrieval_top_k}</div>
          <div>RERANK_TOP_K = {c.rerank_top_k}</div>
          <div>rerank = {c.rerank ? t("advisor.on") : t("advisor.off")}</div>
          <div>CHUNK_SIZE_TOKENS = {c.chunk_size_tokens}</div>
          <div>CHUNK_OVERLAP_TOKENS = {c.chunk_overlap_tokens}</div>
          <div className="break-all">EMBEDDING_MODEL = {c.embedding_model}</div>
          {Object.entries(c.models).map(([stage, model]) => (
            <div key={stage} className="break-all">
              {stage}: {model}
            </div>
          ))}
          <div>
            {t("advisor.relative", {
              cost: formatNumber(c.expected_relative_cost),
              latency: formatNumber(c.expected_relative_latency),
            })}
          </div>
        </div>
        {c.notes.length > 0 && (
          <ul className="mt-2 list-disc pl-5 text-xs text-zinc-400 space-y-1">
            {c.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        )}
      </details>
    </article>
  );
}

function ValidateSection({ advice }: { advice: AdviseResponse }) {
  const { t, tp, rich } = useI18n();
  const [text, setText] = useState(() =>
    advice.profile.example_questions.slice(0, 5).join("\n"),
  );
  const [strategies, setStrategies] = useState<StrategyName[]>(advice.next_step.suggested_strategies);
  const lineCount = text.split(/\r?\n/).filter((l) => l.trim()).length;
  const questions = parseQuestions(text);

  const mut = useMutation<ValidateResponse, Error, void>({
    mutationFn: () => validateStrategies(questions, strategies, advice.profile),
  });

  const toggle = (s: StrategyName) =>
    setStrategies((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]));

  return (
    <section className="card p-4 flex flex-col gap-3">
      <div>
        <h2 className="text-lg font-semibold text-white">{t("advisor.validate")}</h2>
        <p className="text-xs text-zinc-500 mt-1">
          {rich("advisor.validateHelp", {
            syntax: <code className="font-mono text-zinc-300">{t("advisor.validateSyntax")}</code>,
            max: MAX_VALIDATE_QUESTIONS,
          })}
        </p>
      </div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        className="input resize-y text-sm font-mono"
        placeholder={t("advisor.validatePlaceholder")}
      />
      {lineCount > MAX_VALIDATE_QUESTIONS && (
        <p className="text-xs text-amber-300">
          {t("advisor.tooMany", { max: MAX_VALIDATE_QUESTIONS, count: lineCount })}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {(["classic", "graph", "agentic"] as StrategyName[]).map((s) => (
          <label key={s} className="chip cursor-pointer">
            <input
              type="checkbox"
              checked={strategies.includes(s)}
              onChange={() => toggle(s)}
              className="accent-amber-400"
            />
            <span className={META[s].text}>{t(META[s].title)}</span>
          </label>
        ))}
        <button
          onClick={() => mut.mutate()}
          disabled={questions.length === 0 || strategies.length === 0 || mut.isPending}
          className="btn-primary text-sm ml-auto"
        >
          {mut.isPending ? t("common.running") : tp("advisor.runQuestions", questions.length)}
        </button>
      </div>
      {mut.isError && (
        <ErrorAlert
          error={errorMessage(mut.error)}
          onRetry={() => mut.mutate()}
          onDismiss={() => mut.reset()}
        />
      )}
      {mut.isPending && (
        <LoadingSpinner
          message={t("advisor.validating", {
            questions: questions.length,
            strategies: strategies.length,
          })}
        />
      )}
      {mut.data && <Scorecard data={mut.data} />}
    </section>
  );
}

function Scorecard({ data }: { data: ValidateResponse }) {
  const { t, locale } = useI18n();
  const fmtScore = (v: number | null) => (v == null ? "–" : formatScore(v, 2, locale));
  return (
    <div className="flex flex-col gap-3">
      <div className="text-sm">
        {data.measured_winner ? (
          <>
            <span className="text-zinc-400">{t("advisor.measuredWinner")}</span>
            <span className={`font-semibold ${META[data.measured_winner].text}`}>
              {t(META[data.measured_winner].title)}
            </span>
          </>
        ) : (
          <span className="text-amber-300">{t("advisor.noWinner")}</span>
        )}
        <p className="text-xs text-zinc-500 mt-1">{data.winner_reason}</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wider text-zinc-500 border-b border-bg-border">
              <th className="py-2 pr-3">{t("advisor.col.strategy")}</th>
              <th className="py-2 pr-3">{t("advisor.col.refusals")}</th>
              <th className="py-2 pr-3">{t("advisor.col.errors")}</th>
              <th className="py-2 pr-3">{t("advisor.col.latency")}</th>
              <th className="py-2 pr-3">{t("advisor.col.tokens")}</th>
              <th className="py-2 pr-3">{t("advisor.col.judge")}</th>
              <th className="py-2 pr-3">{t("advisor.col.f1")}</th>
            </tr>
          </thead>
          <tbody>
            {data.scorecard.map((c) => (
              <tr
                key={c.strategy}
                className={`border-b border-bg-border/60 ${c.strategy === data.measured_winner ? "bg-bg-elevated" : ""}`}
              >
                <td className={`py-2 pr-3 font-medium ${META[c.strategy].text}`}>{t(META[c.strategy].title)}</td>
                <td className="py-2 pr-3 font-mono">
                  {c.refusals}/{c.questions}
                </td>
                <td className="py-2 pr-3 font-mono">{c.errors}</td>
                <td className="py-2 pr-3 font-mono">{formatLatency(c.avg_latency_ms, locale)}</td>
                <td className="py-2 pr-3 font-mono">
                  {formatTokens(Math.round(c.avg_tokens_per_question), locale)}
                </td>
                <td className="py-2 pr-3 font-mono">
                  {fmtScore(c.avg_judge_score)}
                  {c.judged > 0 && <span className="text-zinc-500"> (n={c.judged})</span>}
                </td>
                <td className="py-2 pr-3 font-mono">{fmtScore(c.avg_answer_f1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <details>
        <summary className="cursor-pointer text-xs text-zinc-400 hover:text-white">
          {t("advisor.perQuestion", { count: data.rows.length })}
        </summary>
        <ul className="mt-2 flex flex-col gap-2">
          {data.rows.map((r, i) => (
            <li key={i} className="border border-bg-border rounded-lg p-2 text-xs">
              <div className="flex flex-wrap gap-2 items-center">
                <span className={META[r.strategy].text}>{t(META[r.strategy].title)}</span>
                <span className="text-zinc-300">{r.question}</span>
                {r.refusal && (
                  <span className="chip !text-rose-300">
                    {r.error ? t("common.error") : t("advisor.refused")}
                  </span>
                )}
                <span className="ml-auto font-mono text-zinc-500">
                  {formatLatency(r.latency_ms, locale)} ·{" "}
                  {formatTokens(r.input_tokens + r.output_tokens, locale)}
                </span>
              </div>
              <p className="mt-1 text-zinc-400 whitespace-pre-wrap">{r.answer}</p>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
