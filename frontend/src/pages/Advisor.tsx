/**
 * The Advisor: the app's front door. A client describes their project, answers
 * a few follow-up questions (only the ones that could change the outcome), and
 * gets a recommended RAG strategy with the reasons and a confidence level.
 * Testing on their own documents is an optional last step.
 *
 * Answers are kept as overrides and re-sent with the description, so every
 * recommendation is recomputed from everything the client said.
 */
import { useRef, useState, type ReactNode } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  AdviseOverrides,
  AdviseResponse,
  Confidence,
  FollowUpQuestion,
  Freshness,
  Level,
  MAX_VALIDATE_QUESTIONS,
  ProjectProfile,
  QuestionMix,
  StrategyName,
  StrategyRecommendation,
  ValidateResponse,
  advise,
  deleteWorkspace,
  newWorkspace,
  parseQuestions,
  validateStrategies,
} from "../api/advisor";
import { errorMessage, upload } from "../api/client";
import ErrorAlert from "../components/ErrorAlert";
import LoadingSpinner from "../components/LoadingSpinner";
import { formatLatency, formatScore, formatTokens } from "../utils/formatting";
import { useI18n, type MessageKey } from "../i18n";
import { getBrowserKey, setBrowserKey } from "../advisor-engine/llm";
import { isStaticDemo } from "../staticDemo";

const META: Record<
  StrategyName,
  { title: MessageKey; plain: MessageKey; text: string; bar: string; border: string }
> = {
  classic: {
    title: "strategy.classic",
    plain: "advisor.plain.classic",
    text: "text-sky-300",
    bar: "bg-sky-400",
    border: "border-sky-500/40",
  },
  graph: {
    title: "strategy.graph",
    plain: "advisor.plain.graph",
    text: "text-violet-300",
    bar: "bg-violet-400",
    border: "border-violet-500/40",
  },
  agentic: {
    title: "strategy.agentic",
    plain: "advisor.plain.agentic",
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

const CONFIDENCE_STYLE: Record<Level, { label: MessageKey; cls: string }> = {
  high: { label: "advisor.conf.high", cls: "!text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10" },
  medium: { label: "advisor.conf.medium", cls: "!text-amber-300 !border-amber-500/40 !bg-amber-500/10" },
  low: { label: "advisor.conf.low", cls: "!text-rose-300 !border-rose-500/40 !bg-rose-500/10" },
};

type Step = "describe" | "questions" | "result";
const STEPS: { id: Step | "test"; label: MessageKey }[] = [
  { id: "describe", label: "advisor.step.describe" },
  { id: "questions", label: "advisor.step.questions" },
  { id: "result", label: "advisor.step.result" },
  { id: "test", label: "advisor.step.test" },
];

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

/** Chosen option id per follow-up question id. */
type Answers = Record<string, string>;

/** Overrides for the answers given, using the options the API offered. */
function answersToOverrides(answers: Answers, asked: FollowUpQuestion[]): AdviseOverrides {
  const out: AdviseOverrides = {};
  for (const q of asked) {
    const option = q.options.find((o) => o.id === answers[q.id]);
    if (option) Object.assign(out, option.overrides);
  }
  return out;
}

export default function Advisor() {
  const { t, lang } = useI18n();
  const [step, setStep] = useState<Step>("describe");
  const [description, setDescription] = useState("");
  const [showOverrides, setShowOverrides] = useState(false);
  const [form, setForm] = useState<OverrideForm>(EMPTY_OVERRIDES);
  // Every question asked so far (any round) and the answers given to them.
  const [asked, setAsked] = useState<FollowUpQuestion[]>([]);
  const [answers, setAnswers] = useState<Answers>({});

  const adviseMut = useMutation<AdviseResponse, Error, { answers: Answers; asked: FollowUpQuestion[] }>({
    mutationFn: ({ answers: a, asked: q }) =>
      advise(description.trim(), { ...toOverrides(form), ...answersToOverrides(a, q) }, lang),
  });
  const result = adviseMut.data;

  const set = <K extends keyof OverrideForm>(k: K, v: OverrideForm[K]) =>
    setForm((f) => ({ ...f, [k]: v }));

  /** First submission: read the description, then ask what is still open. */
  function start() {
    if (!description.trim()) return;
    setAsked([]);
    setAnswers({});
    adviseMut.mutate(
      { answers: {}, asked: [] },
      {
        onSuccess: (data) => {
          const questions = data.follow_up_questions ?? [];
          setAsked(questions);
          // Pre-select what the description already suggested; the client can change it.
          const guessed: Answers = {};
          for (const q of questions) if (q.current) guessed[q.id] = q.current;
          setAnswers(guessed);
          setStep(questions.length > 0 ? "questions" : "result");
        },
      },
    );
  }

  /** Re-score with the answers so far and show the recommendation. */
  function recommend(nextAnswers: Answers, nextAsked: FollowUpQuestion[]) {
    setAnswers(nextAnswers);
    setAsked(nextAsked);
    adviseMut.mutate({ answers: nextAnswers, asked: nextAsked }, { onSuccess: () => setStep("result") });
  }

  function startOver() {
    adviseMut.reset();
    setAsked([]);
    setAnswers({});
    setStep("describe");
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-3xl sm:text-4xl font-semibold text-white">{t("advisor.title")}</h1>
        <p className="text-sm text-zinc-400 max-w-2xl">{t("advisor.intro")}</p>
      </header>

      <Stepper current={step} />

      {step === "describe" && (
        <form
          className="card flex flex-col gap-3 p-4"
          onSubmit={(e) => {
            e.preventDefault();
            start();
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

          {showOverrides && <OverrideFields form={form} set={set} />}

          {isStaticDemo() && <BrowserKeyField />}

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={!description.trim() || adviseMut.isPending}
              className="btn-primary text-sm"
            >
              {adviseMut.isPending ? t("advisor.analysing") : t("advisor.continue")}
            </button>
          </div>
          {adviseMut.isError && (
            <ErrorAlert
              error={errorMessage(adviseMut.error)}
              onRetry={start}
              onDismiss={() => adviseMut.reset()}
            />
          )}
        </form>
      )}

      {step === "questions" && (
        <section className="card p-4 sm:p-5 flex flex-col gap-5" aria-labelledby="advisor-questions-title">
          <div>
            <h2 id="advisor-questions-title" className="text-lg font-semibold text-white">
              {t("advisor.questionsTitle")}
            </h2>
            <p className="text-sm text-zinc-400 mt-1">{t("advisor.questionsHelp")}</p>
          </div>
          {asked.map((q) => (
            <FollowUp
              key={q.id}
              question={q}
              value={answers[q.id]}
              onChange={(v) => setAnswers((a) => ({ ...a, [q.id]: v }))}
            />
          ))}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <button type="button" onClick={() => setStep("describe")} className="btn-ghost text-sm">
              {t("advisor.back")}
            </button>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => recommend({}, asked)}
                disabled={adviseMut.isPending}
                className="btn-ghost text-sm"
              >
                {t("advisor.skip")}
              </button>
              <button
                type="button"
                onClick={() => recommend(answers, asked)}
                disabled={adviseMut.isPending}
                className="btn-primary text-sm"
              >
                {adviseMut.isPending ? t("advisor.analysing") : t("advisor.seeResult")}
              </button>
            </div>
          </div>
          {adviseMut.isError && (
            <ErrorAlert
              error={errorMessage(adviseMut.error)}
              onRetry={() => recommend(answers, asked)}
              onDismiss={() => adviseMut.reset()}
            />
          )}
        </section>
      )}

      {adviseMut.isPending && step !== "describe" && step !== "questions" && (
        <LoadingSpinner message={t("advisor.loading")} />
      )}

      {step === "result" && result && (
        <>
          <AdviceResult
            result={result}
            answered={answers}
            busy={adviseMut.isPending}
            onRefine={(id, option, question) =>
              recommend(
                { ...answers, [id]: option },
                asked.some((q) => q.id === id) ? asked : [...asked, question],
              )
            }
            onEdit={() => setStep(asked.length > 0 ? "questions" : "describe")}
            onStartOver={startOver}
          />
          {isStaticDemo() ? (
            <p className="card p-4 text-sm text-zinc-400">{t("static.testNote")}</p>
          ) : (
            <ValidateSection key={result.top_strategy} advice={result} />
          )}
        </>
      )}
    </div>
  );
}

/**
 * Static demo only: an optional Anthropic key, so Claude (not the keyword
 * rules) reads the description. Kept in this tab's sessionStorage and sent
 * only to Anthropic, straight from the browser.
 */
function BrowserKeyField() {
  const { t } = useI18n();
  const [key, setKey] = useState(() => getBrowserKey() ?? "");
  return (
    <label className="flex flex-col gap-1 text-xs text-zinc-400 border-t border-bg-border pt-3">
      {t("static.keyLabel")}
      <input
        type="password"
        value={key}
        onChange={(e) => {
          setKey(e.target.value);
          setBrowserKey(e.target.value);
        }}
        autoComplete="off"
        spellCheck={false}
        placeholder="sk-ant-…"
        className="input text-sm !py-2 font-mono"
      />
      <span className="text-zinc-500">{t("static.keyHelp")}</span>
    </label>
  );
}

function Stepper({ current }: { current: Step }) {
  const { t } = useI18n();
  const index = STEPS.findIndex((s) => s.id === current);
  return (
    <ol aria-label={t("advisor.stepsAria")} className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
      {STEPS.map((s, i) => {
        const state = i < index ? "done" : i === index ? "current" : "todo";
        return (
          <li key={s.id} className="flex items-center gap-2" aria-current={state === "current" ? "step" : undefined}>
            <span
              className={`inline-flex h-6 w-6 items-center justify-center rounded-full border font-mono ${
                state === "current"
                  ? "border-accent bg-accent text-bg"
                  : state === "done"
                    ? "border-accent/60 text-accent"
                    : "border-bg-border text-zinc-500"
              }`}
            >
              {state === "done" ? "✓" : i + 1}
            </span>
            <span className={state === "current" ? "text-white font-medium" : "text-zinc-500"}>{t(s.label)}</span>
            {i < STEPS.length - 1 && <span aria-hidden="true" className="text-zinc-700">—</span>}
          </li>
        );
      })}
    </ol>
  );
}

/** One follow-up question as a set of radio cards. */
function FollowUp({
  question,
  value,
  onChange,
}: {
  question: FollowUpQuestion;
  value: string | undefined;
  onChange: (v: string) => void;
}) {
  const { t } = useI18n();
  const name = `fu-${question.id}`;
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="text-sm font-medium text-zinc-100 mb-2 flex flex-wrap items-center gap-2">
        {t(`fu.${question.id}` as MessageKey)}
        {question.changes_recommendation && (
          <span className="chip !text-[10px] !py-0.5 !text-accent !border-accent/40">{t("advisor.flips")}</span>
        )}
      </legend>
      <div className="grid gap-2 sm:grid-cols-2">
        {question.options.map((o) => {
          const checked = value === o.id;
          return (
            <label
              key={o.id}
              className={`flex items-start gap-2 rounded-lg border px-3 py-2.5 text-sm cursor-pointer transition-colors min-h-[44px] focus-within:ring-2 focus-within:ring-accent/50 ${
                checked
                  ? "border-accent/60 bg-accent/10 text-white"
                  : "border-bg-border text-zinc-300 hover:border-zinc-600 hover:text-white"
              }`}
            >
              <input
                type="radio"
                name={name}
                value={o.id}
                checked={checked}
                onChange={() => onChange(o.id)}
                className="mt-0.5 accent-[#e8a93c]"
              />
              <span className="flex-1">
                {t(`fu.${question.id}.${o.id}` as MessageKey)}
                {question.current === o.id && (
                  <span className="ml-1.5 text-[11px] text-zinc-500">({t("advisor.ourGuess")})</span>
                )}
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

function OverrideFields({
  form,
  set,
}: {
  form: OverrideForm;
  set: <K extends keyof OverrideForm>(k: K, v: OverrideForm[K]) => void;
}) {
  const { t } = useI18n();
  return (
    <div className="grid gap-3 sm:grid-cols-2 border-t border-bg-border pt-3">
      <Field label={t("advisor.field.corpusSize")}>
        <input
          type="number"
          min={0}
          value={form.corpus_size_docs}
          onChange={(e) => set("corpus_size_docs", e.target.value)}
          className="input text-sm !py-2"
        />
      </Field>
      <Field label={t("advisor.field.languages")}>
        <input
          value={form.languages}
          onChange={(e) => set("languages", e.target.value)}
          className="input text-sm !py-2"
        />
      </Field>
      <Field label={t("advisor.field.latency")}>
        <input
          type="number"
          min={100}
          value={form.latency_budget_ms}
          onChange={(e) => set("latency_budget_ms", e.target.value)}
          className="input text-sm !py-2"
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

function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  const { t, tp } = useI18n();
  const style = CONFIDENCE_STYLE[confidence.level];
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className={`chip ${style.cls}`}>{t(style.label)}</span>
      <span className="text-zinc-400">
        {t("advisor.confMargin", { margin: confidence.margin })}
        {confidence.open_questions > 0 && <> {tp("advisor.confOpen", confidence.open_questions)}</>}
      </span>
    </div>
  );
}

function AdviceResult({
  result,
  answered,
  busy,
  onRefine,
  onEdit,
  onStartOver,
}: {
  result: AdviseResponse;
  answered: Answers;
  busy: boolean;
  onRefine: (id: string, option: string, question: FollowUpQuestion) => void;
  onEdit: () => void;
  onStartOver: () => void;
}) {
  const { t } = useI18n();
  const top = result.recommendations.find((r) => r.strategy === result.top_strategy) ?? result.recommendations[0];
  const others = result.recommendations.filter((r) => r !== top);
  const m = META[top.strategy];
  // Questions that could still flip the pick and that the client has not answered.
  const refine = (result.follow_up_questions ?? []).filter(
    (q) => q.changes_recommendation && !(q.id in answered),
  );
  const label = <T extends string>(map: Record<T, MessageKey>, v: T) => (map[v] ? t(map[v]) : v);

  return (
    <section className="flex flex-col gap-4" aria-live="polite">
      <article className={`card p-5 flex flex-col gap-4 ${m.border}`}>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-wider text-zinc-500">{t("advisor.resultTitle")}</div>
            <h2 className={`display text-3xl font-semibold mt-1 ${m.text}`}>{t(m.title)}</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={onEdit} className="btn-ghost text-xs">
              {t("advisor.editAnswers")}
            </button>
            <button type="button" onClick={onStartOver} className="btn-ghost text-xs">
              {t("advisor.startOver")}
            </button>
          </div>
        </div>
        <p className="text-zinc-200 leading-relaxed">{t(m.plain)}</p>
        {result.confidence && <ConfidenceBadge confidence={result.confidence} />}
        <div>
          <h3 className="text-xs uppercase tracking-wider text-zinc-500 mb-1">{t("advisor.why")}</h3>
          <ul className="list-disc pl-5 text-sm text-zinc-300 space-y-1">
            {top.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
        <details className="text-sm">
          <summary className="cursor-pointer text-xs text-zinc-400 hover:text-white">{t("advisor.technical")}</summary>
          <div className="mt-3">
            <RecommendationCard rec={top} top />
          </div>
        </details>
      </article>

      {refine.length > 0 && (
        <section className="card p-4 flex flex-col gap-3 !border-accent/40" aria-labelledby="advisor-refine-title">
          <div>
            <h3 id="advisor-refine-title" className="text-sm font-semibold text-white">{t("advisor.refineTitle")}</h3>
            <p className="text-xs text-zinc-400 mt-1">{t("advisor.refineHelp")}</p>
          </div>
          <FollowUp question={refine[0]} value={undefined} onChange={(v) => !busy && onRefine(refine[0].id, v, refine[0])} />
        </section>
      )}

      <details className="card p-4">
        <summary className="cursor-pointer text-sm font-semibold text-white">{t("advisor.otherOptions")}</summary>
        <div className="mt-3 flex flex-col gap-3">
          {others.map((r) => (
            <div key={r.strategy} className="flex flex-col gap-2">
              <p className="text-sm text-zinc-400">{t(META[r.strategy].plain)}</p>
              <RecommendationCard rec={r} top={false} />
            </div>
          ))}
        </div>
      </details>

      {result.hybrid_routing.recommended && (
        <div className="card p-4 flex flex-col gap-2 !border-accent/40">
          <h3 className="text-sm font-semibold text-white">{t("advisor.hybrid")}</h3>
          <p className="text-sm text-zinc-300">{result.hybrid_routing.rationale}</p>
          <ul className="text-sm text-zinc-300 space-y-0.5">
            {result.hybrid_routing.routes.map((route) => (
              <li key={route.question_type}>
                {label(QUESTION_TYPE_LABEL, route.question_type)} ({route.share_pct}%) →{" "}
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

      <ProfileSummary profile={result.profile} />
    </section>
  );
}

/** The extracted profile, folded away: useful to check, not the headline. */
function ProfileSummary({ profile: p }: { profile: ProjectProfile }) {
  const { t, locale, formatNumber } = useI18n();
  const mix = p.question_mix;
  const label = <T extends string>(map: Record<T, MessageKey>, v: T) => (map[v] ? t(map[v]) : v);
  return (
    <details className="card p-4">
      <summary className="cursor-pointer text-sm font-semibold text-white">{t("advisor.understood")}</summary>
      <div className="mt-3 flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="chip" title={t("advisor.profileSourceTitle")}>
            {p.source === "llm" ? t("advisor.sourceLlm") : t("advisor.sourceHeuristic")}
          </span>
          <span className="chip">
            {t("advisor.chip.corpus", { size: label(SIZE_LABEL, p.corpus_size) })}
            {p.corpus_size_docs != null ? t("advisor.chip.docs", { count: formatNumber(p.corpus_size_docs) }) : ""}
          </span>
          <span className="chip">{t("advisor.chip.languages", { list: p.languages.join(", ") || "?" })}</span>
          <span className="chip">
            {t("advisor.chip.latency", {
              value: p.latency_budget_ms != null ? formatLatency(p.latency_budget_ms, locale) : t("advisor.notStated"),
            })}
          </span>
          <span className="chip">{t("advisor.chip.cost", { value: label(LEVEL_LABEL, p.cost_sensitivity) })}</span>
          <span className="chip">{t("advisor.chip.freshness", { value: label(FRESHNESS_LABEL, p.data_freshness) })}</span>
          <span className="chip">{t("advisor.chip.entities", { value: label(LEVEL_LABEL, p.entity_richness) })}</span>
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
        </div>
      </div>
    </details>
  );
}

function RecommendationCard({ rec, top }: { rec: StrategyRecommendation; top: boolean }) {
  const { t, formatNumber } = useI18n();
  const m = META[rec.strategy];
  const c = rec.suggested_config;
  const title = t(m.title);
  return (
    <article className={`rounded-lg border border-bg-border bg-bg-elevated/40 p-4 flex flex-col gap-3 ${top ? m.border : ""}`}>
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
        <summary className="cursor-pointer text-xs text-zinc-400 hover:text-white">{t("advisor.suggestedConfig")}</summary>
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

/**
 * Step 4: measure the recommendation on the client's own documents. Files go
 * into a private workspace (an access group) so the strategies retrieve only
 * from them, and the client can erase them afterwards.
 */
function ValidateSection({ advice }: { advice: AdviseResponse }) {
  const { t, tp, rich, lang } = useI18n();
  const [workspace, setWorkspace] = useState(newWorkspace);
  const [docs, setDocs] = useState<{ id: string; name: string }[]>([]);
  const [fileErrors, setFileErrors] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [deleted, setDeleted] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [text, setText] = useState(() => advice.profile.example_questions.slice(0, 5).join("\n"));
  const [strategies, setStrategies] = useState<StrategyName[]>(advice.next_step.suggested_strategies);
  const lineCount = text.split(/\r?\n/).filter((l) => l.trim()).length;
  const questions = parseQuestions(text);

  const mut = useMutation<ValidateResponse, Error, void>({
    mutationFn: () => validateStrategies(questions, strategies, advice.profile, lang, workspace),
  });
  const erase = useMutation<unknown, Error, void>({
    mutationFn: () => deleteWorkspace(workspace),
    onSuccess: () => {
      setDocs([]);
      setDeleted(true);
      mut.reset();
      // A fresh workspace, so anything uploaded next starts clean.
      setWorkspace(newWorkspace());
    },
  });

  async function addFiles(files: FileList) {
    setUploading(true);
    setDeleted(false);
    const errors: string[] = [];
    for (const f of Array.from(files)) {
      try {
        const r = await upload<{ doc_id: string; filename: string }>("/ingest", f, undefined, { groups: workspace });
        setDocs((d) => [...d, { id: r.doc_id, name: r.filename }]);
      } catch (e) {
        errors.push(t("advisor.test.failedFile", { file: f.name, error: errorMessage(e) }));
      }
    }
    setFileErrors(errors);
    setUploading(false);
    if (fileRef.current) fileRef.current.value = "";
  }

  const toggle = (s: StrategyName) =>
    setStrategies((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]));
  const canRun = docs.length > 0 && questions.length > 0 && strategies.length > 0 && !mut.isPending;
  const measured = mut.data?.measured_winner;

  return (
    <details className="card p-4">
      <summary className="cursor-pointer">
        <span className="text-lg font-semibold text-white">{t("advisor.testTitle")}</span>
      </summary>
      <div className="mt-3 flex flex-col gap-4">
        <p className="text-sm text-zinc-400">{t("advisor.test.private")}</p>

        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={fileRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => e.target.files && addFiles(e.target.files)}
          />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={uploading || erase.isPending}
            className="btn-ghost text-sm"
          >
            {uploading ? t("advisor.test.uploading") : t("advisor.test.addDocs")}
          </button>
          {docs.length > 0 && <span className="text-sm text-zinc-300">{tp("advisor.test.ready", docs.length)}</span>}
          {deleted && <span className="text-sm text-emerald-300">{t("advisor.test.deleted")}</span>}
        </div>
        {docs.length > 0 && (
          <ul className="flex flex-wrap gap-1.5 text-xs">
            {docs.map((d) => (
              <li key={d.id} className="chip">{d.name}</li>
            ))}
          </ul>
        )}
        {fileErrors.map((e) => (
          <p key={e} className="text-xs text-rose-300">{e}</p>
        ))}

        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          {t("advisor.test.questions")}
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={5}
            className="input resize-y text-sm"
          />
        </label>
        <p className="text-xs text-zinc-500 -mt-2">
          {rich("advisor.validateHelp", {
            syntax: <code className="font-mono text-zinc-300">{t("advisor.validateSyntax")}</code>,
            max: MAX_VALIDATE_QUESTIONS,
          })}
        </p>
        {lineCount > MAX_VALIDATE_QUESTIONS && (
          <p className="text-xs text-amber-300">{t("advisor.tooMany", { max: MAX_VALIDATE_QUESTIONS, count: lineCount })}</p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          {(["classic", "graph", "agentic"] as StrategyName[]).map((s) => (
            <label key={s} className="chip cursor-pointer min-h-[36px]">
              <input type="checkbox" checked={strategies.includes(s)} onChange={() => toggle(s)} className="accent-amber-400" />
              <span className={META[s].text}>{t(META[s].title)}</span>
            </label>
          ))}
          <button type="button" onClick={() => mut.mutate()} disabled={!canRun} className="btn-primary text-sm ml-auto">
            {mut.isPending ? t("common.running") : tp("advisor.runQuestions", questions.length)}
          </button>
        </div>
        {docs.length === 0 && <p className="text-xs text-zinc-500">{t("advisor.test.needDocs")}</p>}
        {strategies.includes("graph") && <p className="text-xs text-zinc-500">{t("advisor.test.graphNote")}</p>}

        {mut.isError && (
          <ErrorAlert error={errorMessage(mut.error)} onRetry={() => mut.mutate()} onDismiss={() => mut.reset()} />
        )}
        {mut.isPending && (
          <LoadingSpinner message={t("advisor.validating", { questions: questions.length, strategies: strategies.length })} />
        )}
        {measured && (
          <p className={`text-sm font-medium ${measured === advice.top_strategy ? "text-emerald-300" : "text-amber-300"}`}>
            {measured === advice.top_strategy
              ? t("advisor.test.confirms", { strategy: t(META[measured].title) })
              : t("advisor.test.differs", {
                  measured: t(META[measured].title),
                  recommended: t(META[advice.top_strategy].title),
                })}
          </p>
        )}
        {mut.data && <Scorecard data={mut.data} />}

        {docs.length > 0 && (
          <div>
            <button
              type="button"
              onClick={() => erase.mutate()}
              disabled={erase.isPending || uploading || mut.isPending}
              className="btn-ghost text-xs"
            >
              {erase.isPending ? t("advisor.test.deleting") : t("advisor.test.delete")}
            </button>
            {erase.isError && <ErrorAlert error={errorMessage(erase.error)} onDismiss={() => erase.reset()} />}
          </div>
        )}
      </div>
    </details>
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
            <span className={`font-semibold ${META[data.measured_winner].text}`}>{t(META[data.measured_winner].title)}</span>
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
                <td className="py-2 pr-3 font-mono">{formatTokens(Math.round(c.avg_tokens_per_question), locale)}</td>
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
                {r.refusal && <span className="chip !text-rose-300">{r.error ? t("common.error") : t("advisor.refused")}</span>}
                <span className="ml-auto font-mono text-zinc-500">
                  {formatLatency(r.latency_ms, locale)} · {formatTokens(r.input_tokens + r.output_tokens, locale)}
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
