import { useState, type ReactNode } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  AdviseOverrides,
  AdviseResponse,
  Freshness,
  Level,
  MAX_VALIDATE_QUESTIONS,
  StrategyName,
  StrategyRecommendation,
  ValidateResponse,
  advise,
  parseQuestions,
  validateStrategies,
} from "../api/advisor";
import ErrorAlert from "../components/ErrorAlert";
import LoadingSpinner from "../components/LoadingSpinner";
import { formatLatency, formatTokens } from "../utils/formatting";

const META: Record<StrategyName, { title: string; text: string; bar: string; border: string }> = {
  classic: {
    title: "Classic RAG",
    text: "text-sky-300",
    bar: "bg-sky-400",
    border: "border-sky-500/40",
  },
  graph: {
    title: "Graph RAG",
    text: "text-violet-300",
    bar: "bg-violet-400",
    border: "border-violet-500/40",
  },
  agentic: {
    title: "Agentic RAG",
    text: "text-emerald-300",
    bar: "bg-emerald-400",
    border: "border-emerald-500/40",
  },
};

const QUESTION_TYPE_LABEL: Record<string, string> = {
  single_fact: "Single-fact",
  relational_multi_hop: "Relational / multi-hop",
  exploratory_multi_step: "Exploratory / multi-step",
};

const EXAMPLE =
  "We are an insurance company with about 20,000 PDF contracts and policy documents in French and English. " +
  "Agents ask things like 'Which clauses cover water damage for policy X?' and 'Which partners are linked to claim Y?'. " +
  "Answers should come back in under 5 seconds. Data must stay in the EU (GDPR).";

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

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export default function Advisor() {
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
        <h1 className="display text-4xl font-semibold text-white">Advisor</h1>
        <p className="text-sm text-zinc-400">
          Describe your project and get a ranked recommendation between{" "}
          <span className="text-sky-300">Classic</span>, <span className="text-violet-300">Graph</span>{" "}
          and <span className="text-emerald-300">Agentic</span> RAG, then validate it on your own questions.
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
          Your project
        </label>
        <textarea
          id="advisor-description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={6}
          placeholder="Describe your documents, who asks questions, what kinds of questions, how fast answers must be, budget, and any hosting or compliance constraints…"
          className="input resize-y text-sm"
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-zinc-500">
            Any language works: vous pouvez décrire votre projet en français.{" "}
            <button
              type="button"
              onClick={() => setDescription(EXAMPLE)}
              className="text-accent hover:text-accent-hover underline-offset-2 hover:underline"
            >
              Use an example
            </button>
          </p>
          <button
            type="button"
            onClick={() => setShowOverrides((v) => !v)}
            className="text-xs text-zinc-400 hover:text-white"
            aria-expanded={showOverrides}
          >
            {showOverrides ? "▾" : "▸"} Known facts (optional overrides)
          </button>
        </div>

        {showOverrides && (
          <div className="grid gap-3 sm:grid-cols-2 border-t border-bg-border pt-3">
            <Field label="Corpus size (documents)">
              <input
                type="number"
                min={0}
                value={form.corpus_size_docs}
                onChange={(e) => set("corpus_size_docs", e.target.value)}
                className="input text-sm !py-2"
                placeholder="e.g. 20000"
              />
            </Field>
            <Field label="Languages (comma-separated ISO codes)">
              <input
                value={form.languages}
                onChange={(e) => set("languages", e.target.value)}
                className="input text-sm !py-2"
                placeholder="e.g. fr, en"
              />
            </Field>
            <Field label="Latency budget (ms)">
              <input
                type="number"
                min={100}
                value={form.latency_budget_ms}
                onChange={(e) => set("latency_budget_ms", e.target.value)}
                className="input text-sm !py-2"
                placeholder="e.g. 5000"
              />
            </Field>
            <Field label="Cost sensitivity">
              <Select
                value={form.cost_sensitivity}
                onChange={(v) => set("cost_sensitivity", v as OverrideForm["cost_sensitivity"])}
                options={["low", "medium", "high"]}
              />
            </Field>
            <Field label="Data freshness">
              <Select
                value={form.data_freshness}
                onChange={(v) => set("data_freshness", v as OverrideForm["data_freshness"])}
                options={["static", "monthly", "weekly", "daily", "realtime"]}
              />
            </Field>
            <Field label="Entity richness">
              <Select
                value={form.entity_richness}
                onChange={(v) => set("entity_richness", v as OverrideForm["entity_richness"])}
                options={["low", "medium", "high"]}
              />
            </Field>
            <Field label="Residency / compliance (comma-separated)">
              <input
                value={form.compliance}
                onChange={(e) => set("compliance", e.target.value)}
                className="input text-sm !py-2"
                placeholder="e.g. EU only, GDPR, on-prem"
              />
            </Field>
            <Field label="Example questions (one per line)">
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
            {adviseMut.isPending ? "Analysing…" : "Recommend a strategy"}
          </button>
        </div>
        {adviseMut.isError && (
          <ErrorAlert
            error={errorText(adviseMut.error)}
            onRetry={() => adviseMut.mutate()}
            onDismiss={() => adviseMut.reset()}
          />
        )}
      </form>

      {adviseMut.isPending && <LoadingSpinner message="Reading your description and scoring strategies…" />}
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

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="input text-sm !py-2">
      <option value="">From description</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function AdviceResult({ result }: { result: AdviseResponse }) {
  const p = result.profile;
  const mix = p.question_mix;
  return (
    <section className="flex flex-col gap-4">
      <div className="card p-4 flex flex-col gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold text-white">Project profile</h2>
          <span className="chip" title="How the profile was extracted">
            {p.source === "llm" ? "extracted by Claude" : "keyword heuristic (no model call)"}
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5 text-xs">
          <span className="chip">
            corpus: {p.corpus_size}
            {p.corpus_size_docs != null ? ` (${p.corpus_size_docs.toLocaleString()} docs)` : ""}
          </span>
          <span className="chip">languages: {p.languages.join(", ") || "?"}</span>
          <span className="chip">
            latency: {p.latency_budget_ms != null ? formatLatency(p.latency_budget_ms) : "not stated"}
          </span>
          <span className="chip">cost sensitivity: {p.cost_sensitivity}</span>
          <span className="chip">freshness: {p.data_freshness}</span>
          <span className="chip">entities: {p.entity_richness}</span>
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
          Question mix: {mix.single_fact}% single-fact · {mix.relational_multi_hop}% relational ·{" "}
          {mix.exploratory_multi_step}% exploratory
          {p.overridden_fields.length > 0 && (
            <span className="text-zinc-500"> · overridden: {p.overridden_fields.join(", ")}</span>
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
          <h3 className="text-sm font-semibold text-white">Hybrid routing suggested</h3>
          <p className="text-sm text-zinc-300">{result.hybrid_routing.rationale}</p>
          <ul className="text-sm text-zinc-300 space-y-0.5">
            {result.hybrid_routing.routes.map((route) => (
              <li key={route.question_type}>
                {QUESTION_TYPE_LABEL[route.question_type]} ({route.share_pct}%) →{" "}
                <span className={META[route.strategy].text}>{META[route.strategy].title}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.compliance_notes.length > 0 && (
        <div className="card p-4 flex flex-col gap-2 !border-amber-500/30">
          <h3 className="text-sm font-semibold text-white">Hosting, compliance and language notes</h3>
          <ul className="list-disc pl-5 text-sm text-zinc-300 space-y-1">
            {result.compliance_notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="card p-4 text-sm text-zinc-300 !bg-accent-soft/40 !border-accent/30">
        <span className="font-semibold text-accent">Next step: </span>
        {result.next_step.summary}
      </div>
    </section>
  );
}

function RecommendationCard({ rec, top }: { rec: StrategyRecommendation; top: boolean }) {
  const m = META[rec.strategy];
  const c = rec.suggested_config;
  return (
    <article className={`card p-4 flex flex-col gap-3 ${top ? m.border : ""}`}>
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono text-zinc-500">#{rec.rank}</span>
        <h3 className={`font-semibold ${m.text}`}>{m.title}</h3>
        {top && <span className="chip !text-accent !border-accent/40">recommended</span>}
        <span className="ml-auto font-mono text-sm text-white">{rec.score}/100</span>
      </div>
      <div
        className="h-2 w-full rounded-full bg-bg-elevated overflow-hidden"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={rec.score}
        aria-label={`${m.title} score`}
      >
        <div className={`h-full ${m.bar} transition-all`} style={{ width: `${rec.score}%` }} />
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <h4 className="text-xs uppercase tracking-wider text-zinc-500 mb-1">Why</h4>
          <ul className="list-disc pl-5 text-sm text-zinc-300 space-y-1">
            {rec.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="text-xs uppercase tracking-wider text-zinc-500 mb-1">Tradeoffs</h4>
          <ul className="list-disc pl-5 text-sm text-zinc-400 space-y-1">
            {rec.tradeoffs.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
      </div>

      <details className="text-sm">
        <summary className="cursor-pointer text-xs text-zinc-400 hover:text-white">Suggested config</summary>
        <div className="mt-2 grid gap-x-6 gap-y-1 sm:grid-cols-2 font-mono text-xs text-zinc-300">
          <div>RETRIEVAL_TOP_K = {c.retrieval_top_k}</div>
          <div>RERANK_TOP_K = {c.rerank_top_k}</div>
          <div>rerank = {c.rerank ? "on" : "off"}</div>
          <div>CHUNK_SIZE_TOKENS = {c.chunk_size_tokens}</div>
          <div>CHUNK_OVERLAP_TOKENS = {c.chunk_overlap_tokens}</div>
          <div className="break-all">EMBEDDING_MODEL = {c.embedding_model}</div>
          {Object.entries(c.models).map(([stage, model]) => (
            <div key={stage} className="break-all">
              {stage}: {model}
            </div>
          ))}
          <div>
            relative cost ×{c.expected_relative_cost} · latency ×{c.expected_relative_latency}
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
        <h2 className="text-lg font-semibold text-white">Validate on my questions</h2>
        <p className="text-xs text-zinc-500 mt-1">
          Runs against the documents currently ingested. One question per line; add{" "}
          <code className="font-mono text-zinc-300">question || ideal answer</code> to also score
          answer quality. Up to {MAX_VALIDATE_QUESTIONS} questions.
        </p>
      </div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        className="input resize-y text-sm font-mono"
        placeholder={"Which clauses cover water damage? || Clause 4.2 and annex B\nWho is the broker for policy 123?"}
      />
      {lineCount > MAX_VALIDATE_QUESTIONS && (
        <p className="text-xs text-amber-300">
          Only the first {MAX_VALIDATE_QUESTIONS} of {lineCount} questions will be sent.
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
            <span className={META[s].text}>{META[s].title}</span>
          </label>
        ))}
        <button
          onClick={() => mut.mutate()}
          disabled={questions.length === 0 || strategies.length === 0 || mut.isPending}
          className="btn-primary text-sm ml-auto"
        >
          {mut.isPending
            ? "Running…"
            : `Run ${questions.length} question${questions.length === 1 ? "" : "s"}`}
        </button>
      </div>
      {mut.isError && (
        <ErrorAlert
          error={errorText(mut.error)}
          onRetry={() => mut.mutate()}
          onDismiss={() => mut.reset()}
        />
      )}
      {mut.isPending && (
        <LoadingSpinner
          message={`Running ${questions.length} × ${strategies.length} strategy runs sequentially; this can take a few minutes.`}
        />
      )}
      {mut.data && <Scorecard data={mut.data} />}
    </section>
  );
}

function fmtScore(v: number | null): string {
  return v == null ? "–" : v.toFixed(2);
}

function Scorecard({ data }: { data: ValidateResponse }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="text-sm">
        {data.measured_winner ? (
          <>
            <span className="text-zinc-400">Measured winner: </span>
            <span className={`font-semibold ${META[data.measured_winner].text}`}>
              {META[data.measured_winner].title}
            </span>
          </>
        ) : (
          <span className="text-amber-300">No measured winner.</span>
        )}
        <p className="text-xs text-zinc-500 mt-1">{data.winner_reason}</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wider text-zinc-500 border-b border-bg-border">
              <th className="py-2 pr-3">Strategy</th>
              <th className="py-2 pr-3">Refusals</th>
              <th className="py-2 pr-3">Errors</th>
              <th className="py-2 pr-3">Avg latency</th>
              <th className="py-2 pr-3">Tokens / q</th>
              <th className="py-2 pr-3">Judge</th>
              <th className="py-2 pr-3">F1 vs ideal</th>
            </tr>
          </thead>
          <tbody>
            {data.scorecard.map((c) => (
              <tr
                key={c.strategy}
                className={`border-b border-bg-border/60 ${c.strategy === data.measured_winner ? "bg-bg-elevated" : ""}`}
              >
                <td className={`py-2 pr-3 font-medium ${META[c.strategy].text}`}>{META[c.strategy].title}</td>
                <td className="py-2 pr-3 font-mono">
                  {c.refusals}/{c.questions}
                </td>
                <td className="py-2 pr-3 font-mono">{c.errors}</td>
                <td className="py-2 pr-3 font-mono">{formatLatency(c.avg_latency_ms)}</td>
                <td className="py-2 pr-3 font-mono">{formatTokens(Math.round(c.avg_tokens_per_question))}</td>
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
          Per-question answers ({data.rows.length})
        </summary>
        <ul className="mt-2 flex flex-col gap-2">
          {data.rows.map((r, i) => (
            <li key={i} className="border border-bg-border rounded-lg p-2 text-xs">
              <div className="flex flex-wrap gap-2 items-center">
                <span className={META[r.strategy].text}>{META[r.strategy].title}</span>
                <span className="text-zinc-300">{r.question}</span>
                {r.refusal && <span className="chip !text-rose-300">{r.error ? "error" : "refused"}</span>}
                <span className="ml-auto font-mono text-zinc-500">
                  {formatLatency(r.latency_ms)} · {formatTokens(r.input_tokens + r.output_tokens)}
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
