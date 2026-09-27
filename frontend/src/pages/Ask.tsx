import { useRef, useState } from "react";
import { errorMessage, post } from "../api/client";
import { SendIcon } from "../components/Icons";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorAlert from "../components/ErrorAlert";
import Tooltip from "../components/Tooltip";
import MetadataRow from "../components/MetadataRow";
import { cleanAnswer, formatConfidence } from "../utils/formatting";
import { useI18n, type MessageKey } from "../i18n";

type Source = { chunk_id: string; quote: string };
type Answer = {
  question: string;
  answer: string;
  sources: Source[];
  confidence: number;
  refusal: boolean;
  provider?: string | null;
  model?: string | null;
  latency_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
  /** Present when the backend recorded a trace; fetch it from GET /traces/{id}. */
  trace_id?: string | null;
};

/** `id` is stable, so a slow answer lands on its own turn even after a Clear. */
type Turn = { id: number; question: string; answer?: Answer; error?: string; loading?: boolean };

type Strategy = "classic" | "graph" | "agentic";
const STRATEGIES: { id: Strategy; label: MessageKey; hint: MessageKey }[] = [
  { id: "classic", label: "strategy.classic.short", hint: "ask.hint.classic" },
  { id: "graph", label: "strategy.graph.short", hint: "ask.hint.graph" },
  { id: "agentic", label: "strategy.agentic.short", hint: "ask.hint.agentic" },
];

export default function Ask() {
  const { t } = useI18n();
  const [q, setQ] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [strategy, setStrategy] = useState<Strategy>("classic");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const nextId = useRef(0);
  // A ref, not state: two Enter presses inside one render would both see
  // stale state and send the question twice.
  const inFlight = useRef(false);
  const [busy, setBusy] = useState(false);

  function updateTurn(id: number, patch: Omit<Turn, "id" | "question">) {
    setTurns((prev) => prev.map((x) => (x.id === id ? { id, question: x.question, ...patch } : x)));
  }

  async function submit() {
    const question = q.trim();
    if (!question || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    const id = nextId.current++;
    setTurns((prev) => [...prev, { id, question, loading: true }]);
    setQ("");
    try {
      const a = await post<Answer>("/ask", { question, strategy });
      updateTurn(id, { answer: a });
    } catch (e) {
      updateTurn(id, { error: errorMessage(e) });
    } finally {
      inFlight.current = false;
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  function dismiss(id: number) {
    setTurns((prev) => prev.filter((x) => x.id !== id));
  }

  const active = STRATEGIES.find((s) => s.id === strategy);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between">
        <div className="space-y-2">
          <h1 className="display text-4xl font-semibold text-white">{t("ask.title")}</h1>
          <p className="text-sm text-zinc-400">{t("ask.subtitle")}</p>
        </div>
        {turns.length > 0 && (
          <button
            onClick={() => setTurns([])}
            className="btn-ghost text-xs"
          >
            {t("common.clear")}
          </button>
        )}
      </header>

      {turns.length === 0 && <EmptyState onPick={(s) => setQ(s)} />}

      <div className="flex flex-col gap-5">
        {turns.map((turn) => (
          <TurnView key={turn.id} turn={turn} onDismiss={() => dismiss(turn.id)} />
        ))}
      </div>

      <div className="sticky bottom-4 mt-2 space-y-2">
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-zinc-500 uppercase tracking-wider mr-1">{t("ask.strategy")}</span>
          {STRATEGIES.map((s) => {
            const isActive = strategy === s.id;
            return (
              <button
                key={s.id}
                onClick={() => setStrategy(s.id)}
                title={t(s.hint)}
                aria-label={t("ask.strategyAria", { label: t(s.label), hint: t(s.hint) })}
                aria-pressed={isActive}
                className={`chip transition-colors min-h-[44px] sm:min-h-auto ${
                  isActive
                    ? "!text-white !bg-accent/20 !border-accent/50"
                    : "hover:!text-white"
                }`}
              >
                {t(s.label)}
              </button>
            );
          })}
          <span className="text-zinc-500 ml-1 hidden sm:inline">
            · {active ? t(active.hint) : ""}
          </span>
        </div>
        <div className="card !p-2 flex items-end gap-2 ring-1 ring-bg-border focus-within:ring-accent/40 transition-shadow">
          <textarea
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder={t("ask.placeholder")}
            className="input !border-0 !bg-transparent resize-none focus:!ring-0 min-h-[44px] max-h-40"
          />
          <button onClick={submit} disabled={!q.trim() || busy} className="btn-primary h-11">
            <SendIcon className="w-4 h-4" />
            {busy ? t("ask.asking") : t("ask.submit")}
          </button>
        </div>
      </div>
    </div>
  );
}

const SAMPLE_GROUPS: { label: MessageKey; questions: MessageKey[] }[] = [
  {
    label: "ask.samples.g1",
    questions: ["ask.samples.g1.q1", "ask.samples.g1.q2", "ask.samples.g1.q3", "ask.samples.g1.q4"],
  },
  {
    label: "ask.samples.g2",
    questions: ["ask.samples.g2.q1", "ask.samples.g2.q2", "ask.samples.g2.q3", "ask.samples.g2.q4"],
  },
  {
    label: "ask.samples.g3",
    questions: ["ask.samples.g3.q1", "ask.samples.g3.q2", "ask.samples.g3.q3", "ask.samples.g3.q4"],
  },
];

function EmptyState({ onPick }: { onPick: (s: string) => void }) {
  const { t } = useI18n();
  return (
    <div className="card border-dashed py-8 px-8">
      <div className="text-[10px] uppercase tracking-[0.18em] text-zinc-500 mb-5">
        {t("ask.tryQuestion")}
      </div>
      <div className="grid sm:grid-cols-3 gap-x-8 gap-y-6">
        {SAMPLE_GROUPS.map((g) => (
          <div key={g.label}>
            <div className="display text-sm text-accent mb-2">{t(g.label)}</div>
            <div className="flex flex-col gap-1.5">
              {g.questions.map((key) => {
                const s = t(key);
                return (
                  <button
                    key={key}
                    onClick={() => onPick(s)}
                    className="text-left text-zinc-300 hover:text-white transition-colors py-0.5 group text-sm leading-snug"
                  >
                    <span className="text-zinc-600 group-hover:text-accent mr-1.5">→</span>
                    {s}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TurnView({ turn, onDismiss }: { turn: Turn; onDismiss: () => void }) {
  const { t } = useI18n();
  return (
    <div className="flex flex-col gap-3">
      <div className="self-end max-w-[85%] bg-accent/15 border border-accent/30 rounded-2xl rounded-tr-sm px-4 py-2.5 text-zinc-100">
        {turn.question}
      </div>

      <div className="card animate-fade-in">
        {turn.loading && <LoadingSpinner message={t("ask.loading")} />}
        {turn.error && (
          <ErrorAlert error={turn.error} onDismiss={onDismiss} />
        )}
        {turn.answer && <AnswerView a={turn.answer} />}
      </div>
    </div>
  );
}


function AnswerView({ a }: { a: Answer }) {
  const { t, locale } = useI18n();
  const pct = Math.round(a.confidence * 100);
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`chip ${a.refusal ? "!text-amber-300 !border-amber-500/40 !bg-amber-500/10" : "!text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10"}`}>
          {a.refusal ? `⚠ ${t("common.refused")}` : `✓ ${t("common.answered")}`}
        </span>
        {a.provider && <span className="chip">{a.provider}</span>}
        <span className="chip" title={t("ask.confidenceTitle")}>{formatConfidence(a.confidence, locale)}</span>
        <Tooltip label={t("ask.confidenceTip")}>
          <ConfidenceBar confidence={a.confidence} pct={pct} refused={a.refusal} />
        </Tooltip>
      </div>

      <p className="text-zinc-200 leading-relaxed whitespace-pre-wrap">{cleanAnswer(a.answer)}</p>

      <SourcesList sources={a.sources} />

      <MetadataRow
        compact
        model={a.model ?? undefined}
        latency_ms={a.latency_ms}
        input_tokens={a.input_tokens}
        output_tokens={a.output_tokens}
      />

      {a.trace_id && (
        <div className="text-[11px] text-zinc-500 font-mono">
          {t("ask.trace")}{" "}
          <a
            href={`/api/traces/${encodeURIComponent(a.trace_id)}`}
            target="_blank"
            rel="noreferrer"
            className="text-zinc-400 hover:text-accent underline decoration-dotted"
            title={t("ask.traceTitle")}
          >
            {a.trace_id}
          </a>
        </div>
      )}

      {!a.refusal && <RatingWidget a={a} />}
    </div>
  );
}

function SourcesList({ sources }: { sources: Source[] }) {
  const { t } = useI18n();
  // The backend uses a single chunk_id "none" source to mean "nothing cited".
  const real = sources.filter((s) => s.chunk_id && s.chunk_id !== "none");
  if (real.length === 0) return null;
  return (
    <details className="text-xs border-t border-bg-border pt-3">
      <summary className="cursor-pointer text-zinc-400 hover:text-zinc-200 select-none">
        {t("common.sources", { count: real.length })}
      </summary>
      <ol className="mt-2 space-y-2">
        {real.map((s, i) => (
          <li key={`${s.chunk_id}-${i}`} className="flex gap-2">
            <span className="text-zinc-600 font-mono shrink-0">[{i + 1}]</span>
            <div className="min-w-0">
              <div className="font-mono text-[11px] text-accent break-all">{s.chunk_id}</div>
              {s.quote && (
                <blockquote className="text-zinc-300 whitespace-pre-wrap border-l-2 border-bg-border pl-2 mt-1">
                  {s.quote}
                </blockquote>
              )}
            </div>
          </li>
        ))}
      </ol>
    </details>
  );
}

function RatingWidget({ a }: { a: Answer }) {
  const { t, tp } = useI18n();
  const [rating, setRating] = useState<number>(0);
  const [hover, setHover] = useState<number>(0);
  const [comment, setComment] = useState("");
  const [state, setState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [err, setErr] = useState<string>("");

  async function submit() {
    if (!rating) return;
    setState("saving");
    try {
      await post("/eval/human-rate", {
        question: a.question,
        answer: a.answer,
        rating,
        comment: comment.trim() || null,
        provider: a.provider ?? null,
        model: a.model ?? null,
      });
      setState("saved");
    } catch (e) {
      setErr(errorMessage(e));
      setState("error");
    }
  }

  if (state === "saved") {
    return (
      <div className="border-t border-bg-border pt-3 text-xs text-emerald-300">
        {t("ask.rated", { rating })}
      </div>
    );
  }

  return (
    <div className="border-t border-bg-border pt-3 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs uppercase tracking-wider text-zinc-500">{t("ask.yourRating")}</span>
        <div className="flex items-center gap-1">
          {[1, 2, 3, 4, 5].map((n) => {
            const filled = (hover || rating) >= n;
            return (
              <button
                key={n}
                type="button"
                onMouseEnter={() => setHover(n)}
                onMouseLeave={() => setHover(0)}
                onClick={() => setRating(n)}
                className={`text-xl leading-none transition ${filled ? "text-amber-300" : "text-zinc-600 hover:text-zinc-400"}`}
                aria-label={tp("ask.stars", n)}
              >
                ★
              </button>
            );
          })}
        </div>
      </div>
      <textarea
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        rows={2}
        placeholder={t("ask.ratingPlaceholder")}
        className="input resize-none text-sm"
      />
      <div className="flex items-center gap-2">
        <button onClick={submit} disabled={!rating || state === "saving"} className="btn-primary h-8 text-sm">
          {state === "saving" ? t("ask.saving") : t("ask.submitRating")}
        </button>
        {state === "error" && <span className="text-rose-400 text-xs">{err}</span>}
      </div>
    </div>
  );
}

function ConfidenceBar({
  confidence,
  pct,
  refused,
}: {
  confidence: number;
  pct: number;
  refused: boolean;
}) {
  const { t, locale } = useI18n();
  return (
    <div className="flex items-center gap-2 flex-1 max-w-xs">
      <div className="flex-1 h-1.5 bg-bg-elevated rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${refused ? "bg-amber-500" : "bg-gradient-to-r from-accent to-emerald-400"}`}
          style={{ width: `${Math.max(2, pct)}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={t("ask.confidenceAria")}
        />
      </div>
      <span
        className="text-xs text-zinc-400 font-mono tabular-nums w-12 text-right"
        aria-label={t("ask.confidencePercentAria", { pct })}
      >
        {formatConfidence(confidence, locale)}
      </span>
    </div>
  );
}
