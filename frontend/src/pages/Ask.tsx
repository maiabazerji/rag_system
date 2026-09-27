import { useRef, useState } from "react";
import { errorMessage, post } from "../api/client";
import { SendIcon } from "../components/Icons";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorAlert from "../components/ErrorAlert";
import Tooltip from "../components/Tooltip";
import MetadataRow from "../components/MetadataRow";
import { cleanAnswer, formatConfidence } from "../utils/formatting";

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
const STRATEGIES: { id: Strategy; label: string; hint: string }[] = [
  { id: "classic", label: "Classic", hint: "vector search → rerank → answer" },
  { id: "graph", label: "Graph", hint: "entity walk on knowledge graph" },
  { id: "agentic", label: "Agentic", hint: "model loops over search/fetch tools" },
];

export default function Ask() {
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
    setTurns((t) => t.map((x) => (x.id === id ? { id, question: x.question, ...patch } : x)));
  }

  async function submit() {
    const question = q.trim();
    if (!question || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    const id = nextId.current++;
    setTurns((t) => [...t, { id, question, loading: true }]);
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
    setTurns((t) => t.filter((x) => x.id !== id));
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between">
        <div className="space-y-2">
          <h1 className="display text-4xl font-semibold text-white">Ask</h1>
          <p className="text-sm text-zinc-400">
            Question indexed documents. Sources let you verify accuracy.
          </p>
        </div>
        {turns.length > 0 && (
          <button
            onClick={() => setTurns([])}
            className="btn-ghost text-xs"
          >
            Clear
          </button>
        )}
      </header>

      {turns.length === 0 && <EmptyState onPick={(s) => setQ(s)} />}

      <div className="flex flex-col gap-5">
        {turns.map((t) => (
          <TurnView key={t.id} turn={t} onDismiss={() => dismiss(t.id)} />
        ))}
      </div>

      <div className="sticky bottom-4 mt-2 space-y-2">
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-zinc-500 uppercase tracking-wider mr-1">Strategy</span>
          {STRATEGIES.map((s) => {
            const active = strategy === s.id;
            return (
              <button
                key={s.id}
                onClick={() => setStrategy(s.id)}
                title={s.hint}
                aria-label={`${s.label} strategy: ${s.hint}`}
                aria-pressed={active}
                className={`chip transition-colors min-h-[44px] sm:min-h-auto ${
                  active
                    ? "!text-white !bg-accent/20 !border-accent/50"
                    : "hover:!text-white"
                }`}
              >
                {s.label}
              </button>
            );
          })}
          <span className="text-zinc-500 ml-1 hidden sm:inline">
            · {STRATEGIES.find((s) => s.id === strategy)?.hint}
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
            placeholder="Type a question…  ⏎ to send  ·  shift+⏎ for newline"
            className="input !border-0 !bg-transparent resize-none focus:!ring-0 min-h-[44px] max-h-40"
          />
          <button onClick={submit} disabled={!q.trim() || busy} className="btn-primary h-11">
            <SendIcon className="w-4 h-4" />
            {busy ? "Asking…" : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (s: string) => void }) {
  const groups: { label: string; questions: string[] }[] = [
    {
      label: "Retrieval Strategy & Tradeoffs",
      questions: [
        "How does entity extraction in Graph RAG reduce hallucination compared to Classic RAG?",
        "When should you use dense retrieval over hybrid, and what are the latency-quality tradeoffs?",
        "Why might Agentic RAG overshoot on simple queries while excelling on complex reasoning?",
        "How does reranking affect both precision and recall in multi-hop retrieval scenarios?",
      ],
    },
    {
      label: "System Design & Architecture",
      questions: [
        "What's the relationship between chunk size, embedding model, and retrieval quality?",
        "How do you design a RAG pipeline to handle both factual and synthesis queries efficiently?",
        "How does knowledge graph construction from documents affect retrieval coverage vs. hallucination?",
        "What are the failure modes of dense-only retrieval with structurally ambiguous documents?",
      ],
    },
    {
      label: "Evaluation & Production",
      questions: [
        "What's the distinction between context precision and context recall, and how do they interact?",
        "How would you optimize a RAG pipeline for both speed and accuracy with heterogeneous documents?",
        "Why does Agentic RAG use more tokens than Classic on the same corpus, and when is that justified?",
        "How do you detect and prevent retrieval drift as your corpus grows over time?",
      ],
    },
  ];
  return (
    <div className="card border-dashed py-8 px-8">
      <div className="text-[10px] uppercase tracking-[0.18em] text-zinc-500 mb-5">
        Try a question
      </div>
      <div className="grid sm:grid-cols-3 gap-x-8 gap-y-6">
        {groups.map((g) => (
          <div key={g.label}>
            <div className="display text-sm text-accent mb-2">{g.label}</div>
            <div className="flex flex-col gap-1.5">
              {g.questions.map((s) => (
                <button
                  key={s}
                  onClick={() => onPick(s)}
                  className="text-left text-zinc-300 hover:text-white transition-colors py-0.5 group text-sm leading-snug"
                >
                  <span className="text-zinc-600 group-hover:text-accent mr-1.5">→</span>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TurnView({ turn, onDismiss }: { turn: Turn; onDismiss: () => void }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="self-end max-w-[85%] bg-accent/15 border border-accent/30 rounded-2xl rounded-tr-sm px-4 py-2.5 text-zinc-100">
        {turn.question}
      </div>

      <div className="card animate-fade-in">
        {turn.loading && <LoadingSpinner message="Retrieving and generating answer..." />}
        {turn.error && (
          <ErrorAlert error={turn.error} onDismiss={onDismiss} />
        )}
        {turn.answer && <AnswerView a={turn.answer} />}
      </div>
    </div>
  );
}


function AnswerView({ a }: { a: Answer }) {
  const pct = Math.round(a.confidence * 100);
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`chip ${a.refusal ? "!text-amber-300 !border-amber-500/40 !bg-amber-500/10" : "!text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10"}`}>
          {a.refusal ? "⚠ Refused" : "✓ Answered"}
        </span>
        {a.provider && <span className="chip">{a.provider}</span>}
        <span className="chip" title="Model confidence in this answer">{formatConfidence(a.confidence)}</span>
        <Tooltip label="How much the model trusts its answer (0-1 scale). Higher = more confident it's grounded in sources.">
          <ConfidenceBar pct={pct} refused={a.refusal} />
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
          trace{" "}
          <a
            href={`/api/traces/${encodeURIComponent(a.trace_id)}`}
            target="_blank"
            rel="noreferrer"
            className="text-zinc-400 hover:text-accent underline decoration-dotted"
            title="GET /traces/{id}: the retrieval, rerank and generation steps for this answer"
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
  // The backend uses a single chunk_id "none" source to mean "nothing cited".
  const real = sources.filter((s) => s.chunk_id && s.chunk_id !== "none");
  if (real.length === 0) return null;
  return (
    <details className="text-xs border-t border-bg-border pt-3">
      <summary className="cursor-pointer text-zinc-400 hover:text-zinc-200 select-none">
        Sources ({real.length})
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
        Rated {rating}/5, thanks. Your rating was saved.
      </div>
    );
  }

  return (
    <div className="border-t border-bg-border pt-3 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs uppercase tracking-wider text-zinc-500">Your rating</span>
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
                aria-label={`${n} star${n > 1 ? "s" : ""}`}
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
        placeholder="Optional: why this rating?"
        className="input resize-none text-sm"
      />
      <div className="flex items-center gap-2">
        <button onClick={submit} disabled={!rating || state === "saving"} className="btn-primary h-8 text-sm">
          {state === "saving" ? "Saving…" : "Submit rating"}
        </button>
        {state === "error" && <span className="text-rose-400 text-xs">{err}</span>}
      </div>
    </div>
  );
}

function ConfidenceBar({ pct, refused }: { pct: number; refused: boolean }) {
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
          aria-label="Confidence score"
        />
      </div>
      <span className="text-xs text-zinc-400 font-mono tabular-nums w-10 text-right" aria-label={`${pct} percent confidence`}>{pct}%</span>
    </div>
  );
}
