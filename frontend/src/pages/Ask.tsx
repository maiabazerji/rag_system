import { useRef, useState } from "react";
import { post } from "../api/client";
import { SendIcon } from "../components/Icons";

type Source = { chunk_id: string; quote: string };
type Answer = {
  question: string;
  answer: string;
  sources: Source[];
  confidence: number;
  refusal: boolean;
  provider?: string | null;
  model?: string | null;
};

type Turn = { question: string; answer?: Answer; error?: string; loading?: boolean };

export default function Ask() {
  const [q, setQ] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  async function submit() {
    const question = q.trim();
    if (!question) return;
    const idx = turns.length;
    setTurns((t) => [...t, { question, loading: true }]);
    setQ("");
    try {
      const a = await post<Answer>("/ask", { question });
      setTurns((t) => t.map((x, i) => (i === idx ? { question, answer: a } : x)));
    } catch (e: any) {
      setTurns((t) => t.map((x, i) => (i === idx ? { question, error: String(e.message || e) } : x)));
    } finally {
      inputRef.current?.focus();
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-4xl font-semibold text-white">Ask</h1>
        <p className="text-zinc-400 max-w-xl">
          Question your indexed documents. Every answer comes with the chunks it stood on —
          if the source isn't there, the answer isn't trustworthy.
        </p>
      </header>

      {turns.length === 0 && <EmptyState onPick={(s) => setQ(s)} />}

      <div className="flex flex-col gap-5">
        {turns.map((t, i) => (
          <TurnView key={i} turn={t} />
        ))}
      </div>

      <div className="sticky bottom-4 mt-2">
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
          <button onClick={submit} disabled={!q.trim()} className="btn-primary h-11">
            <SendIcon className="w-4 h-4" />
            Ask
          </button>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (s: string) => void }) {
  const groups: { label: string; questions: string[] }[] = [
    {
      label: "Retrieval",
      questions: [
        "How does hybrid retrieval beat dense-only?",
        "What does a cross-encoder reranker actually do?",
        "When should I rebuild embeddings vs. just re-rank?",
        "What is BM25 and why pair it with dense vectors?",
        "How do I pick a chunk size and overlap?",
      ],
    },
    {
      label: "Evaluation",
      questions: [
        "Why do we need a golden dataset?",
        "What is faithfulness and how is it scored?",
        "How does an LLM-as-judge avoid grading itself?",
        "What does a regression in eval metrics look like?",
        "Faithfulness vs. answer relevance — which matters more?",
      ],
    },
    {
      label: "System",
      questions: [
        "What's the difference between RAG and fine-tuning?",
        "How do citations reduce hallucination in practice?",
        "Why use structured outputs for LLM responses?",
        "What does Langfuse give me that logs don't?",
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

function TurnView({ turn }: { turn: Turn }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="self-end max-w-[85%] bg-accent/15 border border-accent/30 rounded-2xl rounded-tr-sm px-4 py-2.5 text-zinc-100">
        {turn.question}
      </div>

      <div className="card animate-fade-in">
        {turn.loading && <Skeleton />}
        {turn.error && <p className="text-rose-400 text-sm">{turn.error}</p>}
        {turn.answer && <AnswerView a={turn.answer} />}
      </div>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="space-y-2">
      <div className="skeleton h-3 w-1/3" />
      <div className="skeleton h-3 w-full" />
      <div className="skeleton h-3 w-11/12" />
      <div className="skeleton h-3 w-2/3" />
    </div>
  );
}

function AnswerView({ a }: { a: Answer }) {
  const pct = Math.round(a.confidence * 100);
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`chip ${a.refusal ? "!text-amber-300 !border-amber-500/40 !bg-amber-500/10" : "!text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10"}`}>
          {a.refusal ? "Refused" : "Answered"}
        </span>
        {a.provider && <span className="chip">{a.provider}{a.model ? ` · ${a.model}` : ""}</span>}
        <ConfidenceBar pct={pct} refused={a.refusal} />
      </div>

      <p className="text-zinc-100 whitespace-pre-wrap leading-relaxed">{a.answer}</p>

      {a.sources.length > 0 && a.sources[0].chunk_id !== "none" && (
        <div>
          <div className="text-xs uppercase tracking-wider text-zinc-500 mb-2">Sources</div>
          <div className="grid gap-2">
            {a.sources.map((s) => (
              <div key={s.chunk_id} className="bg-bg-elevated border border-bg-border rounded-xl p-3">
                <div className="flex items-center gap-2 text-xs text-zinc-400 font-mono mb-1">
                  <span className="text-accent">●</span> {s.chunk_id}
                </div>
                <p className="text-zinc-300 text-sm leading-relaxed">{s.quote}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {!a.refusal && <RatingWidget a={a} />}
    </div>
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
    } catch (e: any) {
      setErr(String(e.message || e));
      setState("error");
    }
  }

  if (state === "saved") {
    return (
      <div className="border-t border-bg-border pt-3 text-xs text-emerald-300">
        Rated {rating}/5 — thanks. Saved to data/human_ratings/ratings.jsonl.
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
        />
      </div>
      <span className="text-xs text-zinc-400 font-mono tabular-nums w-10 text-right">{pct}%</span>
    </div>
  );
}
