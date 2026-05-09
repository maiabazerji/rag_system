import { useState } from "react";
import { post } from "../api/client";

type Source = { chunk_id: string; quote: string };
type AnswerOut = {
  question: string;
  answer: string;
  sources: Source[];
  confidence: number;
  refusal: boolean;
};
type CompareResult = { question: string; results: AnswerOut[] };

const VARIANTS = [
  { model: "claude-sonnet-4-6", prompt_version: "default" },
  { model: "claude-opus-4-6", prompt_version: "default" },
];

export default function Compare() {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try {
      setRes(await post<CompareResult>("/compare", { question: q, variants: VARIANTS }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-4xl font-semibold text-white">Compare</h1>
        <p className="text-zinc-400 max-w-xl">
          Same question, different model or prompt — read the answers side by side and judge which is grounded better.
        </p>
      </header>

      <div className="card flex flex-col gap-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && q && run()}
          placeholder="Question to compare…"
          className="input"
        />
        <div className="flex items-center justify-between">
          <div className="flex flex-wrap gap-1.5">
            {VARIANTS.map((v, i) => (
              <span key={i} className="chip">
                {v.model} · {v.prompt_version}
              </span>
            ))}
          </div>
          <button onClick={run} disabled={!q || busy} className="btn-primary">
            {busy ? "Running…" : "Compare"}
          </button>
        </div>
      </div>

      {res && (
        <div className="grid md:grid-cols-2 gap-4">
          {res.results.map((r, i) => (
            <div key={i} className="card animate-fade-in">
              <div className="flex items-center justify-between mb-3">
                <span className="chip">{VARIANTS[i]?.model ?? `variant ${i}`}</span>
                <span className="text-xs text-zinc-500 font-mono">
                  conf {Math.round(r.confidence * 100)}%
                </span>
              </div>
              <p className="text-zinc-100 whitespace-pre-wrap text-sm leading-relaxed">{r.answer}</p>
              {r.sources.length > 0 && r.sources[0].chunk_id !== "none" && (
                <div className="mt-3 pt-3 border-t border-bg-border">
                  <div className="text-[11px] uppercase tracking-wider text-zinc-500 mb-1.5">Sources</div>
                  <div className="text-xs text-zinc-400 font-mono space-y-1">
                    {r.sources.map((s) => (
                      <div key={s.chunk_id}>{s.chunk_id}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
