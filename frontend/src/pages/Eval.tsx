import { useQuery } from "@tanstack/react-query";
import { get, post } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import Tooltip from "../components/Tooltip";
import ErrorAlert from "../components/ErrorAlert";

type Run = {
  dataset: string;
  model?: string | null;
  n: number;
  aggregate?: { faithfulness?: number; answer_relevance?: number; context_precision?: number; context_recall?: number };
};

const METRICS: { key: keyof NonNullable<Run["aggregate"]>; label: string; desc: string }[] = [
  { key: "faithfulness", label: "Faithfulness", desc: "Answer only asserts what's in retrieved context" },
  { key: "answer_relevance", label: "Relevance", desc: "Answer addresses the question" },
  { key: "context_precision", label: "Precision", desc: "Retrieved chunks are on-topic" },
  { key: "context_recall", label: "Recall", desc: "Retrieved all relevant chunks" },
];

const GOLDEN_SAMPLES = [
  {
    q: "How does entity extraction in Graph RAG reduce hallucination compared to dense-only systems?",
    a: "Graph RAG extracts entities upfront and walks relationships, forcing the model to ground claims in the knowledge structure. Dense-only systems can drift into semantically plausible but unsupported answers because they lack structural constraints.",
  },
  {
    q: "When should you trade latency for reasoning capability in a RAG system?",
    a: "For synthesis queries requiring cross-document reasoning, Agentic RAG's tool-use loop justifies higher latency. For factual lookups, Classic RAG's speed wins. The tradeoff depends on query complexity and tolerance for latency.",
  },
  {
    q: "How do chunk size and embedding model interact in a retrieval system?",
    a: "Smaller chunks with powerful embeddings (e.g., text-embedding-3-large) excel at precision but risk fragmenting context. Larger chunks preserve context but may dilute relevance signals. The optimal pairing depends on document structure and query complexity.",
  },
];

export default function EvalPage() {
  const { data, refetch, isFetching, error, isLoading } = useQuery({
    queryKey: ["runs"],
    queryFn: () => get<Run[]>("/eval/runs"),
  });

  async function runNow() {
    try {
      await post("/eval/run", { dataset: "golden_v1" });
      refetch();
    } catch (e) {
      // Error will be handled by ErrorAlert
    }
  }

  return (
    <div className="flex flex-col gap-6 max-w-6xl">
      <header>
        <h1 className="display text-4xl font-semibold text-white">Evaluation</h1>
        <p className="text-sm text-zinc-400 mt-2">Golden dataset: 100,000 Q&A pairs. Measure strategy performance with precision, recall, and relevance metrics.</p>
      </header>

      {isLoading ? (
        <div className="card p-12">
          <LoadingSpinner message="Loading evaluation results..." />
        </div>
      ) : error ? (
        <ErrorAlert
          error={`Failed to load evaluations: ${error instanceof Error ? error.message : "Unknown error"}`}
          onRetry={() => refetch()}
        />
      ) : !data?.length ? (
        <div className="card p-12 text-center">
          <h2 className="text-xl font-semibold text-white mb-2">No evaluation runs yet</h2>
          <p className="text-sm text-zinc-400 mb-6">Run an evaluation on the golden dataset to measure retrieval strategy performance.</p>
          <button onClick={runNow} disabled={isFetching} className="btn-primary min-h-[48px]">
            {isFetching ? "Running…" : "Run Evaluation"}
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          {data.map((r, i) => (
            <div key={i} className="card p-6">
              <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-white">Evaluation Results</h2>
                  <p className="text-xs text-zinc-500 mt-1">Golden dataset • 100,000 questions</p>
                </div>
                <button onClick={runNow} disabled={isFetching} className="btn-primary text-sm px-4 py-2 min-h-[44px] sm:min-h-auto">
                  {isFetching ? "Running…" : "Re-run"}
                </button>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
                {METRICS.map((m) => {
                  const v = r.aggregate?.[m.key];
                  const pct = v == null ? 0 : Math.round(v * 100);

                  return (
                    <Tooltip key={m.key} label={m.desc} side="top">
                      <div className="p-4 rounded-lg bg-bg-surface border border-bg-border cursor-help">
                        <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">{m.label}</div>
                        <div className="text-2xl sm:text-3xl font-bold text-white mt-2 tabular-nums">
                          {v == null ? "–" : v.toFixed(2)}
                        </div>
                        <div className="mt-3 h-1.5 bg-bg-elevated rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-accent to-emerald-400 transition-all"
                            style={{ width: `${pct}%` }}
                            role="progressbar"
                            aria-valuenow={pct}
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-label={`${m.label}: ${v?.toFixed(2) || "N/A"}`}
                          />
                        </div>
                        <p className="text-[10px] sm:text-[11px] text-zinc-500 mt-2 hidden sm:block">{m.desc}</p>
                      </div>
                    </Tooltip>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Golden Dataset Examples */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Golden Dataset Examples</h2>
        <div className="space-y-4">
          {GOLDEN_SAMPLES.map((s, i) => (
            <div key={i} className="p-4 rounded-lg bg-bg-surface border border-bg-border">
              <div className="text-sm font-semibold text-white mb-2">Q{i + 1}: {s.q}</div>
              <div className="text-sm text-zinc-400 leading-relaxed pl-4 border-l border-accent">{s.a}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
