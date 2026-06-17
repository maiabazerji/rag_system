import { useEffect, useState } from "react";
import { get, post } from "../api/client";

type Source = { chunk_id: string; quote: string };
type TraceStep = { step?: string; [k: string]: unknown };
type StrategyOut = {
  strategy: "classic" | "graph" | "agentic";
  answer: string;
  sources: Source[];
  refusal: boolean;
  confidence: number;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  iterations: number;
  trace: TraceStep[];
  extra: Record<string, unknown>;
};
type CompareOut = { question: string; results: StrategyOut[] };

const STRATEGIES: StrategyOut["strategy"][] = ["classic", "graph", "agentic"];

const META: Record<
  StrategyOut["strategy"],
  { title: string; tagline: string; accent: string }
> = {
  classic: {
    title: "Classic RAG",
    tagline: "embed → vector search → rerank → answer",
    accent: "from-sky-500/30 to-sky-500/0",
  },
  graph: {
    title: "Graph RAG",
    tagline: "extract entities → walk knowledge graph → answer",
    accent: "from-violet-500/30 to-violet-500/0",
  },
  agentic: {
    title: "Agentic RAG",
    tagline: "Claude drives search & fetch tools in a loop",
    accent: "from-emerald-500/30 to-emerald-500/0",
  },
};

const SAMPLES = [
  "How does hybrid retrieval beat dense-only?",
  "What connects BM25, dense vectors, and reranking?",
  "Compare faithfulness and answer relevance — which matters more?",
];

export default function Compare() {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<CompareOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [graphStats, setGraphStats] = useState<{ triples: number; entities: number } | null>(null);
  const [buildingGraph, setBuildingGraph] = useState(false);

  async function refreshStats() {
    try {
      setGraphStats(await get<{ triples: number; entities: number }>("/graph/stats"));
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    refreshStats();
  }, []);

  async function run() {
    setErr(null);
    setBusy(true);
    setRes(null);
    try {
      const r = await post<CompareOut>("/compare/strategies", {
        question: q,
        strategies: STRATEGIES,
      });
      setRes(r);
    } catch (e: any) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function buildGraph() {
    setBuildingGraph(true);
    try {
      await post("/graph/build", {});
      await refreshStats();
    } catch (e: any) {
      setErr(String(e.message || e));
    } finally {
      setBuildingGraph(false);
    }
  }

  const byName = (n: StrategyOut["strategy"]) => res?.results.find((r) => r.strategy === n);

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-4xl font-semibold text-white">Compare strategies</h1>
        <p className="text-zinc-400 max-w-2xl">
          Same question, three RAG flavors —{" "}
          <span className="text-sky-300">Classic</span>,{" "}
          <span className="text-violet-300">Graph</span>, and{" "}
          <span className="text-emerald-300">Agentic</span>. All grounded in your indexed
          documents, all answered by Claude.
        </p>
      </header>

      <div className="card flex flex-col gap-3">
        <textarea
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (q.trim()) run();
            }
          }}
          rows={2}
          placeholder="Ask something… ⏎ to run"
          className="input resize-none"
        />
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-1.5">
            {SAMPLES.map((s) => (
              <button
                key={s}
                onClick={() => setQ(s)}
                className="chip hover:text-white hover:border-accent/40"
              >
                {s}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <GraphStatus
              stats={graphStats}
              busy={buildingGraph}
              onBuild={buildGraph}
            />
            <button
              onClick={run}
              disabled={!q.trim() || busy}
              className="btn-primary"
            >
              {busy ? "Running…" : "Compare 3 strategies"}
            </button>
          </div>
        </div>
        {err && <div className="text-rose-400 text-sm">{err}</div>}
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        {STRATEGIES.map((s) => (
          <StrategyCard key={s} name={s} result={byName(s)} loading={busy} />
        ))}
      </div>

      {res && <WinnersBar results={res.results} />}
    </div>
  );
}

function GraphStatus({
  stats,
  busy,
  onBuild,
}: {
  stats: { triples: number; entities: number } | null;
  busy: boolean;
  onBuild: () => void;
}) {
  const ready = !!stats && stats.triples > 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`chip ${ready ? "!text-emerald-300 !border-emerald-500/40" : "!text-amber-300 !border-amber-500/40"}`}>
        graph: {stats ? `${stats.triples} triples, ${stats.entities} entities` : "—"}
      </span>
      <button
        onClick={onBuild}
        disabled={busy}
        className="btn-ghost h-8 px-3 text-xs"
        title="Extract triples from indexed chunks using Claude Haiku"
      >
        {busy ? "Building…" : ready ? "Rebuild" : "Build graph"}
      </button>
    </div>
  );
}

function StrategyCard({
  name,
  result,
  loading,
}: {
  name: StrategyOut["strategy"];
  result?: StrategyOut;
  loading: boolean;
}) {
  const m = META[name];
  return (
    <div className="card relative overflow-hidden flex flex-col gap-4 min-h-[300px]">
      <div className={`absolute inset-x-0 top-0 h-24 bg-gradient-to-b ${m.accent} pointer-events-none`} />
      <div className="relative flex flex-col gap-1">
        <div className="display text-lg font-semibold text-white">{m.title}</div>
        <div className="text-[11px] text-zinc-400 font-mono">{m.tagline}</div>
      </div>

      {loading && !result && <Skeleton />}

      {result && (
        <>
          <Telemetry r={result} />
          {result.refusal ? (
            <div className="text-amber-300 text-sm whitespace-pre-wrap">{result.answer}</div>
          ) : (
            <p className="text-zinc-100 whitespace-pre-wrap text-sm leading-relaxed">
              {result.answer}
            </p>
          )}

          {result.sources.length > 0 && result.sources[0].chunk_id !== "none" && (
            <Sources sources={result.sources} />
          )}

          {result.extra && Object.keys(result.extra).length > 0 && (
            <ExtraDetails name={name} extra={result.extra} />
          )}

          {result.trace.length > 0 && <TraceDetails trace={result.trace} />}
        </>
      )}
    </div>
  );
}

function Telemetry({ r }: { r: StrategyOut }) {
  const total = r.input_tokens + r.output_tokens;
  return (
    <div className="flex flex-wrap gap-1.5 text-[11px] font-mono">
      <span className="chip">{r.latency_ms} ms</span>
      <span className="chip">{total.toLocaleString()} tok</span>
      <span className="chip" title="input → output tokens">
        in {r.input_tokens.toLocaleString()} · out {r.output_tokens.toLocaleString()}
      </span>
      {r.iterations > 1 && <span className="chip">{r.iterations} iters</span>}
      <span className={`chip ${r.refusal ? "!text-amber-300 !border-amber-500/40" : "!text-emerald-300 !border-emerald-500/40"}`}>
        {r.refusal ? "refused" : "answered"}
      </span>
    </div>
  );
}

function Sources({ sources }: { sources: Source[] }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5">Sources</div>
      <div className="text-xs text-zinc-400 font-mono space-y-0.5 max-h-32 overflow-y-auto">
        {sources.map((s) => (
          <div key={s.chunk_id} title={s.quote}>
            ● {s.chunk_id}
          </div>
        ))}
      </div>
    </div>
  );
}

function ExtraDetails({
  name,
  extra,
}: {
  name: StrategyOut["strategy"];
  extra: Record<string, unknown>;
}) {
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-zinc-500 hover:text-zinc-300 select-none">
        {name === "graph" ? "Entities + subgraph" : "Strategy details"}
      </summary>
      <pre className="mt-2 bg-bg-elevated rounded-md p-2 overflow-x-auto text-[11px] text-zinc-300 whitespace-pre-wrap">
        {JSON.stringify(extra, null, 2)}
      </pre>
    </details>
  );
}

function TraceDetails({ trace }: { trace: TraceStep[] }) {
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-zinc-500 hover:text-zinc-300 select-none">
        Reasoning trace ({trace.length} steps)
      </summary>
      <ol className="mt-2 space-y-1 list-decimal pl-5 text-[11px] text-zinc-300">
        {trace.map((t, i) => (
          <li key={i} className="font-mono break-words">
            <span className="text-accent">{t.step ?? "step"}</span>{" "}
            <span className="text-zinc-400">{JSON.stringify(rest(t))}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

function rest(t: TraceStep) {
  const r: Record<string, unknown> = {};
  for (const k of Object.keys(t)) if (k !== "step") r[k] = (t as Record<string, unknown>)[k];
  return r;
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

function WinnersBar({ results }: { results: StrategyOut[] }) {
  const ok = results.filter((r) => !r.refusal);
  if (ok.length < 2) return null;
  const fastest = ok.reduce((a, b) => (a.latency_ms <= b.latency_ms ? a : b));
  const cheapest = ok.reduce((a, b) =>
    a.input_tokens + a.output_tokens <= b.input_tokens + b.output_tokens ? a : b
  );
  return (
    <div className="card !py-3 flex flex-wrap items-center gap-3 text-xs">
      <span className="text-zinc-500 uppercase tracking-wider">Winners</span>
      <span className="chip">⚡ fastest: {fastest.strategy} ({fastest.latency_ms} ms)</span>
      <span className="chip">💸 cheapest: {cheapest.strategy} ({cheapest.input_tokens + cheapest.output_tokens} tok)</span>
      <span className="text-zinc-500 ml-auto">
        Which "wins" depends on your question. Read each answer — the right pick balances speed, cost, and groundedness.
      </span>
    </div>
  );
}
