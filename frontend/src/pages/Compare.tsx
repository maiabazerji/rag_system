import { useEffect, useState, ReactNode } from "react";
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
    tagline: "model loops over search & fetch tools",
    accent: "from-emerald-500/30 to-emerald-500/0",
  },
};

const SAMPLES = [
  "How does hybrid retrieval beat dense-only?",
  "What connects BM25, dense vectors, and reranking?",
  "Compare faithfulness and answer relevance. Which matters more?",
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
          Same question, three RAG flavors  - {" "}
          <span className="text-sky-300">Classic</span>,{" "}
          <span className="text-violet-300">Graph</span>, and{" "}
          <span className="text-emerald-300">Agentic</span>. All grounded in your indexed
          documents, all answered by the LLM.
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
    <div className="flex items-center gap-2">
      <div className={`text-xs font-mono px-2.5 py-1.5 rounded border flex items-center gap-2 ${
        ready
          ? "bg-violet-500/10 border-violet-500/40 text-violet-300"
          : "bg-amber-500/10 border-amber-500/40 text-amber-300"
      }`}>
        <span className={`w-2 h-2 rounded-full ${ready ? "bg-violet-400" : "bg-amber-400"}`} />
        {stats ? (
          <span>
            <span className="font-semibold">{stats.triples}</span> triples
          </span>
        ) : (
          <span>Graph not built</span>
        )}
      </div>
      <button
        onClick={onBuild}
        disabled={busy}
        className={`text-xs px-3 py-1.5 rounded font-medium transition-colors ${
          busy
            ? "bg-zinc-700/40 text-zinc-500 cursor-not-allowed"
            : "bg-violet-500/20 border border-violet-500/40 text-violet-300 hover:bg-violet-500/30 hover:border-violet-500/60"
        }`}
        title="Extract entity relationships from indexed documents"
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
            <div className="text-amber-300 text-sm p-3 bg-amber-500/10 rounded-md border border-amber-500/20">
              {result.answer}
            </div>
          ) : (
            <div className="text-zinc-100 text-sm leading-relaxed answer-content">
              <FormattedAnswer text={result.answer} />
            </div>
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
    <div className="grid grid-cols-2 sm:flex sm:flex-wrap gap-2 text-[10px]">
      <div className="flex items-center gap-1 bg-sky-500/10 border border-sky-500/30 rounded px-2 py-1">
        <span className="text-sky-300 font-semibold">⏱</span>
        <span className="text-sky-200 font-mono">{r.latency_ms}ms</span>
      </div>
      <div className="flex items-center gap-1 bg-emerald-500/10 border border-emerald-500/30 rounded px-2 py-1">
        <span className="text-emerald-300 font-semibold">💰</span>
        <span className="text-emerald-200 font-mono">{total.toLocaleString()} tok</span>
      </div>
      <div className="flex items-center gap-1 bg-zinc-700/40 border border-zinc-600/40 rounded px-2 py-1 col-span-2 sm:col-span-1 text-[9px]" title="input → output tokens">
        <span className="text-zinc-300 font-mono">{r.input_tokens.toLocaleString()} in</span>
        <span className="text-zinc-400">•</span>
        <span className="text-zinc-300 font-mono">{r.output_tokens.toLocaleString()} out</span>
      </div>
      {r.iterations > 1 && (
        <div className="flex items-center gap-1 bg-purple-500/10 border border-purple-500/30 rounded px-2 py-1">
          <span className="text-purple-300 font-semibold">🔄</span>
          <span className="text-purple-200 font-mono">{r.iterations} iters</span>
        </div>
      )}
      <div className={`flex items-center gap-1 rounded px-2 py-1 col-span-2 sm:col-span-1 ${r.refusal ? "bg-amber-500/10 border border-amber-500/30" : "bg-emerald-500/10 border border-emerald-500/30"}`}>
        <span className={r.refusal ? "text-amber-300 text-lg" : "text-emerald-300 text-lg"}>
          {r.refusal ? "⚠" : "✓"}
        </span>
        <span className={r.refusal ? "text-amber-200 text-[10px] font-semibold" : "text-emerald-200 text-[10px] font-semibold"}>
          {r.refusal ? "refused" : "answered"}
        </span>
      </div>
    </div>
  );
}

function Sources({ sources }: { sources: Source[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2 font-semibold">Sources ({sources.length})</div>
      <div className="space-y-1.5">
        {sources.map((s, i) => (
          <div
            key={`${s.chunk_id}-${i}`}
            className="bg-zinc-900/50 border border-zinc-700/50 rounded-md p-2 hover:border-zinc-600/70 transition-colors cursor-pointer"
            onClick={() => setExpanded(expanded === s.chunk_id ? null : s.chunk_id)}
          >
            <div className="flex items-start gap-2">
              <span className="text-accent text-[10px] font-mono flex-shrink-0 mt-0.5">
                [{i + 1}]
              </span>
              <div className="flex-1 min-w-0">
                <div className="text-[10px] text-zinc-400 font-mono truncate">{s.chunk_id}</div>
                {expanded === s.chunk_id && (
                  <div className="text-[11px] text-zinc-300 mt-1.5 leading-relaxed max-h-24 overflow-y-auto border-t border-zinc-700/50 pt-1.5">
                    {s.quote || "(empty quote)"}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function FormattedAnswer({ text }: { text: string }) {
  const renderInline = (str: string): ReactNode => {
    const parts: ReactNode[] = [];
    let lastIndex = 0;

    // Match **bold**, *italic*, and code`
    const regex = /\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`/g;
    let match;

    while ((match = regex.exec(str)) !== null) {
      // Add text before match
      if (match.index > lastIndex) {
        parts.push(str.substring(lastIndex, match.index));
      }

      // Add styled content
      if (match[1]) {
        // Bold
        parts.push(
          <strong key={`b-${match.index}`} className="font-semibold text-zinc-100">
            {match[1]}
          </strong>
        );
      } else if (match[2]) {
        // Italic
        parts.push(
          <em key={`i-${match.index}`} className="italic text-zinc-200">
            {match[2]}
          </em>
        );
      } else if (match[3]) {
        // Code
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

    // Add remaining text
    if (lastIndex < str.length) {
      parts.push(str.substring(lastIndex));
    }

    return parts.length > 0 ? parts : str;
  };

  const lines = text.split("\n");
  const elements = [];
  let inList = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      elements.push(<div key={`br-${i}`} className="h-2" />);
      inList = false;
      continue;
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
        elements.push(<ul key={`ul-${i}`} className="space-y-1 ml-4 mt-1 mb-2" />);
      }
      const item = trimmed.substring(2);
      elements.push(
        <li key={`li-${i}`} className="text-zinc-200 text-sm list-disc ml-0">
          {renderInline(item)}
        </li>
      );
    } else if (trimmed.startsWith("|")) {
      inList = false;
      elements.push(
        <div key={`table-${i}`} className="text-[11px] font-mono text-zinc-300 overflow-x-auto">
          {trimmed}
        </div>
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

  return <div className="space-y-2">{elements}</div>;
}

function ExtraDetails({
  name,
  extra,
}: {
  name: StrategyOut["strategy"];
  extra: Record<string, unknown>;
}) {
  if (name === "graph") {
    return (
      <details className="text-xs">
        <summary className="cursor-pointer text-zinc-500 hover:text-zinc-300 select-none">
          Knowledge graph details
        </summary>
        <div className="mt-3 space-y-3">
          {extra.entities && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5 font-semibold">
                Extracted Entities
              </div>
              <div className="flex flex-wrap gap-1">
                {Array.isArray(extra.entities) &&
                  extra.entities.map((e, i) => (
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
          {extra.related_entities && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5 font-semibold">
                Related Entities (1 hop)
              </div>
              <div className="flex flex-wrap gap-1">
                {Array.isArray(extra.related_entities) &&
                  extra.related_entities.map((e, i) => (
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
          {extra.subgraph && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5 font-semibold">
                Knowledge Graph Structure
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
        Strategy details
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
      <summary className="cursor-pointer text-zinc-500 hover:text-zinc-300 select-none font-semibold">
        🔍 Trace ({trace.length} steps)
      </summary>
      <div className="mt-2 space-y-1.5 text-[10px]">
        {trace.map((t, i) => (
          <div key={i} className="flex gap-2 items-start">
            <div className="flex-shrink-0 text-zinc-600 font-mono">{String(i + 1).padStart(2, "0")}.</div>
            <div className="flex-1 bg-zinc-900/50 border border-zinc-700/50 rounded px-2 py-1">
              <div className="font-mono text-accent">{t.step ?? "step"}</div>
              <div className="text-zinc-400 mt-0.5">{JSON.stringify(rest(t))}</div>
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

function Skeleton() {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        <div className="skeleton h-5 w-16" />
        <div className="skeleton h-5 w-20" />
        <div className="skeleton h-5 w-24" />
      </div>
      <div className="space-y-2">
        <div className="skeleton h-3 w-full" />
        <div className="skeleton h-3 w-11/12" />
        <div className="skeleton h-3 w-5/6" />
        <div className="skeleton h-3 w-4/5" />
      </div>
      <div className="space-y-1.5 pt-2 border-t border-zinc-700/50">
        <div className="skeleton h-2.5 w-full" />
        <div className="skeleton h-2.5 w-2/3" />
      </div>
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

  const strategyLabel = (s: string) => {
    const labels: Record<string, string> = {
      classic: "Classic RAG",
      graph: "Graph RAG",
      agentic: "Agentic RAG",
    };
    return labels[s] || s;
  };

  const strategyColor = (s: string) => {
    const colors: Record<string, string> = {
      classic: "bg-sky-500/20 border-sky-500/50 text-sky-300",
      graph: "bg-violet-500/20 border-violet-500/50 text-violet-300",
      agentic: "bg-emerald-500/20 border-emerald-500/50 text-emerald-300",
    };
    return colors[s] || "bg-zinc-500/20 border-zinc-500/50 text-zinc-300";
  };

  return (
    <div className="card flex flex-col gap-3">
      <div className="space-y-1">
        <h3 className="display text-sm font-semibold text-white">Performance Summary</h3>
        <p className="text-xs text-zinc-400">Which "wins" depends on your question. Read each answer- the right pick balances speed, cost, and groundedness.</p>
      </div>
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="flex-1 p-3 rounded-md bg-sky-500/10 border border-sky-500/30">
          <div className="text-[10px] uppercase tracking-wider text-sky-400 font-semibold mb-1">⚡ Fastest</div>
          <div className={`text-sm font-semibold border border-sky-500/50 rounded px-2 py-1.5 text-center ${strategyColor(fastest.strategy)}`}>
            {strategyLabel(fastest.strategy)}
          </div>
          <div className="text-[11px] text-sky-200 text-center mt-1 font-mono">{fastest.latency_ms}ms</div>
        </div>
        <div className="flex-1 p-3 rounded-md bg-emerald-500/10 border border-emerald-500/30">
          <div className="text-[10px] uppercase tracking-wider text-emerald-400 font-semibold mb-1">💸 Cheapest</div>
          <div className={`text-sm font-semibold border rounded px-2 py-1.5 text-center ${strategyColor(cheapest.strategy)}`}>
            {strategyLabel(cheapest.strategy)}
          </div>
          <div className="text-[11px] text-emerald-200 text-center mt-1 font-mono">{cheapest.input_tokens + cheapest.output_tokens} tokens</div>
        </div>
      </div>
    </div>
  );
}
