import { useEffect, useRef, useState, ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, errorMessage, get, getAdminKey, post, setAdminKey } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorAlert from "../components/ErrorAlert";
import Tooltip from "../components/Tooltip";
import { formatLatency, formatTokens } from "../utils/formatting";

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
type GraphStats = { triples: number; entities: number };
/** POST /graph/build answers 202 with a job; GET /graph/build/{id} reports on it. */
type BuildJob = {
  id?: string;
  job_id?: string;
  status: "queued" | "running" | "completed" | "failed";
  total?: number;
  completed?: number;
  failures?: string[];
  error?: string;
};

const POLL_INTERVAL_MS = 2000;

/** Graph build and reset need the admin key on top of any API key. */
function graphBuildError(e: unknown): string {
  if (e instanceof ApiError && e.status === 403) {
    return (
      "Building the graph needs the backend's admin key (ADMIN_KEY). Enter it in the " +
      `"Admin key" field below and try again.${e.detail ? ` (${e.detail})` : ""}`
    );
  }
  if (e instanceof ApiError && e.status === 503) {
    return (
      "Graph build is unavailable: the backend has no ADMIN_KEY configured, or a backing " +
      `service is down. Set ADMIN_KEY in .env and restart the backend.${e.detail ? ` (${e.detail})` : ""}`
    );
  }
  return errorMessage(e);
}
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

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
  "How does entity extraction in Graph RAG reduce hallucination compared to dense-only retrieval?",
  "When should you prioritize Agentic RAG's reasoning over Classic RAG's speed?",
  "What happens to retrieval quality when you use smaller chunks with dense embeddings?",
];

export default function Compare() {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<CompareOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [buildingGraph, setBuildingGraph] = useState(false);
  const [buildProgress, setBuildProgress] = useState<string | null>(null);
  const [showAdminKey, setShowAdminKey] = useState(false);
  const { data: graphStats = null, refetch: refetchStats } = useQuery({
    queryKey: ["graph-stats"],
    queryFn: () => get<GraphStats>("/graph/stats"),
  });

  // Stops the build poller when the page unmounts.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  async function run() {
    if (busy) return;
    setErr(null);
    setBusy(true);
    setRes(null);
    try {
      const r = await post<CompareOut>("/compare/strategies", {
        question: q,
        strategies: STRATEGIES,
      });
      setRes(r);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  /** Start a build, then poll its job until it finishes. It can run for minutes. */
  async function buildGraph() {
    if (buildingGraph) return;
    setErr(null);
    setBuildingGraph(true);
    setBuildProgress("queued");
    try {
      const started = await post<BuildJob>("/graph/build", {}, { admin: true });
      const jobId = started.job_id ?? started.id;
      if (!jobId) throw new Error("The backend did not return a graph build job id.");
      let job = started;
      while (mounted.current && job.status !== "completed" && job.status !== "failed") {
        await sleep(POLL_INTERVAL_MS);
        if (!mounted.current) return;
        job = await get<BuildJob>(`/graph/build/${encodeURIComponent(jobId)}`, { admin: true });
        setBuildProgress(
          job.total ? `${job.completed ?? 0}/${job.total} chunks` : job.status,
        );
      }
      if (!mounted.current) return;
      if (job.status === "failed") {
        setErr(`Graph build failed: ${job.error ?? "unknown error"}`);
      } else if (job.failures && job.failures.length > 0) {
        setErr(
          `Graph built, but extraction failed for ${job.failures.length} chunk(s). Rebuild to retry them.`,
        );
      }
      await refetchStats();
    } catch (e) {
      if (!mounted.current) return;
      setErr(graphBuildError(e));
      if (e instanceof ApiError && e.status === 403) setShowAdminKey(true);
    } finally {
      if (mounted.current) {
        setBuildingGraph(false);
        setBuildProgress(null);
      }
    }
  }

  const byName = (n: StrategyOut["strategy"]) => res?.results.find((r) => r.strategy === n);

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-4xl font-semibold text-white">Compare</h1>
        <p className="text-sm text-zinc-400">
          <span className="text-sky-300">Classic</span> • <span className="text-violet-300">Graph</span> • <span className="text-emerald-300">Agentic</span>
        </p>
      </header>

      <div className="card flex flex-col gap-1.5 p-2">
        <textarea
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (q.trim() && !busy) run();
            }
          }}
          rows={2}
          placeholder="Ask something… ⏎ to run"
          className="input resize-none text-sm"
        />
        <div className="flex items-center justify-between gap-2 flex-wrap sm:flex-nowrap">
          <div className="flex flex-wrap gap-1 w-full sm:w-auto">
            {SAMPLES.map((s) => (
              <button
                key={s}
                onClick={() => setQ(s)}
                className="chip text-xs hover:text-white hover:border-accent/40 min-h-[44px] sm:min-h-auto"
              >
                {s}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5 w-full sm:w-auto flex-wrap sm:flex-nowrap">
            <GraphStatus
              stats={graphStats}
              busy={buildingGraph}
              progress={buildProgress}
              onBuild={buildGraph}
            />
            <button
              onClick={run}
              disabled={!q.trim() || busy}
              className="btn-primary text-sm px-3 py-2 min-h-[48px] sm:min-h-auto"
            >
              {busy ? "Running…" : "Compare"}
            </button>
          </div>
        </div>
        {err && (
          <ErrorAlert
            error={err}
            onRetry={() => {
              setErr(null);
              if (q.trim() && !busy) run();
            }}
            onDismiss={() => setErr(null)}
          />
        )}
        <AdminKeyField open={showAdminKey} onToggle={setShowAdminKey} />
      </div>

      {res && <WinnersBar results={res.results} />}

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2">
        {STRATEGIES.map((s) => (
          <StrategyCard key={s} name={s} result={byName(s)} loading={busy} />
        ))}
      </div>
    </div>
  );
}

/** Optional admin key, sent as X-Admin-Key on graph build. Kept in this browser only. */
function AdminKeyField({ open, onToggle }: { open: boolean; onToggle: (open: boolean) => void }) {
  const [value, setValue] = useState(getAdminKey() ?? "");
  const [stored, setStored] = useState(!!getAdminKey());

  function save(e: React.FormEvent) {
    e.preventDefault();
    setAdminKey(value);
    setStored(!!value.trim());
  }

  function clear() {
    setAdminKey(null);
    setValue("");
    setStored(false);
  }

  return (
    <details
      open={open}
      onToggle={(e) => onToggle((e.currentTarget as HTMLDetailsElement).open)}
      className="text-xs text-zinc-500"
    >
      <summary className="cursor-pointer select-none hover:text-zinc-300">
        Admin key {stored ? "(saved)" : "(needed to build the graph)"}
      </summary>
      <form onSubmit={save} className="mt-2 flex gap-2 flex-wrap items-center">
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="ADMIN_KEY from .env"
          autoComplete="off"
          spellCheck={false}
          aria-label="Admin key"
          className="flex-1 min-w-[14rem] rounded border border-bg-border bg-bg-base px-3 py-1.5 text-sm font-mono text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={!value.trim()}
          className="rounded border border-bg-border px-3 py-1.5 text-sm text-zinc-200 hover:border-accent disabled:opacity-40 transition-colors"
        >
          Save
        </button>
        {stored && (
          <button
            type="button"
            onClick={clear}
            className="rounded px-3 py-1.5 text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            Clear
          </button>
        )}
        <span className="basis-full text-[11px] text-zinc-600">
          Sent as <code className="font-mono">X-Admin-Key</code> with graph builds only, and kept in
          this browser.
        </span>
      </form>
    </details>
  );
}

function GraphStatus({
  stats,
  busy,
  progress,
  onBuild,
}: {
  stats: GraphStats | null;
  busy: boolean;
  progress: string | null;
  onBuild: () => void;
}) {
  const ready = !!stats && stats.triples > 0;
  return (
    <div className="flex items-center gap-2">
      <Tooltip label="Knowledge graph: extracted entities and relationships from documents" side="top">
        <div className={`text-xs font-mono px-2.5 py-1.5 rounded border flex items-center gap-2 cursor-help ${
          ready
            ? "bg-violet-500/10 border-violet-500/40 text-violet-300"
            : "bg-amber-500/10 border-amber-500/40 text-amber-300"
        }`}>
          <span className={`w-2 h-2 rounded-full ${ready ? "bg-violet-400" : "bg-amber-400"}`} aria-hidden="true" />
          {stats ? (
            <span>
              <span className="font-semibold">{stats.triples}</span> triples
            </span>
          ) : (
            <span>Graph not built</span>
          )}
        </div>
      </Tooltip>
      <button
        onClick={onBuild}
        disabled={busy}
        className={`text-xs px-3 py-1.5 rounded font-medium transition-colors min-h-[44px] flex items-center ${
          busy
            ? "bg-zinc-700/40 text-zinc-500 cursor-not-allowed"
            : "bg-violet-500/20 border border-violet-500/40 text-violet-300 hover:bg-violet-500/30 hover:border-violet-500/60"
        }`}
        aria-label="Extract entity relationships from indexed documents"
      >
        {busy ? `Building… ${progress ?? ""}`.trim() : ready ? "Rebuild" : "Build graph"}
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
  const [expanded, setExpanded] = useState(false);

  const truncatedAnswer = (text: string, sentences = 2) => {
    const sents = text.split(/(?<=[.!?])\s+/);
    const truncated = sents.slice(0, sentences).join(" ");
    return { text: truncated, isTruncated: sents.length > sentences };
  };

  const answerPreview = result && !result.refusal ? truncatedAnswer(result.answer, 2) : null;

  return (
    <div className="card relative overflow-hidden flex flex-col gap-1.5 min-h-[280px] p-2">
      <div className={`absolute inset-x-0 top-0 h-12 bg-gradient-to-b ${m.accent} pointer-events-none`} />

      <div className="relative flex flex-col gap-0.5">
        <div className="text-sm font-bold text-white">{m.title}</div>
        <div className="text-[8px] text-zinc-500 font-mono">{m.tagline}</div>
      </div>

      {loading && !result && <LoadingSpinner />}

      {result && (
        <>
          {/* Compact metrics badges */}
          <div className="grid grid-cols-2 gap-2">
            <Tooltip label="Time from request to response" side="top">
              <div className="bg-sky-500/20 border border-sky-500/40 rounded p-2 cursor-help">
                <div className="text-[8px] text-sky-300 font-bold uppercase">⏱ Latency</div>
                <div className="text-lg font-mono font-bold text-sky-100 mt-0.5">{formatLatency(result.latency_ms)}</div>
              </div>
            </Tooltip>
            <Tooltip label="Input + output tokens used (cost indicator)" side="top">
              <div className="bg-emerald-500/20 border border-emerald-500/40 rounded p-2 cursor-help">
                <div className="text-[8px] text-emerald-300 font-bold uppercase">💰 Tokens</div>
                <div className="text-lg font-mono font-bold text-emerald-100 mt-0.5">{formatTokens(result.input_tokens + result.output_tokens)}</div>
              </div>
            </Tooltip>
          </div>

          {/* Status badge */}
          <div className={`flex items-center gap-1.5 rounded px-2 py-1.5 text-xs font-semibold min-h-[44px] flex-wrap sm:min-h-auto ${result.refusal ? "bg-amber-500/15 border border-amber-500/40 text-amber-300" : "bg-emerald-500/15 border border-emerald-500/40 text-emerald-300"}`}>
            <span aria-hidden="true">{result.refusal ? "⚠" : "✓"}</span>
            <span>{result.refusal ? "Refused" : "Answered"}</span>
            {result.iterations > 1 && (
              <Tooltip label="Number of reasoning iterations (loops)" side="top">
                <span className="text-[10px] text-zinc-400 ml-auto cursor-help">🔄 {result.iterations}x</span>
              </Tooltip>
            )}
          </div>

          {/* Answer preview or full text */}
          {result.refusal ? (
            <div className="text-amber-200 text-xs p-2 bg-amber-500/10 rounded border border-amber-500/30">
              {result.answer}
            </div>
          ) : (
            <div className="flex-1 flex flex-col gap-1">
              <div className={`text-zinc-200 text-xs leading-relaxed answer-content ${!expanded ? "line-clamp-4" : ""}`}>
                <FormattedAnswer text={expanded ? result.answer : answerPreview?.text || result.answer} />
              </div>
              {answerPreview?.isTruncated && (
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="text-[10px] text-accent hover:text-accent/80 font-semibold self-start"
                >
                  {expanded ? "Show less" : "Read more →"}
                </button>
              )}
            </div>
          )}

          {/* Sources (collapsed by default) */}
          {!result.refusal && result.sources.length > 0 && result.sources[0].chunk_id !== "none" && (
            <div className="text-[9px] text-zinc-500">
              <details>
                <summary className="cursor-pointer font-semibold hover:text-zinc-300">Sources ({result.sources.length})</summary>
                <div className="mt-1 space-y-1">
                  {result.sources.slice(0, 3).map((s, i) => (
                    <div key={i} className="text-[8px] text-zinc-400 font-mono truncate">
                      [{i + 1}] {s.chunk_id}
                    </div>
                  ))}
                  {result.sources.length > 3 && <div className="text-[8px] text-zinc-500">+{result.sources.length - 3} more</div>}
                </div>
              </details>
            </div>
          )}

          {/* Reasoning trace and strategy-specific detail */}
          {result.extra && Object.keys(result.extra).length > 0 && (
            <ExtraDetails name={name} extra={result.extra} />
          )}
          {result.trace && result.trace.length > 0 && (
            <TraceDetails trace={result.trace} />
          )}
        </>
      )}
    </div>
  );
}




function FormattedAnswer({ text }: { text: string }) {
  const renderInline = (str: string): ReactNode => {
    const parts: ReactNode[] = [];
    let lastIndex = 0;

    // Match **bold**, *italic*, and `code`
    const regex = /\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`/g;
    let match;

    while ((match = regex.exec(str)) !== null) {
      if (match.index > lastIndex) {
        parts.push(str.substring(lastIndex, match.index));
      }

      if (match[1]) {
        parts.push(
          <strong key={`b-${match.index}`} className="font-semibold text-zinc-100">
            {match[1]}
          </strong>
        );
      } else if (match[2]) {
        parts.push(
          <em key={`i-${match.index}`} className="italic text-zinc-200">
            {match[2]}
          </em>
        );
      } else if (match[3]) {
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

    if (lastIndex < str.length) {
      parts.push(str.substring(lastIndex));
    }

    return parts.length > 0 ? parts : str;
  };

  const lines = text.split("\n");
  const elements = [];
  let inList = false;
  let tableLines: string[] = [];

  const flushTable = () => {
    if (tableLines.length > 0) {
      const rows = tableLines.map(l => l.split("|").map(c => c.trim()).filter(c => c));
      if (rows.length > 1) {
        elements.push(
          <table key={`table-${elements.length}`} className="text-xs border-collapse border border-zinc-600 my-2">
            <thead>
              <tr className="bg-zinc-800">
                {rows[0].map((cell, i) => (
                  <th key={i} className="border border-zinc-600 px-2 py-1 text-zinc-200 font-semibold">
                    {cell}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(2).map((row, ri) => (
                <tr key={ri} className="hover:bg-zinc-800/50">
                  {row.map((cell, ci) => (
                    <td key={ci} className="border border-zinc-600 px-2 py-1 text-zinc-300">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        );
      }
      tableLines = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      flushTable();
      elements.push(<div key={`br-${i}`} className="h-2" />);
      inList = false;
      continue;
    }

    if (trimmed.startsWith("|")) {
      inList = false;
      tableLines.push(trimmed);
      continue;
    } else {
      flushTable();
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
      }
      const item = trimmed.substring(2);
      elements.push(
        <li key={`li-${i}`} className="text-zinc-200 text-sm list-disc ml-4 mt-1">
          {renderInline(item)}
        </li>
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

  flushTable();
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
          {Array.isArray(extra.entities) && extra.entities.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5 font-semibold">
                Extracted Entities
              </div>
              <div className="flex flex-wrap gap-1">
                {extra.entities.map((e, i) => (
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
          {Array.isArray(extra.related_entities) && extra.related_entities.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5 font-semibold">
                Related Entities (1 hop)
              </div>
              <div className="flex flex-wrap gap-1">
                {extra.related_entities.map((e, i) => (
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
          {typeof extra.subgraph === "string" && extra.subgraph.length > 0 && (
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


function WinnersBar({ results }: { results: StrategyOut[] }) {
  const ok = results.filter((r) => !r.refusal);
  if (ok.length < 2) return null;
  const fastest = ok.reduce((a, b) => (a.latency_ms <= b.latency_ms ? a : b));
  const cheapest = ok.reduce((a, b) =>
    a.input_tokens + a.output_tokens <= b.input_tokens + b.output_tokens ? a : b
  );
  const secondFastest = ok.filter(r => r.strategy !== fastest.strategy).reduce((a, b) => (a.latency_ms <= b.latency_ms ? a : b), ok[0]);
  const secondCheapest = ok.filter(r => r.strategy !== cheapest.strategy).reduce((a, b) =>
    a.input_tokens + a.output_tokens <= b.input_tokens + b.output_tokens ? a : b, ok[0]);

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
      classic: "text-sky-300",
      graph: "text-violet-300",
      agentic: "text-emerald-300",
    };
    return colors[s] || "text-zinc-300";
  };

  const speedup = Math.round((secondFastest.latency_ms / fastest.latency_ms - 1) * 100);
  const savings = Math.round((1 - (cheapest.input_tokens + cheapest.output_tokens) / (secondCheapest.input_tokens + secondCheapest.output_tokens)) * 100);

  return (
    <div className="card flex flex-col gap-2 border-accent/30 p-3">
      <h3 className="text-sm font-bold text-white">⚡ Winners</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <div className="p-2 rounded bg-sky-500/15 border border-sky-500/40 min-h-[100px]">
          <div className="text-[8px] uppercase tracking-wide text-sky-300 font-bold">Fastest</div>
          <div className={`text-lg font-bold mt-1 ${strategyColor(fastest.strategy)}`}>
            {strategyLabel(fastest.strategy).split(" ")[0]}
          </div>
          <div className="text-sm font-mono text-sky-100 mt-0.5">{formatLatency(fastest.latency_ms)}</div>
          {speedup > 0 && <div className="text-[9px] font-semibold text-sky-200 mt-1">↓ {speedup}% faster</div>}
        </div>
        <div className="p-2 rounded bg-emerald-500/15 border border-emerald-500/40 min-h-[100px]">
          <div className="text-[8px] uppercase tracking-wide text-emerald-300 font-bold">Cheapest</div>
          <div className={`text-lg font-bold mt-1 ${strategyColor(cheapest.strategy)}`}>
            {strategyLabel(cheapest.strategy).split(" ")[0]}
          </div>
          <div className="text-sm font-mono text-emerald-100 mt-0.5">{formatTokens(cheapest.input_tokens + cheapest.output_tokens)}</div>
          {savings > 0 && <div className="text-[9px] font-semibold text-emerald-200 mt-1">↓ {savings}% cheaper</div>}
        </div>
      </div>
    </div>
  );
}
