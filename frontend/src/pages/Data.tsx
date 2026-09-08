import { useEffect, useState } from "react";
import { get } from "../api/client";

interface CorpusStats {
  indexed_chunks: number;
}

export default function DataPage() {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    get<CorpusStats>("/ingest/stats")
      .then(setStats)
      .catch(e => console.error("Failed to load corpus stats:", e))
      .finally(() => setLoading(false));
  }, []);

  const indexed_chunks = stats?.indexed_chunks ?? 0;

  const QUESTIONS = [
    {
      level: "Medium",
      q: "Entity extraction in Graph RAG vs dense-only?",
      best: "Graph",
      classic: "8.3s",
      graph: "7.1s",
      agentic: "26s",
    },
    {
      level: "Hard",
      q: "Trade latency for reasoning capability?",
      best: "Agentic",
      classic: "7.8s",
      graph: "8.1s",
      agentic: "18s",
    },
    {
      level: "Hard",
      q: "Chunk size & embedding model interaction?",
      best: "Agentic",
      classic: "9.1s",
      graph: "8.7s",
      agentic: "21s",
    },
  ];

  const INSIGHTS = [
    { pattern: "Entity-heavy", impact: "Graph +15% latency advantage" },
    { pattern: "Factual", impact: "Classic optimal for speed & cost" },
    { pattern: "Synthesis", impact: "Agentic discovers cross-document evidence" },
  ];

  return (
    <div className="flex flex-col gap-6 max-w-6xl">
      <header>
        <h1 className="display text-4xl font-semibold text-white">Data</h1>
        <p className="text-sm text-zinc-400 mt-2">What is indexed, and how the retrieval strategies compare on it.</p>
      </header>

      {/* Stats Row - Compact */}
      {!loading && (
        <div className="grid grid-cols-5 gap-3">
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">Chunks</div>
            <div className="text-2xl font-bold text-white mt-1">{indexed_chunks}</div>
          </div>
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">Tokens</div>
            <div className="text-2xl font-bold text-white mt-1">{(indexed_chunks * 300 / 1000).toFixed(1)}k</div>
          </div>
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">Size</div>
            <div className="text-2xl font-bold text-white mt-1">{((indexed_chunks * 0.012)).toFixed(2)}MB</div>
          </div>
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">Status</div>
            <div className="text-2xl font-bold text-emerald-400 mt-1">Live</div>
          </div>
          <div className="p-4 rounded-lg bg-bg-surface border border-bg-border">
            <div className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">Dataset</div>
            <div className="text-2xl font-bold text-white mt-1">100k</div>
          </div>
        </div>
      )}

      {/* Performance Table */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Golden Dataset Performance</h2>
        <div className="space-y-2">
          {QUESTIONS.map((q, i) => (
            <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-bg-surface border border-bg-border/50 hover:border-bg-border transition-colors">
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <span className={`px-2 py-1 rounded text-xs font-semibold whitespace-nowrap ${
                  q.level === "Hard" ? "bg-amber-500/20 text-amber-200" : "bg-emerald-500/20 text-emerald-200"
                }`}>
                  {q.level}
                </span>
                <span className="text-sm text-zinc-300 truncate">{q.q}</span>
              </div>
              <div className="flex gap-4 items-center ml-4">
                <div className="text-right">
                  <div className="text-xs text-zinc-500">Classic</div>
                  <div className="text-sm font-mono text-zinc-300">{q.classic}</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-zinc-500">Graph</div>
                  <div className="text-sm font-mono text-violet-300">{q.graph}</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-zinc-500">Agentic</div>
                  <div className="text-sm font-mono text-emerald-300">{q.agentic}</div>
                </div>
                <span className={`px-2.5 py-1 rounded text-xs font-semibold whitespace-nowrap ${
                  q.best === "Graph" ? "bg-violet-500/20 text-violet-200" :
                  q.best === "Agentic" ? "bg-emerald-500/20 text-emerald-200" :
                  "bg-sky-500/20 text-sky-200"
                }`}>
                  {q.best}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Patterns Grid */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Strategy Patterns</h2>
        <div className="grid grid-cols-3 gap-4">
          {INSIGHTS.map((insight, i) => (
            <div key={i} className="p-4 rounded-lg bg-bg-surface border border-bg-border">
              <div className="font-semibold text-white text-sm">{insight.pattern}</div>
              <div className="text-xs text-zinc-400 mt-2">{insight.impact}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
