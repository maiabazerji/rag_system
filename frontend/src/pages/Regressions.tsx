import { useQuery } from "@tanstack/react-query";
import { get } from "../api/client";

type Reg = { metric: string; delta: number; from: string; to: string };

export default function Regressions() {
  const { data, isLoading } = useQuery({
    queryKey: ["regressions"],
    queryFn: () => get<Reg[]>("/eval/regressions"),
  });

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-4xl font-semibold text-white">Regressions</h1>
        <p className="text-zinc-400 max-w-xl">
          Metric drift between consecutive eval runs. The early-warning system for prompt-and-pray.
        </p>
      </header>

      {isLoading && <div className="card text-zinc-400">Loading…</div>}

      {!isLoading && (!data || data.length === 0) && (
        <div className="card text-center py-10">
          <div className="text-emerald-400 text-3xl">✓</div>
          <h2 className="text-white font-semibold mt-2">No regressions detected</h2>
          <p className="text-zinc-400 text-sm mt-1">All metrics are stable run-over-run.</p>
        </div>
      )}

      {data && data.length > 0 && (
        <div className="grid gap-2">
          {data.map((r, i) => {
            const bad = r.delta < 0;
            return (
              <div key={i} className="card !p-4 flex items-center justify-between">
                <div>
                  <div className="font-medium text-white">{r.metric}</div>
                  <div className="text-xs text-zinc-500 font-mono mt-0.5">
                    {r.from} → {r.to}
                  </div>
                </div>
                <div
                  className={`chip ${bad ? "!text-rose-300 !border-rose-500/40 !bg-rose-500/10" : "!text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10"}`}
                >
                  Δ {r.delta > 0 ? "+" : ""}{r.delta.toFixed(3)}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
