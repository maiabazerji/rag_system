import { useQuery } from "@tanstack/react-query";
import { get, post } from "../api/client";

type Run = {
  dataset: string;
  model?: string | null;
  n: number;
  aggregate?: { faithfulness?: number; answer_relevance?: number; context_precision?: number; context_recall?: number };
};

const METRICS: { key: keyof NonNullable<Run["aggregate"]>; label: string }[] = [
  { key: "faithfulness", label: "Faithfulness" },
  { key: "answer_relevance", label: "Relevance" },
  { key: "context_precision", label: "Precision" },
  { key: "context_recall", label: "Recall" },
];

export default function EvalPage() {
  const { data, refetch, isFetching } = useQuery({
    queryKey: ["runs"],
    queryFn: () => get<Run[]>("/eval/runs"),
  });

  async function runNow() {
    await post("/eval/run", { dataset: "golden_v1" });
    refetch();
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="display text-4xl font-semibold text-white">Evaluation</h1>
          <p className="text-zinc-400 mt-2 max-w-xl">
            Score the system against a golden set. Faithfulness, relevance, precision, recall —
            run-over-run, so a prompt tweak can't quietly regress.
          </p>
        </div>
        <button onClick={runNow} disabled={isFetching} className="btn-primary">
          Run eval
        </button>
      </header>

      {!data?.length ? (
        <div className="card text-center text-zinc-400">No runs yet. Click <b className="text-white">Run eval</b> to start.</div>
      ) : (
        <div className="grid gap-3">
          {data.map((r, i) => (
            <div key={i} className="card !p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <div className="font-medium text-white">{r.dataset}</div>
                  <div className="text-xs text-zinc-500 mt-0.5">
                    {r.model ?? "default"} · n={r.n}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {METRICS.map((m) => {
                  const v = r.aggregate?.[m.key];
                  const pct = v == null ? 0 : Math.round(v * 100);
                  return (
                    <div key={m.key} className="bg-bg-elevated border border-bg-border rounded-xl p-3">
                      <div className="text-[11px] text-zinc-500 uppercase tracking-wider">{m.label}</div>
                      <div className="text-xl font-semibold text-white tabular-nums mt-1">
                        {v == null ? "—" : v.toFixed(2)}
                      </div>
                      <div className="h-1 bg-bg-border rounded-full mt-2 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-accent to-emerald-400"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
