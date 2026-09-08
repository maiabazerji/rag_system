import Tooltip from "./Tooltip";
import { formatTokens, formatLatency, formatCost } from "../utils/formatting";

type Props = {
  strategy?: string;
  model?: string;
  input_tokens?: number;
  output_tokens?: number;
  latency_ms?: number;
  iterations?: number;
  compact?: boolean;
};

export default function MetadataRow({
  strategy,
  model,
  input_tokens,
  output_tokens,
  latency_ms,
  iterations,
  compact = false,
}: Props) {
  const total_tokens = input_tokens && output_tokens ? input_tokens + output_tokens : 0;
  const cost = total_tokens > 0 ? formatCost(total_tokens) : null;

  if (compact) {
    return (
      <div className="flex items-center gap-3 text-xs text-zinc-400 flex-wrap">
        {strategy && <span className="chip !px-2 !py-0.5">{strategy}</span>}
        {model && <span className="chip !px-2 !py-0.5">{model}</span>}
        <div className="flex items-center gap-2">
          <Tooltip label="Time from request to response">
            <span className="font-mono">⏱ {latency_ms ? formatLatency(latency_ms) : "n/a"}</span>
          </Tooltip>
          <Tooltip label={cost ? `Estimated cost: ${cost}` : "Input + output tokens"}>
            <span className="font-mono">💰 {total_tokens ? formatTokens(total_tokens) : "n/a"}</span>
          </Tooltip>
          {iterations && iterations > 1 && (
            <Tooltip label="Number of reasoning iterations">
              <span className="font-mono">🔄 {iterations}x</span>
            </Tooltip>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
      {strategy && (
        <div className="bg-bg-elevated rounded p-2 border border-bg-border">
          <div className="text-[10px] text-zinc-500 uppercase font-semibold">Strategy</div>
          <div className="text-sm font-medium text-white mt-1">{strategy}</div>
        </div>
      )}
      {model && (
        <div className="bg-bg-elevated rounded p-2 border border-bg-border">
          <div className="text-[10px] text-zinc-500 uppercase font-semibold">Model</div>
          <div className="text-sm font-medium text-white mt-1 truncate">{model}</div>
        </div>
      )}
      {latency_ms != null && (
        <Tooltip label="Time from request to response" side="top">
          <div className="bg-sky-500/20 border border-sky-500/40 rounded p-2 cursor-help">
            <div className="text-[10px] text-sky-300 uppercase font-semibold">Latency</div>
            <div className="text-sm font-mono font-bold text-sky-100 mt-1">{formatLatency(latency_ms)}</div>
          </div>
        </Tooltip>
      )}
      {total_tokens > 0 && (
        <Tooltip label={cost ? `Estimated cost: ${cost}` : "Input + output tokens used"} side="top">
          <div className="bg-emerald-500/20 border border-emerald-500/40 rounded p-2 cursor-help">
            <div className="text-[10px] text-emerald-300 uppercase font-semibold">Tokens</div>
            <div className="text-sm font-mono font-bold text-emerald-100 mt-1">{formatTokens(total_tokens)}</div>
            {cost && <div className="text-[10px] text-emerald-300/70 font-mono mt-0.5">≈ {cost}</div>}
          </div>
        </Tooltip>
      )}
      {iterations && iterations > 1 && (
        <Tooltip label="Number of reasoning iterations (loops)" side="top">
          <div className="bg-violet-500/20 border border-violet-500/40 rounded p-2 cursor-help">
            <div className="text-[10px] text-violet-300 uppercase font-semibold">Iterations</div>
            <div className="text-sm font-mono font-bold text-violet-100 mt-1">{iterations}x</div>
          </div>
        </Tooltip>
      )}
    </div>
  );
}
