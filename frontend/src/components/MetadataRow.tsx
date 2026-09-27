import Tooltip from "./Tooltip";
import { formatTokens, formatLatency, formatCost } from "../utils/formatting";
import { useI18n } from "../i18n";

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
  const { t, locale } = useI18n();
  const total_tokens = input_tokens && output_tokens ? input_tokens + output_tokens : 0;
  const cost = total_tokens > 0 ? formatCost(total_tokens, locale) : null;

  if (compact) {
    return (
      <div className="flex items-center gap-3 text-xs text-zinc-400 flex-wrap">
        {strategy && <span className="chip !px-2 !py-0.5">{strategy}</span>}
        {model && <span className="chip !px-2 !py-0.5">{model}</span>}
        <div className="flex items-center gap-2">
          <Tooltip label={t("meta.latencyTip")}>
            <span className="font-mono">
              ⏱ {latency_ms ? formatLatency(latency_ms, locale) : t("common.na")}
            </span>
          </Tooltip>
          <Tooltip label={cost ? t("meta.costTip", { cost }) : t("meta.tokensTip")}>
            <span className="font-mono">
              💰 {total_tokens ? formatTokens(total_tokens, locale) : t("common.na")}
            </span>
          </Tooltip>
          {iterations && iterations > 1 && (
            <Tooltip label={t("meta.iterationsTip")}>
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
          <div className="text-[10px] text-zinc-500 uppercase font-semibold">{t("meta.strategy")}</div>
          <div className="text-sm font-medium text-white mt-1">{strategy}</div>
        </div>
      )}
      {model && (
        <div className="bg-bg-elevated rounded p-2 border border-bg-border">
          <div className="text-[10px] text-zinc-500 uppercase font-semibold">{t("meta.model")}</div>
          <div className="text-sm font-medium text-white mt-1 truncate">{model}</div>
        </div>
      )}
      {latency_ms != null && (
        <Tooltip label={t("meta.latencyTip")} side="top">
          <div className="bg-sky-500/20 border border-sky-500/40 rounded p-2 cursor-help">
            <div className="text-[10px] text-sky-300 uppercase font-semibold">{t("meta.latency")}</div>
            <div className="text-sm font-mono font-bold text-sky-100 mt-1">
              {formatLatency(latency_ms, locale)}
            </div>
          </div>
        </Tooltip>
      )}
      {total_tokens > 0 && (
        <Tooltip label={cost ? t("meta.costTip", { cost }) : t("meta.tokensUsedTip")} side="top">
          <div className="bg-emerald-500/20 border border-emerald-500/40 rounded p-2 cursor-help">
            <div className="text-[10px] text-emerald-300 uppercase font-semibold">{t("meta.tokens")}</div>
            <div className="text-sm font-mono font-bold text-emerald-100 mt-1">
              {formatTokens(total_tokens, locale)}
            </div>
            {cost && <div className="text-[10px] text-emerald-300/70 font-mono mt-0.5">≈ {cost}</div>}
          </div>
        </Tooltip>
      )}
      {iterations && iterations > 1 && (
        <Tooltip label={t("meta.iterationsLoopsTip")} side="top">
          <div className="bg-violet-500/20 border border-violet-500/40 rounded p-2 cursor-help">
            <div className="text-[10px] text-violet-300 uppercase font-semibold">{t("meta.iterations")}</div>
            <div className="text-sm font-mono font-bold text-violet-100 mt-1">{iterations}x</div>
          </div>
        </Tooltip>
      )}
    </div>
  );
}
