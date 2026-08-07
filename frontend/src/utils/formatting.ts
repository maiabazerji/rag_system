export function formatTokens(count: number): string {
  if (count >= 1000000) {
    return `${(count / 1000000).toFixed(1)}M`;
  }
  if (count >= 1000) {
    return `${(count / 1000).toFixed(1)}K`;
  }
  return count.toLocaleString();
}

export function formatLatency(ms: number): string {
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${Math.round(ms)}ms`;
}

export function formatCost(tokens: number): string {
  // Approximate cost: $0.50 per 1M input tokens, $1.50 per 1M output tokens
  // Rough average: $1 per 1M tokens
  const costUSD = (tokens / 1000000) * 0.001;
  if (costUSD < 0.0001) return "< $0.0001";
  return `$${costUSD.toFixed(4)}`;
}

export function formatConfidence(conf: number): string {
  const pct = Math.round(conf * 100);
  return `${pct}%`;
}

export function formatDuration(ms: number): string {
  if (ms < 60000) {
    return `${Math.round(ms / 1000)}s`;
  }
  const mins = Math.floor(ms / 60000);
  const secs = Math.round((ms % 60000) / 1000);
  return `${mins}m ${secs}s`;
}
