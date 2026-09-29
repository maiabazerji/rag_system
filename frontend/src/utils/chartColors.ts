/**
 * Chart colours for the dark UI surface (#16171b).
 *
 * Categorical slots from the reference palette's dark column, in fixed order.
 * Validated with the dataviz palette validator against #16171b:
 * - strategies (3 slots, all pairs, for the scatter): worst CVD ΔE 9.4,
 *   worst normal-vision ΔE 20.9, every slot >= 3:1 contrast;
 * - latency stages (5 slots, adjacent pairs, stacked bar): worst CVD ΔE 8.4,
 *   worst normal-vision ΔE 19.3.
 * Colour follows the entity, never its rank: a strategy keeps its hue whatever
 * is filtered. Text never wears these colours; a swatch beside it does.
 */
export const STRATEGY_COLORS: Record<string, string> = {
  classic: "#3987e5",
  graph: "#d95926",
  agentic: "#199e70",
};

/** For a strategy name the palette does not know. Neutral, never a generated hue. */
export const UNKNOWN_SERIES_COLOR = "#898781";

export const STAGE_COLORS = {
  retrieval: "#3987e5",
  rerank: "#d95926",
  generation: "#199e70",
  citation_validation: "#c98500",
  other: "#d55181",
} as const;

export const CHART_INK = {
  grid: "#2a2c33",
  axis: "#3a3c44",
  muted: "#898781",
  secondary: "#c3c2b7",
  surface: "#16171b",
} as const;

export function strategyColor(strategy: string): string {
  return STRATEGY_COLORS[strategy] ?? UNKNOWN_SERIES_COLOR;
}
