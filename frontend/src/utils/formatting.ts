/**
 * Display formatting. Every number goes through Intl with the caller's locale
 * (from `useI18n().locale`); the default keeps the English output stable.
 */
const DEFAULT_LOCALE = "en-US";

function num(value: number, locale: string, options?: Intl.NumberFormatOptions): string {
  return new Intl.NumberFormat(locale, options).format(value);
}

const ONE_DECIMAL = { minimumFractionDigits: 1, maximumFractionDigits: 1 };

export function formatTokens(count: number, locale: string = DEFAULT_LOCALE): string {
  if (count >= 1000000) {
    return `${num(count / 1000000, locale, ONE_DECIMAL)}M`;
  }
  if (count >= 1000) {
    return `${num(count / 1000, locale, ONE_DECIMAL)}K`;
  }
  return num(count, locale);
}

export function formatLatency(ms: number, locale: string = DEFAULT_LOCALE): string {
  if (ms >= 1000) {
    return num(ms / 1000, locale, {
      style: "unit",
      unit: "second",
      unitDisplay: "narrow",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  return num(Math.round(ms), locale, {
    style: "unit",
    unit: "millisecond",
    unitDisplay: "narrow",
  });
}

export function formatCost(tokens: number, locale: string = DEFAULT_LOCALE): string {
  // Approximate cost: $0.50 per 1M input tokens, $1.50 per 1M output tokens
  // Rough average: $1 per 1M tokens
  const costUSD = (tokens / 1000000) * 0.001;
  const usd = (v: number) =>
    num(v, locale, {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 4,
      maximumFractionDigits: 4,
    });
  if (costUSD < 0.0001) return `< ${usd(0.0001)}`;
  return usd(costUSD);
}

export function formatConfidence(conf: number, locale: string = DEFAULT_LOCALE): string {
  return num(Math.round(conf * 100) / 100, locale, { style: "percent" });
}

/** A score with fixed decimals: 0.8 -> "0.80" (en) or "0,80" (fr). */
export function formatScore(value: number, digits = 2, locale: string = DEFAULT_LOCALE): string {
  return num(value, locale, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function formatDuration(ms: number): string {
  if (ms < 60000) {
    return `${Math.round(ms / 1000)}s`;
  }
  const mins = Math.floor(ms / 60000);
  const secs = Math.round((ms % 60000) / 1000);
  return `${mins}m ${secs}s`;
}

/**
 * Strip Markdown syntax but keep the structure: headings stay as their own
 * lines, paragraphs keep their blank lines, and citations like [abc:4] stay so
 * they can be matched against the sources list. Rendered with pre-wrap.
 */
export function cleanAnswer(text: string): string {
  return text
    .split("\n")
    .filter((line) => line.trim() !== "---")
    .map((line) =>
      line
        .replace(/^#+\s+/, "")
        .replace(/^>\s?/, "")
        .replace(/\*\*(.+?)\*\*/g, "$1")
        .replace(/\*(.+?)\*/g, "$1")
        .replace(/`(.+?)`/g, "$1")
        .replace(/\[(.+?)\]\(.+?\)/g, "$1"),
    )
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** An estimated USD amount: 4 decimals below $1, 2 above; "< $0.0001" for tiny non-zero values. */
export function formatUsd(value: number, locale: string = DEFAULT_LOCALE): string {
  const usd = (v: number, digits: number) =>
    num(v, locale, {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  if (value > 0 && value < 0.0001) return `< ${usd(0.0001, 4)}`;
  return usd(value, value < 1 ? 4 : 2);
}
