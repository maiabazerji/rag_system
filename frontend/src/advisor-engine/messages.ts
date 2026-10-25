/**
 * The advisor's message catalog, exported verbatim from backend/app/advisor/messages.py
 * by scripts/export_advisor_fixtures.py. Same keys, same wording, same helpers.
 */
import catalog from "./generated/messages.json";
import { pyFixed, pyPercent0 } from "./pyfmt";

export type Language = "en" | "fr";

const MESSAGES = catalog as Record<Language, Record<string, string>>;

/** Map any language tag to a supported one ("fr-FR" -> "fr"), defaulting to English. */
export function normalize(lang: string | null | undefined): Language {
  return (lang ?? "en").trim().toLowerCase().slice(0, 2) === "fr" ? "fr" : "en";
}

/** Format message `key` in `lang` (Python `str.format` placeholders), falling back to English. */
export function msg(key: string, lang: string | null | undefined, kw: Record<string, string | number> = {}): string {
  const template = MESSAGES[normalize(lang)][key] || MESSAGES.en[key];
  if (template === undefined) throw new Error(`Unknown advisor message: ${key}`);
  return template.replace(/\{\{|\}\}|\{(\w+)\}/g, (m, name: string | undefined) => {
    if (m === "{{") return "{";
    if (m === "}}") return "}";
    if (!(name! in kw)) throw new Error(`Missing placeholder ${name} for ${key}`);
    return String(kw[name!]);
  });
}

/** A whole percentage: "80%" in English, "80 %" (no-break space) in French. */
export function pct(fraction: number, lang: string | null | undefined): string {
  // A no-break space before "%", as in the Python catalog.
  return normalize(lang) === "fr" ? `${pyFixed(fraction * 100, 0)}\u00a0%` : pyPercent0(fraction);
}

/** A number with a decimal comma in French. */
export function decimal(value: number, digits: number, lang: string | null | undefined): string {
  const text = pyFixed(value, digits);
  return normalize(lang) === "fr" ? text.replace(".", ",") : text;
}
