/**
 * Framework-free half of the i18n layer: language detection and persistence,
 * and `translate`. The React provider and hooks live in `./index.tsx`; code
 * outside React (the API client) reads the active language via `getLanguage()`.
 */
import { en, type Dictionary, type MessageKey } from "./en";
import { fr } from "./fr";

export type Language = "en" | "fr";
export type { MessageKey };

export const LANGUAGES: readonly Language[] = ["en", "fr"];
export const LANGUAGE_STORAGE_KEY = "evalrag.lang";

/** BCP 47 locale used for Intl number, date and plural formatting. */
export const LOCALES: Record<Language, string> = { en: "en-US", fr: "fr-FR" };

export const DICTIONARIES: Record<Language, Dictionary> = { en, fr };

/** Values substituted into `{name}` placeholders. */
export type Vars = Record<string, string | number>;

/** Keys `x` for which both `x_one` and `x_other` exist. */
export type PluralKey = {
  [K in MessageKey]: K extends `${infer Base}_one`
    ? `${Base}_other` extends MessageKey
      ? Base
      : never
    : never;
}[MessageKey];

export function isLanguage(value: unknown): value is Language {
  return typeof value === "string" && (LANGUAGES as readonly string[]).includes(value);
}

/** The saved language, or null when none is saved or storage is unavailable. */
export function readStoredLanguage(): Language | null {
  try {
    const v = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    return isLanguage(v) ? v : null;
  } catch {
    return null; // private browsing, or storage blocked
  }
}

/** Persist the language choice; silently a no-op when storage is unavailable. */
export function storeLanguage(lang: Language): void {
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, lang);
  } catch {
    /* storage unavailable; the choice simply will not persist */
  }
}

/** The first supported language in the browser's preferences, else English. */
export function browserLanguage(): Language {
  const prefs =
    typeof navigator === "undefined"
      ? []
      : navigator.languages?.length
        ? navigator.languages
        : [navigator.language];
  for (const tag of prefs) {
    const base = (tag ?? "").toLowerCase().split("-")[0];
    if (isLanguage(base)) return base;
  }
  return "en";
}

/** Saved choice first, then the browser's preference. */
export function initialLanguage(): Language {
  return readStoredLanguage() ?? browserLanguage();
}

let current: Language = initialLanguage();

/** The active UI language, for code outside React. */
export function getLanguage(): Language {
  return current;
}

/** Make `lang` the active language for non-React callers. Does not persist it. */
export function setCurrentLanguage(lang: Language): void {
  current = lang;
}

/** Value for the `Accept-Language` request header. */
export function acceptLanguage(lang: Language = current): string {
  return lang === "en" ? "en" : `${lang},en;q=0.5`;
}

/**
 * Look a message up and fill its placeholders. Unknown placeholders are left as
 * written, so a literal `{id}` in a message survives.
 */
export function translate(lang: Language, key: MessageKey, vars?: Vars): string {
  const template = DICTIONARIES[lang][key] ?? en[key] ?? key;
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (match, name: string) =>
    Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match,
  );
}

/** Pick `base_one` or `base_other` by the language's plural rules; `{count}` is filled. */
export function translatePlural(lang: Language, base: PluralKey, count: number, vars?: Vars): string {
  const category = new Intl.PluralRules(LOCALES[lang]).select(count) === "one" ? "one" : "other";
  return translate(lang, `${base}_${category}` as MessageKey, { count, ...vars });
}
