/**
 * A deliberately small i18n layer: typed dictionaries (`en.ts`, `fr.ts`), a
 * context holding the active language, and `useT()` / `useI18n()` hooks.
 *
 * The language comes from localStorage when the user picked one, otherwise from
 * `navigator.language`. Numbers and dates are formatted with Intl for the
 * active locale.
 */
import {
  createContext,
  Fragment,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  LOCALES,
  getLanguage,
  setCurrentLanguage,
  storeLanguage,
  translate,
  translatePlural,
  type Language,
  type MessageKey,
  type PluralKey,
  type Vars,
} from "./core";

export * from "./core";

export type I18n = {
  lang: Language;
  /** BCP 47 locale for Intl APIs, e.g. "fr-FR". */
  locale: string;
  setLanguage: (lang: Language) => void;
  /** Translate `key`, filling `{name}` placeholders from `vars`. */
  t: (key: MessageKey, vars?: Vars) => string;
  /** Plural-aware translate: picks `base_one` / `base_other` for `count`. */
  tp: (base: PluralKey, count: number, vars?: Vars) => string;
  /** Translate with React nodes (links, `<code>`) in the placeholders. */
  rich: (key: MessageKey, nodes: Record<string, ReactNode>) => ReactNode;
  /** Locale-aware `Intl.NumberFormat`. */
  formatNumber: (value: number, options?: Intl.NumberFormatOptions) => string;
  /** Locale-aware `Intl.DateTimeFormat`; "" for an unparseable date. */
  formatDate: (value: string | number | Date, options?: Intl.DateTimeFormatOptions) => string;
};

/** Split a template on `{name}` placeholders and drop React nodes into them. */
export function interpolateNodes(template: string, nodes: Record<string, ReactNode>): ReactNode {
  return template
    .split(/(\{\w+\})/)
    .filter((part) => part !== "")
    .map((part, i) => {
      const name = /^\{(\w+)\}$/.exec(part)?.[1];
      return (
        <Fragment key={i}>
          {name !== undefined && Object.prototype.hasOwnProperty.call(nodes, name)
            ? nodes[name]
            : part}
        </Fragment>
      );
    });
}

export function makeI18n(lang: Language, setLanguage: (lang: Language) => void): I18n {
  const locale = LOCALES[lang];
  return {
    lang,
    locale,
    setLanguage,
    t: (key, vars) => translate(lang, key, vars),
    tp: (base, count, vars) => translatePlural(lang, base, count, vars),
    rich: (key, nodes) => interpolateNodes(translate(lang, key), nodes),
    formatNumber: (value, options) => new Intl.NumberFormat(locale, options).format(value),
    formatDate: (value, options) => {
      const d = value instanceof Date ? value : new Date(value);
      return Number.isNaN(d.getTime()) ? "" : new Intl.DateTimeFormat(locale, options).format(d);
    },
  };
}

const I18nContext = createContext<I18n | null>(null);

export function I18nProvider({
  children,
  initialLanguage,
}: {
  children: ReactNode;
  /** Overrides the saved / browser language; mainly for tests. */
  initialLanguage?: Language;
}) {
  const [lang, setLang] = useState<Language>(() => {
    const start = initialLanguage ?? getLanguage();
    setCurrentLanguage(start);
    return start;
  });

  const setLanguage = useCallback((next: Language) => {
    setCurrentLanguage(next);
    storeLanguage(next);
    setLang(next);
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  const value = useMemo(() => makeI18n(lang, setLanguage), [lang, setLanguage]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

/** The i18n context. Outside a provider it reflects the module-level language. */
export function useI18n(): I18n {
  return useContext(I18nContext) ?? makeI18n(getLanguage(), setCurrentLanguage);
}

/** The translate function alone: `const t = useT(); t("nav.ask")`. */
export function useT(): I18n["t"] {
  return useI18n().t;
}
