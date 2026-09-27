import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "./en";
import { fr } from "./fr";
import {
  I18nProvider,
  LANGUAGE_STORAGE_KEY,
  browserLanguage,
  getLanguage,
  initialLanguage,
  setCurrentLanguage,
  translate,
  translatePlural,
  useI18n,
  useT,
} from "./index";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

describe("dictionaries", () => {
  it("fr and en have identical keys", () => {
    expect(Object.keys(fr).sort()).toEqual(Object.keys(en).sort());
  });

  it("no message is empty", () => {
    for (const [lang, dict] of Object.entries({ en, fr })) {
      for (const [key, value] of Object.entries(dict)) {
        expect(value.trim(), `${lang}:${key}`).not.toBe("");
      }
    }
  });

  it("each translation uses the same placeholders as the English", () => {
    for (const key of Object.keys(en) as (keyof typeof en)[]) {
      expect(placeholders(fr[key]), key).toEqual(placeholders(en[key]));
    }
  });
});

describe("translate", () => {
  it("fills placeholders and leaves unknown ones alone", () => {
    expect(translate("en", "common.sources", { count: 3 })).toBe("Sources (3)");
    expect(translate("fr", "ask.traceTitle")).toContain("GET /traces/{id}");
  });

  it("uses each language's plural rules", () => {
    expect(translatePlural("en", "ask.stars", 1)).toBe("1 star");
    expect(translatePlural("en", "ask.stars", 0)).toBe("0 stars");
    // French treats 0 as singular.
    expect(translatePlural("fr", "ask.stars", 0)).toBe("0 étoile");
    expect(translatePlural("fr", "ask.stars", 2)).toBe("2 étoiles");
  });
});

describe("initial language", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => vi.restoreAllMocks());

  it("prefers the saved choice", () => {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, "fr");
    expect(initialLanguage()).toBe("fr");
  });

  it("ignores an unsupported saved value and falls back to the browser", () => {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, "xx");
    vi.spyOn(navigator, "languages", "get").mockReturnValue(["fr-CA", "en"]);
    expect(initialLanguage()).toBe("fr");
  });

  it("defaults to English for an unsupported browser language", () => {
    vi.spyOn(navigator, "languages", "get").mockReturnValue(["ja-JP"]);
    expect(browserLanguage()).toBe("en");
  });

  it("survives storage that throws", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(navigator, "languages", "get").mockReturnValue(["en-GB"]);
    expect(initialLanguage()).toBe("en");
  });
});

function Probe() {
  const t = useT();
  const { lang, setLanguage, formatNumber } = useI18n();
  return (
    <div>
      <span data-testid="title">{t("nav.eval")}</span>
      <span data-testid="num">{formatNumber(1234.5)}</span>
      <button onClick={() => setLanguage(lang === "en" ? "fr" : "en")}>toggle</button>
    </div>
  );
}

describe("useT", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    setCurrentLanguage("en");
  });

  const text = (id: string) =>
    container.querySelector(`[data-testid="${id}"]`)?.textContent?.replace(/\s/g, " ");

  it("renders the provider's language and switches live", () => {
    act(() => {
      root.render(
        <I18nProvider initialLanguage="en">
          <Probe />
        </I18nProvider>,
      );
    });
    expect(text("title")).toBe("Evaluation");
    expect(text("num")).toBe("1,234.5");

    act(() => {
      container.querySelector("button")!.click();
    });
    expect(text("title")).toBe("Évaluation");
    expect(text("num")).toBe("1 234,5");
    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe("fr");
    expect(getLanguage()).toBe("fr");
    expect(document.documentElement.lang).toBe("fr");
  });

  it("works without a provider, using the module language", () => {
    setCurrentLanguage("fr");
    act(() => {
      root.render(<Probe />);
    });
    expect(text("title")).toBe("Évaluation");
  });
});
