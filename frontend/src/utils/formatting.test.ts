import { describe, expect, it } from "vitest";
import {
  cleanAnswer,
  formatConfidence,
  formatLatency,
  formatScore,
  formatTokens,
} from "./formatting";

describe("locale-aware numbers", () => {
  const plain = (s: string) => s.replace(/\s/g, " ");

  it("keeps the English output", () => {
    expect(formatTokens(1500)).toBe("1.5K");
    expect(formatLatency(1234)).toBe("1.23s");
    expect(formatLatency(250)).toBe("250ms");
    expect(formatConfidence(0.85)).toBe("85%");
  });

  it("uses French separators", () => {
    expect(formatTokens(1500, "fr-FR")).toBe("1,5K");
    expect(plain(formatLatency(1234, "fr-FR"))).toBe("1,23s");
    expect(plain(formatConfidence(0.85, "fr-FR"))).toBe("85 %");
    expect(formatScore(0.8, 2, "fr-FR")).toBe("0,80");
  });
});

describe("cleanAnswer", () => {
  it("keeps headings as text instead of dropping them", () => {
    expect(cleanAnswer("## Summary\nBody text")).toBe("Summary\nBody text");
  });

  it("keeps paragraph breaks but collapses runs of blank lines", () => {
    expect(cleanAnswer("First.\n\n\n\nSecond.")).toBe("First.\n\nSecond.");
  });

  it("strips inline markup and link targets", () => {
    expect(cleanAnswer("**bold** *it* `code` [site](https://x.y)")).toBe("bold it code site");
  });

  it("keeps citations so they can be matched against sources", () => {
    expect(cleanAnswer("According to [9fa3c1b0e2:4], yes.")).toBe(
      "According to [9fa3c1b0e2:4], yes.",
    );
  });

  it("drops horizontal rules and blockquote markers", () => {
    expect(cleanAnswer("> quoted\n---\nafter")).toBe("quoted\nafter");
  });
});
