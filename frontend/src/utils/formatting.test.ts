import { describe, expect, it } from "vitest";
import { cleanAnswer } from "./formatting";

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
