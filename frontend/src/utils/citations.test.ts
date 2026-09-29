import { describe, expect, it } from "vitest";
import { handleNumber, sourceAnchor, splitCitations } from "./citations";

describe("splitCitations", () => {
  it("splits text and markers in order", () => {
    expect(splitCitations("BM25 is lexical [S1]. Dense is not [S2][S3].")).toEqual([
      { kind: "text", text: "BM25 is lexical " },
      { kind: "cite", handle: "S1" },
      { kind: "text", text: ". Dense is not " },
      { kind: "cite", handle: "S2" },
      { kind: "cite", handle: "S3" },
      { kind: "text", text: "." },
    ]);
  });

  it("leaves text without markers alone", () => {
    expect(splitCitations("No citations here [x].")).toEqual([
      { kind: "text", text: "No citations here [x]." },
    ]);
    expect(splitCitations("")).toEqual([]);
  });

  it("does not treat old chunk-id brackets as handles", () => {
    expect(splitCitations("[9fa3c1:4]")).toEqual([{ kind: "text", text: "[9fa3c1:4]" }]);
  });
});

describe("helpers", () => {
  it("numbers and anchors handles", () => {
    expect(handleNumber("S12")).toBe("12");
    expect(sourceAnchor("r1", "S2")).toBe("r1-src-S2");
  });
});
