import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n";
import { CitedText, GroundingBadge, InvalidCitationsWarning, type CitedSource } from "./Citations";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function render(node: ReactNode, lang: "en" | "fr" = "en") {
  act(() => {
    root.render(<I18nProvider initialLanguage={lang}>{node}</I18nProvider>);
  });
}

const sources: CitedSource[] = [
  { chunk_id: "a:1", quote: "q", handle: "S1" },
  { chunk_id: "b:2", quote: "q", handle: "S3" },
];

describe("CitedText", () => {
  it("renders known markers as superscript links to their source", () => {
    render(<CitedText text="One [S1]. Two [S3]. Stale [S9]." anchorPrefix="p" sources={sources} />);
    const links = [...container.querySelectorAll("sup a")];
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["#p-src-S1", "#p-src-S3"]);
    expect(links.map((a) => a.textContent)).toEqual(["[1]", "[3]"]);
    expect(container.textContent).toContain("Stale [S9].");
  });
});

describe("GroundingBadge", () => {
  it.each([
    [{ status: "answered", grounded: true }, "Grounded"],
    [{ status: "answered", grounded: false }, "Weakly grounded"],
    [{ status: "partial", grounded: false }, "Partial"],
    [{ status: "insufficient_context", grounded: false }, "Insufficient context"],
  ] as const)("labels %o as %s", (g, label) => {
    render(<GroundingBadge g={g} />);
    expect(container.textContent).toBe(label);
  });

  it("renders nothing without a status", () => {
    render(<GroundingBadge g={{}} />);
    expect(container.textContent).toBe("");
  });

  it("is translated", () => {
    render(<GroundingBadge g={{ status: "insufficient_context" }} />, "fr");
    expect(container.textContent).toBe("Contexte insuffisant");
  });
});

describe("InvalidCitationsWarning", () => {
  it("lists removed citations with the right plural", () => {
    render(<InvalidCitationsWarning invalid={["S9", "S7"]} />);
    expect(container.textContent).toContain("2 citations pointed outside");
    expect(container.textContent).toContain("S9, S7");
  });

  it("renders nothing when every citation was valid", () => {
    render(<InvalidCitationsWarning invalid={[]} />);
    expect(container.textContent).toBe("");
  });
});
