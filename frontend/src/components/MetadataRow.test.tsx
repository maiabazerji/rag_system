/**
 * The cost shown next to an answer is the backend's per-model estimate
 * (config/model_pricing.toml). The component never prices tokens itself.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n";
import MetadataRow from "./MetadataRow";

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

function renderRow(costUsd: number | null | undefined) {
  act(() => {
    root.render(
      <I18nProvider>
        <MetadataRow input_tokens={1200} output_tokens={300} latency_ms={900} costUsd={costUsd} />
      </I18nProvider>,
    );
  });
  return host.textContent ?? "";
}

describe("MetadataRow cost", () => {
  it("shows the server's estimate when there is one", () => {
    expect(renderRow(0.0084)).toMatch(/\$0[.,]0084|0[.,]0084\s?\$/);
  });

  it("shows no cost when the model is unpriced or the backend sent none", () => {
    for (const cost of [null, undefined]) {
      const text = renderRow(cost);
      expect(text).not.toMatch(/\$|US\s?\$/);
    }
  });
});
