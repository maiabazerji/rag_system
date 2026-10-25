/**
 * Parity: the in-browser advisor must answer exactly like POST /advise.
 * fixtures.json holds real responses from the Python route (heuristic path),
 * written by scripts/export_advisor_fixtures.py.
 */
import { describe, expect, it } from "vitest";
import type { AdviseOverrides, AdviseResponse } from "../api/advisor";
import fixtures from "./generated/fixtures.json";
import { adviseHeuristic } from "./index";
import { pct } from "./messages";
import { pyFixed, pyPercent0, pyRound } from "./pyfmt";

type Fixture = {
  request: { description: string; overrides: AdviseOverrides; language: string };
  response: AdviseResponse;
};

describe("Python rounding", () => {
  it("rounds half to even on the exact binary value", () => {
    expect(pyRound(62.5)).toBe(62);
    expect(pyRound(63.5)).toBe(64);
    expect(pyRound(0.15, 1)).toBe(0.1); // 0.1499999... in binary
    expect(pyRound(0.25, 1)).toBe(0.2);
    expect(pyRound(2.675, 2)).toBe(2.67);
    expect(pyRound(-2.5)).toBe(-2);
    expect(pyFixed(0.5, 0)).toBe("0");
    expect(pyFixed(1.5, 0)).toBe("2");
    expect(pyPercent0(0.625)).toBe("62%");
    expect(pyPercent0(0.715)).toBe("72%"); // 0.715 * 100 == 71.5 as a double, a tie: even
    expect(pct(0.8, "fr")).toBe("80\u00a0%");
  });
});

describe("advisor engine parity with the Python route", () => {
  const cases = fixtures as unknown as Fixture[];

  it("has fixtures to check", () => {
    expect(cases.length).toBeGreaterThanOrEqual(25);
  });

  it.each(cases.map((c, i) => [i, c] as const))("fixture %i", (_i, c) => {
    const got = adviseHeuristic(c.request.description, c.request.overrides, c.request.language);
    expect(got).toEqual(c.response);
  });
});
