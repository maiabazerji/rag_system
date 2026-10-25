/**
 * Port of backend/app/advisor/followup.py: ask only about the facts whose
 * answer could change the recommendation, and say how settled the pick is.
 */
import type {
  AdviseOverrides,
  Confidence,
  FollowUpId,
  FollowUpQuestion,
  Level,
  ProjectProfile,
  StrategyRecommendation,
} from "../api/advisor";
import { applyOverrides, parseOverrides } from "./profile";
import { scoreStrategies } from "./scoring";

const MAX_QUESTIONS = 5;
const CLEAR_MARGIN = 15;
const THIN_MARGIN = 5;

type Candidate = {
  field: FollowUpId;
  options: [string, Record<string, unknown>][];
  /** Asked even when it cannot change the ranking: it changes where data may go. */
  alwaysRelevant?: boolean;
};

const CANDIDATES: Candidate[] = [
  {
    field: "question_mix",
    options: [
      ["facts", { question_mix: { single_fact: 80, relational_multi_hop: 10, exploratory_multi_step: 10 } }],
      ["connect", { question_mix: { single_fact: 20, relational_multi_hop: 70, exploratory_multi_step: 10 } }],
      ["research", { question_mix: { single_fact: 20, relational_multi_hop: 20, exploratory_multi_step: 60 } }],
      ["mixed", { question_mix: { single_fact: 40, relational_multi_hop: 30, exploratory_multi_step: 30 } }],
    ],
  },
  {
    field: "corpus_size_docs",
    options: [
      ["lt100", { corpus_size_docs: 50 }],
      ["lt1k", { corpus_size_docs: 500 }],
      ["lt10k", { corpus_size_docs: 5_000 }],
      ["lt100k", { corpus_size_docs: 50_000 }],
      ["more", { corpus_size_docs: 200_000 }],
    ],
  },
  {
    field: "entity_richness",
    options: [
      ["high", { entity_richness: "high" }],
      ["medium", { entity_richness: "medium" }],
      ["low", { entity_richness: "low" }],
    ],
  },
  {
    field: "latency_budget_ms",
    options: [
      ["instant", { latency_budget_ms: 2_000 }],
      ["fast", { latency_budget_ms: 5_000 }],
      ["patient", { latency_budget_ms: 15_000 }],
      ["no_rush", { latency_budget_ms: 60_000 }],
    ],
  },
  {
    field: "cost_sensitivity",
    options: [
      ["low", { cost_sensitivity: "low" }],
      ["medium", { cost_sensitivity: "medium" }],
      ["high", { cost_sensitivity: "high" }],
    ],
  },
  {
    field: "data_freshness",
    options: [
      ["static", { data_freshness: "static" }],
      ["monthly", { data_freshness: "monthly" }],
      ["weekly", { data_freshness: "weekly" }],
      ["daily", { data_freshness: "daily" }],
      ["realtime", { data_freshness: "realtime" }],
    ],
  },
  {
    field: "compliance",
    options: [
      ["anywhere", { compliance: [] }],
      ["eu", { compliance: ["EU only", "GDPR"] }],
      ["on_prem", { compliance: ["on-prem"] }],
    ],
    alwaysRelevant: true,
  },
];

function currentOption(c: Candidate, p: ProjectProfile): string | null {
  if (c.field === "corpus_size_docs") {
    const n = p.corpus_size_docs;
    if (n == null) return null;
    return n < 100 ? "lt100" : n < 1_000 ? "lt1k" : n < 10_000 ? "lt10k" : n < 100_000 ? "lt100k" : "more";
  }
  if (c.field === "latency_budget_ms") {
    const ms = p.latency_budget_ms;
    if (ms == null) return null;
    return ms <= 2_000 ? "instant" : ms <= 5_000 ? "fast" : ms < 30_000 ? "patient" : "no_rush";
  }
  if (c.field === "compliance") {
    if (p.compliance.length === 0 && !p.data_residency) return null;
    const joined = [...p.compliance, p.data_residency ?? ""].join(" ").toLowerCase();
    if (["prem", "self-host", "air-gap", "sur site"].some((h) => joined.includes(h))) return "on_prem";
    return joined.trim() ? "eu" : "anywhere";
  }
  if (c.field === "question_mix") {
    const m = p.question_mix;
    const shares: [string, number][] = [
      ["facts", m.single_fact],
      ["connect", m.relational_multi_hop],
      ["research", m.exploratory_multi_step],
    ];
    // Python's max() keeps the first of equal values.
    const top = shares.reduce((best, s) => (s[1] > best[1] ? s : best));
    return top[1] >= 50 ? top[0] : "mixed";
  }
  const value = p[c.field as keyof ProjectProfile];
  return value != null ? String(value) : null;
}

/** Up to five questions whose answer could change the advice, most decisive first. */
export function followUpQuestions(p: ProjectProfile, ranked: StrategyRecommendation[]): FollowUpQuestion[] {
  const confirmed = new Set(p.overridden_fields);
  const currentTop = ranked[0].strategy;
  const scored: [number, number, FollowUpQuestion][] = [];

  for (const c of CANDIDATES) {
    if (confirmed.has(c.field)) continue;
    const tops = new Set<string>();
    let spread = 0;
    const options: FollowUpQuestion["options"] = [];
    for (const [id, override] of c.options) {
      const rescored = scoreStrategies(applyOverrides(p, parseOverrides(override)));
      tops.add(rescored[0].strategy);
      const byName = Object.fromEntries(rescored.map((r) => [r.strategy, r.score]));
      spread = Math.max(spread, ...ranked.map((r) => Math.abs(byName[r.strategy] - r.score)));
      options.push({ id, overrides: override as AdviseOverrides, top_strategy: rescored[0].strategy });
    }
    const changesTop = tops.size > 1 || (tops.size > 0 && !tops.has(currentTop));
    if (!changesTop && spread < 5 && !c.alwaysRelevant) continue;
    const priority = changesTop ? 0 : !c.alwaysRelevant ? 1 : 2;
    scored.push([
      priority,
      -spread,
      { id: c.field, options, current: currentOption(c, p), changes_recommendation: changesTop },
    ]);
  }

  // Stable, like Python's sort.
  scored.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  // When no answer can change the top pick and its lead is already clear, a
  // question that only nudges scores would not change the advice: skip it.
  const settled =
    !scored.some(([pr]) => pr === 0) && ranked.length > 1 && ranked[0].score - ranked[1].score >= CLEAR_MARGIN;
  const ranking = scored.filter(([pr]) => pr === 0 || (pr === 1 && !settled)).map(([, , q]) => q);
  const deployment = scored.filter(([pr]) => pr === 2).map(([, , q]) => q);
  return [...ranking.slice(0, MAX_QUESTIONS - deployment.length), ...deployment];
}

/** How settled the top pick is, from the score margin and open questions. */
export function confidence(ranked: StrategyRecommendation[], questions: FollowUpQuestion[]): Confidence {
  const margin = ranked.length > 1 ? ranked[0].score - ranked[1].score : 100;
  const openFlips = questions.filter((q) => q.changes_recommendation).length;
  let level: Level;
  if (margin >= CLEAR_MARGIN && openFlips === 0) level = "high";
  else if (margin < THIN_MARGIN || openFlips >= 2) level = "low";
  else level = "medium";
  return { level, margin, open_questions: openFlips };
}
