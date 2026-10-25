/**
 * Client for the strategy advisor: POST /advise and POST /advise/validate.
 *
 * Kept separate from client.ts; it reuses that module's `post` helper so auth
 * headers and error handling stay identical.
 */
import { adviseLocally } from "../advisor-engine";
import { getBrowserKey } from "../advisor-engine/llm";
import { isStaticDemo } from "../staticDemo";
import { LONG_TIMEOUT_MS, del, post } from "./client";

export type StrategyName = "classic" | "graph" | "agentic";
export type Level = "low" | "medium" | "high";
export type Freshness = "static" | "monthly" | "weekly" | "daily" | "realtime";

export interface QuestionMix {
  single_fact: number;
  relational_multi_hop: number;
  exploratory_multi_step: number;
}

export interface AdviseOverrides {
  corpus_size_docs?: number;
  languages?: string[];
  latency_budget_ms?: number;
  cost_sensitivity?: Level;
  data_freshness?: Freshness;
  question_examples?: string[];
  compliance?: string[];
  entity_richness?: Level;
  question_mix?: QuestionMix;
}

export interface ProjectProfile {
  corpus_size: "tiny" | "small" | "medium" | "large" | "xlarge";
  corpus_size_docs: number | null;
  document_types: string[];
  languages: string[];
  question_mix: QuestionMix;
  latency_budget_ms: number | null;
  cost_sensitivity: Level;
  data_freshness: Freshness;
  compliance: string[];
  data_residency: string | null;
  entity_richness: Level;
  example_questions: string[];
  source: "llm" | "heuristic";
  overridden_fields: string[];
}

export interface SuggestedConfig {
  retrieval_top_k: number;
  rerank_top_k: number;
  rerank: boolean;
  chunk_size_tokens: number;
  chunk_overlap_tokens: number;
  embedding_model: string;
  models: Record<string, string>;
  expected_relative_cost: number;
  expected_relative_latency: number;
  notes: string[];
}

export interface StrategyRecommendation {
  strategy: StrategyName;
  rank: number;
  score: number;
  reasons: string[];
  tradeoffs: string[];
  suggested_config: SuggestedConfig;
}

export interface HybridRouting {
  recommended: boolean;
  rationale: string;
  routes: { question_type: keyof QuestionMix; strategy: StrategyName; share_pct: number }[];
}

export interface AdviseResponse {
  profile: ProjectProfile;
  recommendations: StrategyRecommendation[];
  top_strategy: StrategyName;
  hybrid_routing: HybridRouting;
  compliance_notes: string[];
  next_step: {
    summary: string;
    validate_endpoint: string;
    suggested_strategies: StrategyName[];
    example_payload: Record<string, unknown>;
  };
  /** Facts worth confirming, most decisive first. Older backends omit it. */
  follow_up_questions?: FollowUpQuestion[];
  confidence?: Confidence | null;
}

/** Ids of the follow-up questions: each is the AdviseOverrides field it sets. */
export type FollowUpId =
  | "question_mix"
  | "corpus_size_docs"
  | "entity_richness"
  | "latency_budget_ms"
  | "cost_sensitivity"
  | "data_freshness"
  | "compliance";

export interface FollowUpOption {
  id: string;
  /** Merge into the request overrides to give this answer. */
  overrides: AdviseOverrides;
  /** Strategy that would come first with this answer. */
  top_strategy: StrategyName;
}

export interface FollowUpQuestion {
  id: FollowUpId;
  options: FollowUpOption[];
  /** Option the description pointed to (a guess), if any. */
  current: string | null;
  changes_recommendation: boolean;
}

export interface Confidence {
  level: Level;
  margin: number;
  open_questions: number;
}

export interface ValidateQuestion {
  question: string;
  ideal_answer?: string | null;
}

export interface StrategyScorecard {
  strategy: StrategyName;
  questions: number;
  errors: number;
  refusals: number;
  refusal_rate: number;
  avg_latency_ms: number;
  max_latency_ms: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_tokens_per_question: number;
  judged: number;
  avg_judge_score: number | null;
  avg_answer_f1: number | null;
}

export interface ValidateRow {
  question: string;
  strategy: StrategyName;
  answer: string;
  refusal: boolean;
  error: string | null;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  answer_f1: number | null;
  judge_score: number | null;
}

export interface ValidateResponse {
  strategies: StrategyName[];
  scorecard: StrategyScorecard[];
  measured_winner: StrategyName | null;
  winner_reason: string;
  rows: ValidateRow[];
}

export const MAX_VALIDATE_QUESTIONS = 20;

/** Drop empty and undefined override fields so the backend only sees real facts. */
export function cleanOverrides(o: AdviseOverrides): AdviseOverrides {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(o)) {
    if (v === undefined || v === null || v === "") continue;
    if (typeof v === "number" && Number.isNaN(v)) continue;
    if (Array.isArray(v) && v.length === 0) continue;
    out[k] = v;
  }
  return out as AdviseOverrides;
}

/** Language of the backend-written text (reasons, notes, next step). */
export type AdviceLanguage = "en" | "fr";

export function advise(
  description: string,
  overrides: AdviseOverrides = {},
  language?: AdviceLanguage,
): Promise<AdviseResponse> {
  // No backend in the static demo: the same advice, computed in the browser.
  if (isStaticDemo()) {
    return adviseLocally(description, cleanOverrides(overrides), language ?? "en", getBrowserKey());
  }
  return post<AdviseResponse>("/advise", {
    description,
    overrides: cleanOverrides(overrides),
    ...(language ? { language } : {}),
  });
}

/**
 * Parse one question per line. `question || ideal answer` attaches an expected
 * answer. Blank lines are skipped; only the first MAX_VALIDATE_QUESTIONS are kept.
 */
export function parseQuestions(text: string): ValidateQuestion[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const idx = line.indexOf("||");
      if (idx === -1) return { question: line };
      const question = line.slice(0, idx).trim();
      const ideal = line.slice(idx + 2).trim();
      return ideal ? { question, ideal_answer: ideal } : { question };
    })
    .filter((q) => q.question.length > 0)
    .slice(0, MAX_VALIDATE_QUESTIONS);
}

export function validateStrategies(
  questions: ValidateQuestion[],
  strategies?: StrategyName[],
  profile?: ProjectProfile,
  language?: AdviceLanguage,
  workspace?: string,
): Promise<ValidateResponse> {
  return post<ValidateResponse>(
    "/advise/validate",
    {
      questions,
      ...(strategies && strategies.length ? { strategies } : {}),
      ...(profile ? { profile } : {}),
      ...(language ? { language } : {}),
      ...(workspace ? { workspace } : {}),
    },
    { timeoutMs: LONG_TIMEOUT_MS },
  );
}

/** A fresh private test workspace id (an access group, see /advise/validate). */
export function newWorkspace(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(10));
  return "audit-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Erase every document uploaded into a test workspace. */
export function deleteWorkspace(workspace: string): Promise<unknown> {
  return del(`/advise/workspace/${encodeURIComponent(workspace)}`);
}
