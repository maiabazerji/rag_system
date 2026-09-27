/**
 * Client for the strategy advisor: POST /advise and POST /advise/validate.
 *
 * Kept separate from client.ts; it reuses that module's `post` helper so auth
 * headers and error handling stay identical.
 */
import { post } from "./client";

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

export function advise(description: string, overrides: AdviseOverrides = {}): Promise<AdviseResponse> {
  return post<AdviseResponse>("/advise", { description, overrides: cleanOverrides(overrides) });
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
): Promise<ValidateResponse> {
  return post<ValidateResponse>("/advise/validate", {
    questions,
    ...(strategies && strategies.length ? { strategies } : {}),
    ...(profile ? { profile } : {}),
  });
}
