/**
 * The strategy advisor, in the browser: the same answer POST /advise gives,
 * computed locally. Used by the static (Netlify) demo, which has no backend.
 *
 * Keep it in step with the Python advisor by re-running
 * scripts/export_advisor_fixtures.py; engine.test.ts checks every fixture.
 */
import type { AdviseOverrides, AdviseResponse, ProjectProfile } from "../api/advisor";
import { confidence, followUpQuestions } from "./followup";
import { extractWithBrowserKey } from "./llm";
import { msg, normalize } from "./messages";
import { applyOverrides, extractHeuristic, parseOverrides } from "./profile";
import { complianceNotes, hybridRouting, scoreStrategies } from "./scoring";

/** Assemble the /advise response for a profile, exactly as the route does. */
export function adviceFor(profile: ProjectProfile, language: string = "en"): AdviseResponse {
  const lang = normalize(language);
  const ranked = scoreStrategies(profile, lang);
  const questions = followUpQuestions(profile, ranked);
  const topTwo = ranked.slice(0, 2).map((r) => r.strategy);
  const examples = profile.example_questions.slice(0, 3);
  const exampleQuestions = examples.length > 0 ? examples : [msg("next_step.example_question", lang)];
  return {
    profile,
    recommendations: ranked,
    top_strategy: ranked[0].strategy,
    hybrid_routing: hybridRouting(profile, lang),
    compliance_notes: complianceNotes(profile, lang),
    next_step: {
      summary: msg("next_step.summary", lang, { strategies: topTwo.join(msg("next_step.and", lang)) }),
      validate_endpoint: "/advise/validate",
      suggested_strategies: topTwo,
      example_payload: {
        questions: exampleQuestions.map((q) => ({ question: q, ideal_answer: null })),
        strategies: topTwo,
      },
    },
    follow_up_questions: questions,
    confidence: confidence(ranked, questions),
  };
}

/** Heuristic advice: no network, no key. */
export function adviseHeuristic(description: string, overrides: AdviseOverrides = {}, language = "en"): AdviseResponse {
  const profile = applyOverrides(extractHeuristic(description), parseOverrides({ ...overrides }));
  return adviceFor(profile, language);
}

/**
 * Advice computed in the browser. With an Anthropic key, Claude reads the
 * description (like the backend does); any failure falls back to the keyword rules.
 */
export async function adviseLocally(
  description: string,
  overrides: AdviseOverrides = {},
  language = "en",
  apiKey?: string | null,
): Promise<AdviseResponse> {
  let profile: ProjectProfile;
  if (apiKey) {
    try {
      profile = await extractWithBrowserKey(description, apiKey);
    } catch {
      profile = extractHeuristic(description);
    }
  } else {
    profile = extractHeuristic(description);
  }
  return adviceFor(applyOverrides(profile, parseOverrides({ ...overrides })), language);
}
