/**
 * Port of backend/app/advisor/scoring.py: deterministic, explainable scores for
 * the three strategies. Same weights, same order of additions (so the same
 * floating-point sums), same Python rounding.
 */
import type {
  HybridRouting,
  ProjectProfile,
  QuestionMix,
  StrategyName,
  StrategyRecommendation,
  SuggestedConfig,
} from "../api/advisor";
import engine from "./generated/engine.json";
import { msg, pct } from "./messages";
import { pyRound } from "./pyfmt";

/** The backend settings the scorer reads, at their defaults (see the export script). */
export const SETTINGS = engine.settings;
const EMBEDDING_IS_MULTILINGUAL = engine.embedding_is_multilingual;

export const STRATEGIES: StrategyName[] = ["classic", "graph", "agentic"];

// Measured on golden_fr_business_v1 (tokens per question, p50 latency); classic = 1.0.
const BASELINE_RELATIVE_TOKENS: Record<StrategyName, number> = {
  classic: 1.0,
  graph: pyRound(6_732 / 6_336, 2),
  agentic: pyRound(10_367 / 6_336, 2),
};
const BASELINE_RELATIVE_LATENCY: Record<StrategyName, number> = {
  classic: 1.0,
  graph: pyRound(27_423 / 27_251, 2),
  agentic: pyRound(9_207 / 27_251, 2),
};

const SIZE_ORDINAL: Record<ProjectProfile["corpus_size"], number> = {
  tiny: 0,
  small: 1,
  medium: 2,
  large: 3,
  xlarge: 4,
};

const TIGHT_LATENCY_MS = 5_000;
const VERY_TIGHT_LATENCY_MS = 2_000;
const GENEROUS_LATENCY_MS = 30_000;

const MULTILINGUAL_EMBEDDING = "BAAI/bge-m3";
const MULTILINGUAL_EMBEDDING_ALT = "intfloat/multilingual-e5-large";
const MULTILINGUAL_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1";
const SMALL_MODEL = "claude-haiku-4-5";

const SHORT_DOC_HINTS = ["faq", "ticket", "chat", "email", "message", "q&a", "snippet"];
const LONG_DOC_HINTS = ["contract", "legal", "manual", "report", "paper", "book", "regulation", "spec"];

const W = "[\\p{L}\\p{N}_]";
const EU_PATTERN = new RegExp(
  `(?<!${W})(eu|ue|europe${W}*|gdpr|rgpd|union europ${W}*|france|germany|allemagne|schrems)(?!${W})`,
  "u",
);
const ONPREM_HINTS = ["on-prem", "on prem", "onprem", "premise", "self-host", "air-gap", "airgap", "sovereign", "souverain", "sur site"];

/** Accumulates a score and the sentences that justify it. */
class Score {
  reasons: string[];
  tradeoffs: string[] = [];
  constructor(
    public value: number,
    reasons: string[] = [],
  ) {
    this.reasons = reasons;
  }

  add(delta: number, reason?: string | null): void {
    this.value += delta;
    if (reason) this.reasons.push(reason);
  }

  cap(ceiling: number, reason: string): void {
    if (this.value > ceiling) {
      this.value = ceiling;
      this.reasons.push(reason);
    }
  }

  get clamped(): number {
    return pyRound(Math.max(0, Math.min(100, this.value)));
  }
}

// --- profile predicates -----------------------------------------------------------

export function mixFractions(p: ProjectProfile): [number, number, number] {
  const m: QuestionMix = p.question_mix;
  return [m.single_fact / 100, m.relational_multi_hop / 100, m.exploratory_multi_step / 100];
}

const isTightLatency = (p: ProjectProfile) => p.latency_budget_ms != null && p.latency_budget_ms <= TIGHT_LATENCY_MS;
const isGenerousLatency = (p: ProjectProfile) => p.latency_budget_ms != null && p.latency_budget_ms >= GENEROUS_LATENCY_MS;
const isNonEnglish = (p: ProjectProfile) => p.languages.some((l) => !l.toLowerCase().startsWith("en"));
const haystack = (p: ProjectProfile) => [...p.compliance, p.data_residency ?? ""].join(" ").toLowerCase();
const needsEuResidency = (p: ProjectProfile) => EU_PATTERN.test(haystack(p));
const needsOnPrem = (p: ProjectProfile) => ONPREM_HINTS.some((h) => haystack(p).includes(h));
const graphViable = (p: ProjectProfile) => SIZE_ORDINAL[p.corpus_size] >= 2 && p.entity_richness !== "low";
const agenticViable = (p: ProjectProfile) => !isTightLatency(p) && p.cost_sensitivity !== "high";

// --- per-strategy scores -------------------------------------------------------------

function scoreClassic(p: ProjectProfile, lang: string): Score {
  const [sf, rel, exp] = mixFractions(p);
  const s = new Score(60, [msg("classic.baseline", lang)]);
  s.add(25 * sf, sf >= 0.4 ? msg("classic.single_fact", lang, { pct: pct(sf, lang) }) : null);
  if (rel > 0) s.add(-15 * rel, rel >= 0.4 ? msg("classic.relational", lang) : null);
  if (exp > 0) s.add(-15 * exp, exp >= 0.4 ? msg("classic.exploratory", lang) : null);
  if (isTightLatency(p)) s.add(10, msg("classic.tight_latency", lang, { ms: p.latency_budget_ms! }));
  if (p.cost_sensitivity === "high") s.add(8, msg("classic.high_cost", lang));
  if (SIZE_ORDINAL[p.corpus_size] <= 1) s.add(5, msg("classic.small_corpus", lang));
  if (p.data_freshness === "daily" || p.data_freshness === "realtime") s.add(4, msg("classic.fresh", lang));
  s.tradeoffs.push(msg("classic.tradeoff.single_pass", lang), msg("classic.tradeoff.rerank", lang));
  return s;
}

function scoreGraph(p: ProjectProfile, lang: string): Score {
  const [, rel] = mixFractions(p);
  const s = new Score(45);
  s.add(40 * rel, rel >= 0.3 ? msg("graph.relational", lang, { pct: pct(rel, lang) }) : null);
  const size = p.corpus_size;
  const sizeLabel = msg(`size.${size}`, lang);
  const sizeDelta = { tiny: -10, small: -10, medium: 5, large: 12, xlarge: 15 }[size];
  s.add(
    sizeDelta,
    sizeDelta > 0
      ? msg("graph.size_large", lang, { size: sizeLabel })
      : msg("graph.size_small", lang, { size: sizeLabel }),
  );
  const entDelta = { low: -12, medium: 3, high: 12 }[p.entity_richness];
  s.add(
    entDelta,
    entDelta > 3 ? msg("graph.entities_rich", lang) : entDelta < 0 ? msg("graph.entities_sparse", lang) : null,
  );
  if (p.data_freshness === "daily") s.add(-6, msg("graph.daily", lang));
  else if (p.data_freshness === "realtime") s.add(-12, msg("graph.realtime", lang));
  if (p.cost_sensitivity === "high") s.add(-5, msg("graph.high_cost", lang));
  s.tradeoffs.push(
    msg("graph.tradeoff.extraction", lang, { model: SETTINGS.graph_extraction_model }),
    msg("graph.tradeoff.top_k", lang),
    msg("graph.tradeoff.quality", lang),
  );
  return s;
}

function scoreAgentic(p: ProjectProfile, lang: string): Score {
  const [sf, rel, exp] = mixFractions(p);
  const s = new Score(30);
  s.add(50 * exp, exp >= 0.3 ? msg("agentic.exploratory", lang, { pct: pct(exp, lang) }) : null);
  s.add(25 * rel, rel >= 0.3 ? msg("agentic.relational", lang) : null);
  if (sf > 0) s.add(-15 * sf, sf >= 0.5 ? msg("agentic.single_fact", lang) : null);
  const costDelta = { low: 5, medium: -5, high: -15 }[p.cost_sensitivity];
  s.add(
    costDelta,
    costDelta > 0
      ? msg("agentic.low_cost", lang)
      : msg("agentic.cost", lang, { level: msg(`level.${p.cost_sensitivity}`, lang) }),
  );
  if (isTightLatency(p)) s.add(-20, msg("agentic.tight_latency", lang, { ms: p.latency_budget_ms! }));
  else if (isGenerousLatency(p)) s.add(5, msg("agentic.generous_latency", lang));
  if (!agenticViable(p)) s.cap(40, msg("agentic.capped", lang));
  s.tradeoffs.push(
    msg("agentic.tradeoff.tokens", lang),
    msg("agentic.tradeoff.latency", lang, { iters: SETTINGS.agentic_max_iters }),
    msg("agentic.tradeoff.precision", lang),
  );
  return s;
}

// --- suggested configuration ---------------------------------------------------------

function chunkSize(p: ProjectProfile): number {
  const types = p.document_types.join(" ").toLowerCase();
  const longDocs = LONG_DOC_HINTS.some((h) => types.includes(h));
  if (SHORT_DOC_HINTS.some((h) => types.includes(h)) && !longDocs) return 300;
  if (longDocs) return 800;
  return 600;
}

export function suggestedConfig(strategy: StrategyName, p: ProjectProfile, lang: string): SuggestedConfig {
  const chunk = chunkSize(p);
  const overlap = pyRound(chunk * 0.13);
  const multilingual = isNonEnglish(p);
  const embedding = multilingual ? MULTILINGUAL_EMBEDDING : SETTINGS.embedding_model;
  const reranker = multilingual ? MULTILINGUAL_RERANKER : SETTINGS.reranker_model;
  const size = SIZE_ORDINAL[p.corpus_size];
  const rerankTopK = p.cost_sensitivity === "high" ? 5 : 8;
  let generator = SETTINGS.generator_model;
  const notes: string[] = [];

  if (multilingual) {
    notes.push(
      msg("config.multilingual", lang, { embedding: MULTILINGUAL_EMBEDDING, embedding_alt: MULTILINGUAL_EMBEDDING_ALT }),
    );
  }
  const [sf] = mixFractions(p);
  if (p.cost_sensitivity === "high" && sf >= 0.6 && strategy === "classic") {
    generator = SMALL_MODEL;
    notes.push(msg("config.small_model", lang, { model: SMALL_MODEL }));
  }

  if (strategy === "classic") {
    const rerank = !(p.latency_budget_ms != null && p.latency_budget_ms <= VERY_TIGHT_LATENCY_MS);
    if (!rerank) notes.push(msg("config.rerank_off", lang));
    return {
      retrieval_top_k: [20, 30, 50, 50, 50][size],
      rerank_top_k: rerankTopK,
      rerank,
      chunk_size_tokens: chunk,
      chunk_overlap_tokens: overlap,
      embedding_model: embedding,
      models: { generator, reranker },
      expected_relative_cost: BASELINE_RELATIVE_TOKENS.classic,
      expected_relative_latency: BASELINE_RELATIVE_LATENCY.classic,
      notes,
    };
  }

  if (strategy === "graph") {
    notes.push(msg("config.graph_top_k", lang), msg("config.graph_build", lang));
    return {
      retrieval_top_k: size >= 3 ? 15 : 10,
      rerank_top_k: rerankTopK,
      rerank: true,
      chunk_size_tokens: chunk,
      chunk_overlap_tokens: overlap,
      embedding_model: embedding,
      models: { generator, graph_extraction: SETTINGS.graph_extraction_model, reranker },
      expected_relative_cost: BASELINE_RELATIVE_TOKENS.graph,
      expected_relative_latency: BASELINE_RELATIVE_LATENCY.graph,
      notes: [...notes, msg("config.graph_cost", lang)],
    };
  }

  notes.push(msg("config.agentic_iters", lang, { iters: SETTINGS.agentic_max_iters }));
  return {
    retrieval_top_k: 5,
    rerank_top_k: rerankTopK,
    rerank: false,
    chunk_size_tokens: chunk,
    chunk_overlap_tokens: overlap,
    embedding_model: embedding,
    models: { agent: SETTINGS.agentic_model },
    expected_relative_cost: BASELINE_RELATIVE_TOKENS.agentic,
    expected_relative_latency: BASELINE_RELATIVE_LATENCY.agentic,
    notes: [...notes, msg("config.agentic_latency", lang)],
  };
}

// --- ranking, routing and notes --------------------------------------------------------

/** Score and rank the strategies. Ties break classic, graph, agentic. */
export function scoreStrategies(p: ProjectProfile, lang = "en"): StrategyRecommendation[] {
  const scores: Record<StrategyName, Score> = {
    classic: scoreClassic(p, lang),
    graph: scoreGraph(p, lang),
    agentic: scoreAgentic(p, lang),
  };
  if (SIZE_ORDINAL[p.corpus_size] <= 1) {
    scores.graph.cap(scores.classic.value - 5, msg("graph.capped_small", lang));
  }
  const order = [...STRATEGIES].sort(
    (a, b) => scores[b].clamped - scores[a].clamped || STRATEGIES.indexOf(a) - STRATEGIES.indexOf(b),
  );
  return order.map((name, i) => ({
    strategy: name,
    rank: i + 1,
    score: scores[name].clamped,
    reasons: scores[name].reasons,
    tradeoffs: scores[name].tradeoffs,
    suggested_config: suggestedConfig(name, p, lang),
  }));
}

/** Suggest routing by question type when the question mix is split. */
export function hybridRouting(p: ProjectProfile, lang = "en"): HybridRouting {
  const [sf, rel, exp] = mixFractions(p);
  const shares: [keyof QuestionMix, number][] = [
    ["single_fact", sf],
    ["relational_multi_hop", rel],
    ["exploratory_multi_step", exp],
  ];
  const values = shares.map(([, v]) => v);
  const split = Math.max(...values) < 0.7 && values.filter((v) => v >= 0.25).length >= 2;
  const targets: Record<keyof QuestionMix, StrategyName> = {
    single_fact: "classic",
    relational_multi_hop: graphViable(p) ? "graph" : "classic",
    exploratory_multi_step: agenticViable(p) ? "agentic" : "classic",
  };
  const routes = shares
    .filter(([, v]) => v > 0)
    .map(([qt, v]) => ({ question_type: qt, strategy: targets[qt], share_pct: pyRound(v * 100, 1) }));
  const distinct = new Set(routes.filter((r) => r.share_pct >= 20).map((r) => r.strategy));
  if (!split) return { recommended: false, rationale: msg("hybrid.single", lang), routes };
  if (distinct.size < 2) return { recommended: false, rationale: msg("hybrid.collapsed", lang), routes };
  return { recommended: true, rationale: msg("hybrid.recommended", lang), routes };
}

/** Generic deployment notes for residency, compliance and language needs. */
export function complianceNotes(p: ProjectProfile, lang = "en"): string[] {
  const notes: string[] = [];
  if (needsEuResidency(p)) {
    notes.push(msg("notes.eu_hosting", lang), msg("notes.eu_claude", lang), msg("notes.eu_embeddings", lang));
  }
  if (p.compliance.some((c) => c.toLowerCase().includes("gdpr") || c.toLowerCase().includes("rgpd"))) {
    notes.push(msg("notes.gdpr", lang), msg("notes.gdpr_tracing", lang));
  }
  if (p.compliance.some((c) => c.toLowerCase().includes("sensitive"))) notes.push(msg("notes.sensitive", lang));
  if (needsOnPrem(p)) notes.push(msg("notes.on_prem", lang));
  if (isNonEnglish(p)) {
    const langs = p.languages.join(", ");
    if (EMBEDDING_IS_MULTILINGUAL) {
      notes.push(
        msg("notes.languages_multilingual", lang, {
          langs,
          embedding: SETTINGS.embedding_model,
          better: MULTILINGUAL_EMBEDDING,
        }),
      );
    } else {
      notes.push(
        msg("notes.languages_english_only", lang, {
          langs,
          embedding: SETTINGS.embedding_model,
          reranker: SETTINGS.reranker_model,
          multi: MULTILINGUAL_EMBEDDING,
          multi_alt: MULTILINGUAL_EMBEDDING_ALT,
          multi_reranker: MULTILINGUAL_RERANKER,
        }),
      );
    }
  }
  if (notes.length > 0) notes.push(msg("notes.disclaimer", lang));
  return notes;
}
