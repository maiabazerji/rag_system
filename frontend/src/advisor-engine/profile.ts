/**
 * Port of backend/app/advisor/profile.py (heuristic extraction and overrides)
 * and the profile parts of schemas.py (corpus buckets, question-mix
 * normalisation, level aliases).
 *
 * Python's `\b` and `\w` are Unicode-aware ("é" is a word character), while
 * JavaScript's are ASCII-only even with the `u` flag. The patterns below spell
 * them out with `W`, so "contrôles" does not count the French word "les" as it
 * would with a JavaScript `\b`.
 */
import type { AdviseOverrides, Freshness, Level, ProjectProfile, QuestionMix } from "../api/advisor";
import { pyRound } from "./pyfmt";

/** Python's Unicode `\w`. */
const W = "[\\p{L}\\p{N}_]";
/** Python's `\b` before a word character, and after one. */
const B = `(?<!${W})`;
const E = `(?!${W})`;

type CorpusSize = ProjectProfile["corpus_size"];

const CORPUS_BUCKETS: [number, CorpusSize][] = [
  [100, "tiny"],
  [1_000, "small"],
  [10_000, "medium"],
  [100_000, "large"],
];

/** tiny < 100 <= small < 1k <= medium < 10k <= large < 100k <= xlarge. */
export function corpusBucket(docs: number): CorpusSize {
  for (const [limit, bucket] of CORPUS_BUCKETS) if (docs < limit) return bucket;
  return "xlarge";
}

/** Accept "med"/"mid"/"moderate" for "medium", like the Python validators. */
export function normalizeLevel(value: unknown): unknown {
  if (typeof value === "string") {
    const v = value.trim().toLowerCase();
    return ["med", "mid", "moderate"].includes(v) ? "medium" : v;
  }
  return value;
}

/** QuestionMix's validator: rescale to sum to 100, one decimal each. */
export function normalizeMix(mix: Partial<QuestionMix> = {}): QuestionMix {
  let sf = mix.single_fact ?? 60;
  let rel = mix.relational_multi_hop ?? 25;
  let exp = mix.exploratory_multi_step ?? 15;
  const total = sf + rel + exp;
  if (total <= 0) return { single_fact: 60, relational_multi_hop: 25, exploratory_multi_step: 15 };
  const scale = 100 / total;
  sf = pyRound(sf * scale, 1);
  rel = pyRound(rel * scale, 1);
  exp = pyRound(100 - sf - rel, 1);
  return { single_fact: sf, relational_multi_hop: rel, exploratory_multi_step: exp };
}

// --- heuristic extraction -------------------------------------------------------

const NUM_UNIT = new RegExp(
  `(\\d[\\d\\s.,]*)\\s*(?:(k|m|million|millions|mille|thousand)${E})?\\s*` +
    // One qualifier word may sit before the unit: "300 help pages", "60 Notion pages".
    "(?:de\\s+|d')?([a-zà-ÿ-]+\\s+)?" +
    "(documents?|docs?|fichiers?|files?|pages?|pdfs?|articles?|tickets?|" +
    `contrats?|contracts?|records?|emails?|mails?)${E}`,
  "giu",
);
const LATENCY = new RegExp(
  `(\\d+(?:[.,]\\d+)?)\\s*(ms|milliseconds?|millisecondes?|s|secs?|seconds?|secondes?)${E}`,
  "iu",
);
// "Pas de contrainte de temps réel" denies real time: it must not read as a tight budget.
const NO_REALTIME =
  "(pas de|sans|aucune?|no|not|without)\\s+(contrainte de\\s+|need for\\s+)?(temps r[ée]el|real[- ]time)";
const GENEROUS_LATENCY =
  `latenc${W}* (n'est pas|is not|isn't) (un probl[èe]me|une contrainte|an issue|a concern|` +
  "a problem|critical|critique|important)|pas de contrainte de (temps|latence|d[ée]lai)|" +
  "no latency (constraint|requirement)|(latency|speed) (does not|doesn't) matter|" +
  "can take (a|one|several) minutes?|peu(t|vent) prendre (une|quelques|plusieurs) minutes?";
/** Budget assumed when the description says latency does not matter (above the generous bar). */
const GENEROUS_LATENCY_HINT_MS = 60_000;
const SENSITIVE =
  "donn[ée]es (sensibles|confidentielles)|sensitive|confidenti|secret professionnel|" +
  "secret m[ée]dical|health data|donn[ée]es de sant[ée]";
const PERSONAL_DATA = `donn[ée]es (personnelles|à caract[èe]re personnel)|personal data|${B}pii${E}`;

const LANGS: [string, RegExp][] = [
  ["fr", new RegExp(`${B}(french|fran[cç]ais|francophone)${E}`, "u")],
  ["en", new RegExp(`${B}(english|anglais)${E}`, "u")],
  ["de", new RegExp(`${B}(german|allemand|deutsch)${E}`, "u")],
  ["es", new RegExp(`${B}(spanish|espagnol|espa[nñ]ol)${E}`, "u")],
  ["it", new RegExp(`${B}(italian|italien)${E}`, "u")],
  ["nl", new RegExp(`${B}(dutch|n[eé]erlandais)${E}`, "u")],
  ["pt", new RegExp(`${B}(portuguese|portugais)${E}`, "u")],
];
const FRENCH_STOPWORDS = new RegExp(`${B}(nous|notre|nos|les|des|est|une|pour|avec|dans|sont|qui)${E}`, "giu");

const SINGLE_FACT = [
  "faq", "lookup", "what is", "quick answer", "support", "policy", "policies", "definition",
  "procedure", "how do i", "hr", "helpdesk", "quel est", "quelle est", "combien", "prix",
  "price", "horaires", "politique", "comment faire",
];
const RELATIONAL = [
  "relationship", "relation", "connected", "between", "link", "lien", "entre", "dependenc",
  "supplier", "fournisseur", "graph", "entity", "entities", "entité", "multi-hop",
  "who works", "subsidiar", "filiale", "ownership", "cross-reference", "croiser",
];
const EXPLORATORY = [
  "research", "analysis", "analyse", "investigat", "explore", "synthes", "report on",
  "rapport", "why", "pourquoi", "open-ended", "due diligence", "recherche", "compare",
  "comparer", "trend", "tendance", "summari", "résum", "ouverte", "plusieurs documents",
  "multiple documents", "several documents", "across documents",
];
const ENTITIES = [
  "contract", "contrat", "legal", "juridique", "compan", "entreprise", "société",
  "supplier", "fournisseur", "people", "personne", "customer", "client", "product",
  "produit", "org chart", "organigramme", "crm", "erp", "part number", "patent", "brevet",
  "drug", "médicament", "gene", "clause",
];
const DOC_TYPES: [string, string[]][] = [
  ["contracts", ["contract", "contrat"]],
  ["support tickets", ["ticket"]],
  ["faq", ["faq"]],
  ["manuals", ["manual", "manuel", "documentation technique"]],
  ["emails", ["email", "e-mail", "courriel"]],
  ["reports", ["report", "rapport"]],
  ["policies", ["policy", "policies", "politique", "procédure", "procedure"]],
  ["wiki pages", ["wiki", "confluence", "notion"]],
  ["pdf documents", ["pdf"]],
  ["scientific papers", ["paper", "publication", "article scientifique"]],
  ["invoices", ["invoice", "facture"]],
  ["legal texts", ["regulation", "réglementation", "loi", "law"]],
];

/** Python's `str.count`: non-overlapping occurrences. */
function countOf(text: string, word: string): number {
  return text.split(word).length - 1;
}

function countAll(text: string, words: string[]): number {
  return words.reduce((n, w) => n + countOf(text, w), 0);
}

function parseNumber(raw: string, scale: string | undefined): number | null {
  const digits = raw.replace(/[\s.,]/gu, "");
  if (!/^\d+$/.test(digits)) return null;
  let n = Number(digits);
  const s = (scale ?? "").toLowerCase();
  if (s === "k" || s === "mille" || s === "thousand") n *= 1_000;
  else if (s === "m" || s === "million" || s === "millions") n *= 1_000_000;
  return n;
}

/** Python's `str.strip(chars)`. */
function stripChars(s: string, chars: string): string {
  const set = new Set(chars);
  const cps = [...s];
  let a = 0;
  let b = cps.length;
  while (a < b && set.has(cps[a])) a++;
  while (b > a && set.has(cps[b - 1])) b--;
  return cps.slice(a, b).join("");
}

const test = (pattern: string, text: string) => new RegExp(pattern, "u").test(text);

/** Keyword-rule extraction (English and French). Always succeeds. */
export function extractHeuristic(description: string): ProjectProfile {
  const text = description.toLowerCase();

  let docs: number | null = null;
  for (const m of description.matchAll(NUM_UNIT)) {
    let n = parseNumber(m[1], m[2]);
    if (n !== null) {
      // "2 million pages of filings" is page count, not documents; "300 help
      // pages" names the documents themselves.
      if (m[4].toLowerCase().startsWith("page") && m[3] === undefined) n = Math.max(1, Math.floor(n / 10));
      docs = Math.max(docs ?? 0, n);
    }
  }

  const langs = LANGS.filter(([, pat]) => pat.test(text)).map(([code]) => code);
  if ((description.match(FRENCH_STOPWORDS) ?? []).length >= 3 && !langs.includes("fr")) langs.push("fr");
  if (langs.length === 0) langs.push("en");

  const mix = normalizeMix({
    single_fact: 1.5 + countAll(text, SINGLE_FACT),
    relational_multi_hop: 0.5 + countAll(text, RELATIONAL),
    exploratory_multi_step: 0.5 + countAll(text, EXPLORATORY),
  });

  const noRealtime = test(NO_REALTIME, text);
  let latency: number | null = null;
  const lat = LATENCY.exec(text);
  if (test(GENEROUS_LATENCY, text)) {
    latency = GENEROUS_LATENCY_HINT_MS;
  } else if (lat) {
    const value = Number(lat[1].replace(",", "."));
    latency = Math.trunc(lat[2].startsWith("m") ? value : value * 1000);
    latency = Math.max(100, latency);
  } else if (!noRealtime && test("real[- ]time|temps r[ée]el|instant", text)) {
    latency = 2_000;
  }

  let cost: Level = "medium";
  if (test("cost is not|budget is not|no budget constraint|budget g[ée]n[ée]reux|money is not", text)) {
    cost = "low";
  } else if (
    test(
      "cheap|low cost|low-cost|tight budget|limited budget|pas cher|" +
        "budget (serr|limit)|co[uû]ts? (bas|r[ée]duit|ma[iî]tris)|économ|minimi[sz]e cost",
      text,
    )
  ) {
    cost = "high";
  }

  let freshness: Freshness = "weekly";
  if (!noRealtime && test(`real[- ]time|temps r[ée]el|${B}live${E}|continuous`, text)) freshness = "realtime";
  else if (test("daily|every day|quotidien|chaque jour|tous les jours", text)) freshness = "daily";
  else if (test("monthly|mensuel|chaque mois", text)) freshness = "monthly";
  else if (test("static|archive|rarely|rarement|never change", text)) freshness = "static";

  const compliance: string[] = [];
  let residency: string | null = null;
  if (test(`${B}gdpr${E}|${B}rgpd${E}`, text) || test(PERSONAL_DATA, text)) compliance.push("GDPR");
  if (test(SENSITIVE, text)) compliance.push("sensitive data");
  if (test(`${B}eu${E}|${B}ue${E}|europe|union europ|h[ée]berg${W}* en france|hosted in (the )?eu`, text)) {
    compliance.push("EU data residency");
    residency = "EU";
  }
  if (test("on[- ]?prem|on[- ]premise|sur site|self[- ]host|air[- ]?gap|souverain|sovereign", text)) {
    compliance.push("on-prem");
  }
  if (test(`${B}hipaa${E}`, text)) compliance.push("HIPAA");
  if (test(`${B}hds${E}`, text)) compliance.push("HDS");

  const entHits = countAll(text, ENTITIES);
  const entity: Level = entHits >= 3 ? "high" : entHits >= 1 ? "medium" : "low";

  const docTypes = DOC_TYPES.filter(([, kws]) => kws.some((k) => text.includes(k))).map(([label]) => label);
  const questions = (description.match(/[^.!?\n]*\?/g) ?? [])
    .filter((q) => [...q.trim()].length > 8)
    .map((q) => stripChars(q, " -*•\"'«»“”"))
    .slice(0, 10);

  return {
    corpus_size: docs !== null ? corpusBucket(docs) : "small",
    corpus_size_docs: docs,
    document_types: docTypes,
    languages: langs,
    question_mix: mix,
    latency_budget_ms: latency,
    cost_sensitivity: cost,
    data_freshness: freshness,
    compliance,
    data_residency: residency,
    entity_richness: entity,
    example_questions: questions,
    source: "heuristic",
    overridden_fields: [],
  };
}

/** Parse overrides like AdviseOverrides does (level aliases, normalised mix). */
export function parseOverrides(raw: Record<string, unknown>): AdviseOverrides {
  const o = { ...raw } as Record<string, unknown>;
  for (const key of ["cost_sensitivity", "entity_richness"]) {
    if (o[key] != null) o[key] = normalizeLevel(o[key]);
  }
  if (o.question_mix != null) o.question_mix = normalizeMix(o.question_mix as Partial<QuestionMix>);
  return o as AdviseOverrides;
}

/** Return a copy of `profile` where every explicit override wins. */
export function applyOverrides(profile: ProjectProfile, overrides: AdviseOverrides): ProjectProfile {
  const updates: Partial<ProjectProfile> = {};
  const fields: string[] = [];

  if (overrides.corpus_size_docs != null) {
    updates.corpus_size_docs = overrides.corpus_size_docs;
    updates.corpus_size = corpusBucket(overrides.corpus_size_docs);
    fields.push("corpus_size_docs");
  }
  if (overrides.languages && overrides.languages.length > 0) {
    updates.languages = overrides.languages.filter(Boolean).map((l) => l.trim().toLowerCase());
    fields.push("languages");
  }
  if (overrides.latency_budget_ms != null) {
    updates.latency_budget_ms = overrides.latency_budget_ms;
    fields.push("latency_budget_ms");
  }
  if (overrides.cost_sensitivity != null) {
    updates.cost_sensitivity = overrides.cost_sensitivity;
    fields.push("cost_sensitivity");
  }
  if (overrides.data_freshness != null) {
    updates.data_freshness = overrides.data_freshness;
    fields.push("data_freshness");
  }
  if (overrides.question_examples && overrides.question_examples.length > 0) {
    updates.example_questions = overrides.question_examples.slice(0, 20);
    fields.push("question_examples");
  }
  if (overrides.compliance != null) {
    updates.compliance = overrides.compliance;
    const joined = overrides.compliance.join(" ").toLowerCase();
    if (test(`${B}eu${E}|${B}ue${E}|europe|gdpr|rgpd`, joined)) updates.data_residency = "EU";
    fields.push("compliance");
  }
  if (overrides.entity_richness != null) {
    updates.entity_richness = overrides.entity_richness;
    fields.push("entity_richness");
  }
  if (overrides.question_mix != null) {
    updates.question_mix = overrides.question_mix;
    fields.push("question_mix");
  }
  return { ...profile, ...updates, overridden_fields: fields };
}
