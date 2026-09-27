/**
 * English UI strings: the source of truth for message keys.
 *
 * Placeholders are `{name}` and are filled by `t(key, { name })`. Keys ending in
 * `_one` / `_other` are plural pairs, picked by `tp(base, count)` with the
 * locale's plural rules. Every other dictionary must be typed `Dictionary`, so a
 * missing or extra key is a compile error.
 */
export const en = {
  // Language switch
  "lang.label": "Language",
  "lang.switchTo": "Switch language to {language}",
  "lang.en": "English",
  "lang.fr": "Français",

  // Shared
  "common.clear": "Clear",
  "common.save": "Save",
  "common.retry": "Retry",
  "common.dismissError": "Dismiss error",
  "common.running": "Running…",
  "common.loading": "Loading…",
  "common.na": "n/a",
  "common.refused": "Refused",
  "common.answered": "Answered",
  "common.error": "error",
  "common.sources": "Sources ({count})",
  "common.step": "step",

  // Strategy names
  "strategy.classic": "Classic RAG",
  "strategy.graph": "Graph RAG",
  "strategy.agentic": "Agentic RAG",
  "strategy.classic.short": "Classic",
  "strategy.graph.short": "Graph",
  "strategy.agentic.short": "Agentic",

  // Navigation
  "nav.ask": "Ask",
  "nav.ingest": "Ingest",
  "nav.compare": "Compare",
  "nav.data": "Data",
  "nav.eval": "Evaluation",
  "nav.regressions": "Regressions",
  "nav.advisor": "Advisor",
  "nav.tagline": "retrieval that grades itself",
  "nav.workspace": "Workspace",
  "nav.blurb":
    "A RAG system is only as good as the eval that catches it drifting. Ingest, ask, score, ship.",

  // API client errors
  "client.unauthorized": "This backend requires an API key.",
  "client.forbidden": "Not allowed{detail}. This action needs the backend's admin key (ADMIN_KEY).",
  "client.rateLimited": "Rate limit reached. Wait a moment and try again.",
  "client.unavailable":
    "Service unavailable: {detail}. A backing service (such as Qdrant) may be down; try again shortly.",
  "client.timeout": "The request timed out after {seconds}s.",
  "client.unreachable": "Could not reach the backend ({reason}).",

  // API key banner
  "apikey.rejected": "The backend rejected the saved API key",
  "apikey.required": "This backend requires an API key",
  "apikey.unreachable": "Could not reach the backend",
  "apikey.saved": "API key saved",
  "apikey.unreachableHelp":
    "{endpoint} failed, so it is unknown whether an API key is needed. Check the backend is running; if it requires a key, paste it below.",
  "apikey.storedHelp": "Requests are sent with {header}. The key is kept in this browser only.",
  "apikey.createHelp": "Create one with {command}, then paste it below.",
  "apikey.savedBadge": "saved",
  "apikey.label": "API key",
  "apikey.saveKey": "Save key",

  // Answer metadata
  "meta.latencyTip": "Time from request to response",
  "meta.costTip": "Estimated cost: {cost}",
  "meta.tokensTip": "Input + output tokens",
  "meta.tokensUsedTip": "Input + output tokens used",
  "meta.tokensCostTip": "Input + output tokens used (cost indicator)",
  "meta.iterationsTip": "Number of reasoning iterations",
  "meta.iterationsLoopsTip": "Number of reasoning iterations (loops)",
  "meta.strategy": "Strategy",
  "meta.model": "Model",
  "meta.latency": "Latency",
  "meta.tokens": "Tokens",
  "meta.iterations": "Iterations",

  // Ask
  "ask.title": "Ask",
  "ask.subtitle": "Question indexed documents, in English or French. Sources let you verify accuracy.",
  "ask.strategy": "Strategy",
  "ask.hint.classic": "vector search → rerank → answer",
  "ask.hint.graph": "entity walk on knowledge graph",
  "ask.hint.agentic": "model loops over search/fetch tools",
  "ask.strategyAria": "{label} strategy: {hint}",
  "ask.placeholder": "Type a question…  ⏎ to send  ·  shift+⏎ for newline",
  "ask.asking": "Asking…",
  "ask.submit": "Ask",
  "ask.tryQuestion": "Try a question",
  "ask.samples.g1": "Retrieval Strategy & Tradeoffs",
  "ask.samples.g1.q1":
    "How does entity extraction in Graph RAG reduce hallucination compared to Classic RAG?",
  "ask.samples.g1.q2":
    "When should you use dense retrieval over hybrid, and what are the latency-quality tradeoffs?",
  "ask.samples.g1.q3":
    "Why might Agentic RAG overshoot on simple queries while excelling on complex reasoning?",
  "ask.samples.g1.q4":
    "How does reranking affect both precision and recall in multi-hop retrieval scenarios?",
  "ask.samples.g2": "System Design & Architecture",
  "ask.samples.g2.q1":
    "What's the relationship between chunk size, embedding model, and retrieval quality?",
  "ask.samples.g2.q2":
    "How do you design a RAG pipeline to handle both factual and synthesis queries efficiently?",
  "ask.samples.g2.q3":
    "How does knowledge graph construction from documents affect retrieval coverage vs. hallucination?",
  "ask.samples.g2.q4":
    "What are the failure modes of dense-only retrieval with structurally ambiguous documents?",
  "ask.samples.g3": "Evaluation & Production",
  "ask.samples.g3.q1":
    "What's the distinction between context precision and context recall, and how do they interact?",
  "ask.samples.g3.q2":
    "How would you optimize a RAG pipeline for both speed and accuracy with heterogeneous documents?",
  "ask.samples.g3.q3":
    "Why does Agentic RAG use more tokens than Classic on the same corpus, and when is that justified?",
  "ask.samples.g3.q4": "How do you detect and prevent retrieval drift as your corpus grows over time?",
  "ask.loading": "Retrieving and generating answer...",
  "ask.confidenceTitle": "Model confidence in this answer",
  "ask.confidenceTip":
    "How much the model trusts its answer (0-1 scale). Higher = more confident it's grounded in sources.",
  "ask.confidenceAria": "Confidence score",
  "ask.confidencePercentAria": "{pct} percent confidence",
  "ask.trace": "trace",
  "ask.traceTitle": "GET /traces/{id}: the retrieval, rerank and generation steps for this answer",
  "ask.rated": "Rated {rating}/5, thanks. Your rating was saved.",
  "ask.yourRating": "Your rating",
  "ask.stars_one": "{count} star",
  "ask.stars_other": "{count} stars",
  "ask.ratingPlaceholder": "Optional: why this rating?",
  "ask.saving": "Saving…",
  "ask.submitRating": "Submit rating",

  // Compare
  "compare.title": "Compare",
  "compare.adminNeeded":
    'Building the graph needs the backend\'s admin key (ADMIN_KEY). Enter it in the "Admin key" field below and try again.{detail}',
  "compare.buildUnavailable":
    "Graph build is unavailable: the backend has no ADMIN_KEY configured, or a backing service is down. Set ADMIN_KEY in .env and restart the backend.{detail}",
  "compare.tagline.classic": "embed → vector search → rerank → answer",
  "compare.tagline.graph": "extract entities → walk knowledge graph → answer",
  "compare.tagline.agentic": "model loops over search & fetch tools",
  "compare.sample1":
    "How does entity extraction in Graph RAG reduce hallucination compared to dense-only retrieval?",
  "compare.sample2": "When should you prioritize Agentic RAG's reasoning over Classic RAG's speed?",
  "compare.sample3":
    "What happens to retrieval quality when you use smaller chunks with dense embeddings?",
  "compare.noJobId": "The backend did not return a graph build job id.",
  "compare.progressChunks": "{done}/{total} chunks",
  "compare.status.queued": "queued",
  "compare.status.running": "running",
  "compare.status.completed": "completed",
  "compare.status.failed": "failed",
  "compare.buildFailed": "Graph build failed: {error}",
  "compare.unknownError": "unknown error",
  "compare.partialFailures_one":
    "Graph built, but extraction failed for {count} chunk. Rebuild to retry it.",
  "compare.partialFailures_other":
    "Graph built, but extraction failed for {count} chunks. Rebuild to retry them.",
  "compare.placeholder": "Ask something… ⏎ to run",
  "compare.run": "Compare",
  "compare.adminKey": "Admin key",
  "compare.adminSaved": "(saved)",
  "compare.adminNeededHint": "(needed to build the graph)",
  "compare.adminPlaceholder": "ADMIN_KEY from .env",
  "compare.adminHelp": "Sent as {header} with graph builds only, and kept in this browser.",
  "compare.graphTip": "Knowledge graph: extracted entities and relationships from documents",
  "compare.triples": "{count} triples",
  "compare.graphNotBuilt": "Graph not built",
  "compare.buildAria": "Extract entity relationships from indexed documents",
  "compare.building": "Building… {progress}",
  "compare.rebuild": "Rebuild",
  "compare.buildGraph": "Build graph",
  "compare.showLess": "Show less",
  "compare.readMore": "Read more →",
  "compare.more": "+{count} more",
  "compare.kgDetails": "Knowledge graph details",
  "compare.extractedEntities": "Extracted Entities",
  "compare.relatedEntities": "Related Entities (1 hop)",
  "compare.kgStructure": "Knowledge Graph Structure",
  "compare.strategyDetails": "Strategy details",
  "compare.trace_one": "Trace ({count} step)",
  "compare.trace_other": "Trace ({count} steps)",
  "compare.winners": "Winners",
  "compare.fastest": "Fastest",
  "compare.cheapest": "Cheapest",
  "compare.faster": "↓ {pct} faster",
  "compare.cheaper": "↓ {pct} cheaper",

  // Evaluation
  "eval.title": "Evaluation",
  "eval.subtitle":
    "Score a golden dataset on faithfulness, relevance, and retrieval precision and recall.",
  "eval.metric.faithfulness": "Faithfulness",
  "eval.metric.faithfulness.desc": "Answer only asserts what's in retrieved context",
  "eval.metric.answer_relevance": "Relevance",
  "eval.metric.answer_relevance.desc": "Answer addresses the question, in its language",
  "eval.metric.context_precision": "Precision",
  "eval.metric.context_precision.desc": "Retrieved chunks are on-topic",
  "eval.metric.context_recall": "Recall",
  "eval.metric.context_recall.desc": "Retrieved all relevant chunks",
  "eval.dataset": "Dataset",
  "eval.dataset.golden_v1": "golden_v1 (English)",
  "eval.dataset.golden_v2": "golden_v2 (English, relabelled)",
  "eval.dataset.golden_fr_v1": "golden_fr_v1 (French questions)",
  "eval.running":
    "Running evaluation. This answers and judges every golden question, so it can take several minutes...",
  "eval.runFailed": "Evaluation run failed: {error}",
  "eval.loading": "Loading evaluation results...",
  "eval.loadFailed": "Failed to load evaluations: {error}",
  "eval.empty": "No scored evaluation runs yet",
  "eval.emptyUnscored_one":
    "{count} earlier run produced no scores. Run an evaluation to measure retrieval strategy performance.",
  "eval.emptyUnscored_other":
    "{count} earlier runs produced no scores. Run an evaluation to measure retrieval strategy performance.",
  "eval.emptyHelp":
    "Run an evaluation on the golden dataset to measure retrieval strategy performance.",
  "eval.run": "Run Evaluation",
  "eval.rerun": "Re-run",
  "eval.scored_one": "{scored} of {count} question scored",
  "eval.scored_other": "{scored} of {count} questions scored",
  "eval.hidden_one": "{count} earlier run is hidden because the judge returned no scores.",
  "eval.hidden_other": "{count} earlier runs are hidden because the judge returned no scores.",
  "eval.metricAria": "{label}: {value}",

  // Regressions
  "reg.title": "Regressions",
  "reg.subtitle":
    "Metric drift between consecutive eval runs. The early-warning system for prompt-and-pray.",
  "reg.loadFailed": "Failed to load regressions: {error}",
  "reg.none": "No regressions detected",
  "reg.stable": "All metrics are stable run-over-run.",

  // Ingest
  "ingest.title": "Ingest",
  "ingest.intro":
    "Drop in {types} files. They're chunked, embedded, and indexed into Qdrant, ready to ground answers on the Ask page.",
  "ingest.chunksIndexed": "{count} chunks indexed",
  "ingest.failedSome_one": "{failed} of {count} file failed to ingest; see the list below.",
  "ingest.failedSome_other": "{failed} of {count} files failed to ingest; see the list below.",
  "ingest.processing": "Processing and indexing files...",
  "ingest.dropHere": "Drop files here",
  "ingest.orBrowse": "or click below to browse",
  "ingest.choose": "Choose files",
  "ingest.recent": "Recently uploaded",
  "ingest.failed": "failed",
  "ingest.addedChunks": "+{count} chunks",
  "ingest.total": "{count} total",
  "ingest.groupsLabel": "Readable by (groups)",
  "ingest.groupsPlaceholder": "e.g. legal, finance",
  "ingest.groupsHint":
    "Comma-separated. Leave blank to use your own groups. Only groups you belong to are accepted.",
  "ingest.readableBy": "Readable by: {groups}",

  // Data
  "data.title": "Data",
  "data.subtitle": "What is indexed, and how the retrieval strategies compare on it.",
  "data.chunks": "Chunks",
  "data.tokens": "Tokens",
  "data.size": "Size",
  "data.status": "Status",
  "data.live": "Live",
  "data.dataset": "Dataset",
  "data.goldenPerf": "Golden Dataset Performance",
  "data.level.medium": "Medium",
  "data.level.hard": "Hard",
  "data.q1": "Entity extraction in Graph RAG vs dense-only?",
  "data.q2": "Trade latency for reasoning capability?",
  "data.q3": "Chunk size & embedding model interaction?",
  "data.patterns": "Strategy Patterns",
  "data.pattern.entity": "Entity-heavy",
  "data.pattern.entity.impact": "Graph +15% latency advantage",
  "data.pattern.factual": "Factual",
  "data.pattern.factual.impact": "Classic optimal for speed & cost",
  "data.pattern.synthesis": "Synthesis",
  "data.pattern.synthesis.impact": "Agentic discovers cross-document evidence",

  // Advisor
  "advisor.title": "Advisor",
  "advisor.intro":
    "Describe your project and get a ranked recommendation between {classic}, {graph} and {agentic} RAG, then validate it on your own questions.",
  "advisor.yourProject": "Your project",
  "advisor.placeholder":
    "Describe your documents, who asks questions, what kinds of questions, how fast answers must be, budget, and any hosting or compliance constraints…",
  "advisor.anyLanguage": "Any language works: vous pouvez décrire votre projet en français.",
  "advisor.useExample": "Use an example",
  "advisor.example":
    "We are an insurance company with about 20,000 PDF contracts and policy documents in French and English. Agents ask things like 'Which clauses cover water damage for policy X?' and 'Which partners are linked to claim Y?'. Answers should come back in under 5 seconds. Data must stay in the EU (GDPR).",
  "advisor.knownFacts": "Known facts (optional overrides)",
  "advisor.field.corpusSize": "Corpus size (documents)",
  "advisor.field.languages": "Languages (comma-separated ISO codes)",
  "advisor.field.latency": "Latency budget (ms)",
  "advisor.field.cost": "Cost sensitivity",
  "advisor.field.freshness": "Data freshness",
  "advisor.field.entities": "Entity richness",
  "advisor.field.compliance": "Residency / compliance (comma-separated)",
  "advisor.field.examples": "Example questions (one per line)",
  "advisor.eg": "e.g. {value}",
  "advisor.egCompliance": "e.g. EU only, GDPR, on-prem",
  "advisor.fromDescription": "From description",
  "advisor.level.low": "low",
  "advisor.level.medium": "medium",
  "advisor.level.high": "high",
  "advisor.freshness.static": "static",
  "advisor.freshness.monthly": "monthly",
  "advisor.freshness.weekly": "weekly",
  "advisor.freshness.daily": "daily",
  "advisor.freshness.realtime": "realtime",
  "advisor.size.tiny": "tiny",
  "advisor.size.small": "small",
  "advisor.size.medium": "medium",
  "advisor.size.large": "large",
  "advisor.size.xlarge": "very large",
  "advisor.analysing": "Analysing…",
  "advisor.recommend": "Recommend a strategy",
  "advisor.loading": "Reading your description and scoring strategies…",
  "advisor.profile": "Project profile",
  "advisor.profileSourceTitle": "How the profile was extracted",
  "advisor.sourceLlm": "extracted by Claude",
  "advisor.sourceHeuristic": "keyword heuristic (no model call)",
  "advisor.chip.corpus": "corpus: {size}",
  "advisor.chip.docs": " ({count} docs)",
  "advisor.chip.languages": "languages: {list}",
  "advisor.chip.latency": "latency: {value}",
  "advisor.notStated": "not stated",
  "advisor.chip.cost": "cost sensitivity: {value}",
  "advisor.chip.freshness": "freshness: {value}",
  "advisor.chip.entities": "entities: {value}",
  "advisor.mix":
    "Question mix: {single}% single-fact · {relational}% relational · {exploratory}% exploratory",
  "advisor.overridden": " · overridden: {fields}",
  "advisor.qtype.single_fact": "Single-fact",
  "advisor.qtype.relational_multi_hop": "Relational / multi-hop",
  "advisor.qtype.exploratory_multi_step": "Exploratory / multi-step",
  "advisor.hybrid": "Hybrid routing suggested",
  "advisor.notes": "Hosting, compliance and language notes",
  "advisor.nextStep": "Next step: ",
  "advisor.recommended": "recommended",
  "advisor.scoreAria": "{title} score",
  "advisor.why": "Why",
  "advisor.tradeoffs": "Tradeoffs",
  "advisor.suggestedConfig": "Suggested config",
  "advisor.on": "on",
  "advisor.off": "off",
  "advisor.relative": "relative cost ×{cost} · latency ×{latency}",
  "advisor.validate": "Validate on my questions",
  "advisor.validateHelp":
    "Runs against the documents currently ingested. One question per line; add {syntax} to also score answer quality. Up to {max} questions.",
  "advisor.validateSyntax": "question || ideal answer",
  "advisor.validatePlaceholder":
    "Which clauses cover water damage? || Clause 4.2 and annex B\nWho is the broker for policy 123?",
  "advisor.tooMany": "Only the first {max} of {count} questions will be sent.",
  "advisor.runQuestions_one": "Run {count} question",
  "advisor.runQuestions_other": "Run {count} questions",
  "advisor.validating":
    "Running {questions} × {strategies} strategy runs sequentially; this can take a few minutes.",
  "advisor.measuredWinner": "Measured winner: ",
  "advisor.noWinner": "No measured winner.",
  "advisor.col.strategy": "Strategy",
  "advisor.col.refusals": "Refusals",
  "advisor.col.errors": "Errors",
  "advisor.col.latency": "Avg latency",
  "advisor.col.tokens": "Tokens / q",
  "advisor.col.judge": "Judge",
  "advisor.col.f1": "F1 vs ideal",
  "advisor.perQuestion": "Per-question answers ({count})",
  "advisor.refused": "refused",
} as const;

export type MessageKey = keyof typeof en;
export type Dictionary = Record<MessageKey, string>;
