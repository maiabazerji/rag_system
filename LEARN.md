# LEARN: Three flavors of RAG, side by side

> A practical, file-by-file tour of how this project does **Classic RAG**, **Graph RAG**,
> and **Agentic RAG**, all powered by the **Anthropic API** (Claude).

This is the document I wish existed when I first learned RAG. Every concept is mapped
to the exact file and function that implements it. Read it top-to-bottom or jump to
the strategy you're curious about.

---

## 0. The 60-second mental model

A **Retrieval-Augmented Generation** system answers a question by:

1. **Finding** the right pieces of your private documents → *Retrieval*
2. **Stuffing** those pieces into an LLM prompt → *Augmentation*
3. **Asking** the LLM to answer, citing what it used → *Generation*

The LLM (Claude) never trained on your documents. But by handing it the right
passages at runtime, you get the model's reasoning skill **grounded** in your data.

The three flavors differ mainly in **step 1, how we find the right pieces**. Steps 2 and 3
(context with citation handles, grounded generation, citation validation) are shared.

| Flavor      | Retrieval = "find by…"                          | Designed for                     | Expected weak spot      |
|-------------|-------------------------------------------------|----------------------------------|-------------------------|
| **Classic** | Meaning (dense) + keywords (BM25), fused         | Single-fact lookups, FAQ         | Multi-hop, synthesis    |
| **Graph**   | Entity → relation walks, plus the same hybrid search | "How is X connected to Y?"   | Vague questions         |
| **Agentic** | The LLM decides, step by step                    | Multi-step, ambiguous queries    | Cheap simple lookups    |

The last two columns are design intent, not measurements. On the current pipeline no
strategy comparison has been measured yet; the README's "Current benchmark" section says
what has been.

---

## 1. The shared backbone (used by all three)

These pieces are the *plumbing*. Whichever strategy you pick, this code runs.

```
data/ ─► ingest ─► chunk ─► embed ─► Qdrant (vector DB) ─► BM25 index (built from the payloads)
                                          │                      │
                                          └──── both retrieved at query time, fused with RRF
```

| Piece              | File                                                | What it does |
|--------------------|-----------------------------------------------------|--------------|
| **Parsing**        | `backend/app/rag/parsers/`                          | One module per format (PDF with optional OCR, docx, pptx, odt/ods, eml, HTML, text, CSV), a registry that picks one by extension, MIME type or file signature, and the one list of supported types. Output is markdown-ish text plus file metadata. |
| **Chunking**       | `backend/app/rag/chunking.py:chunk_structured`      | Cuts at headings (markdown and French legal divisions) first, then packs whole paragraphs into `CHUNK_SIZE_TOKENS`-word pieces (default 300, capped to fit the 512-token embedder), splitting an oversized paragraph only between sentences, with `CHUNK_OVERLAP_TOKENS` of whole trailing sentences as overlap (default 50), never inside a table row. Each chunk records its `heading_path`, pages and character offsets. `CHUNK_STRATEGY=fixed` keeps the legacy word windows for comparison. |
| **Embeddings**     | `backend/app/rag/embed.py`                          | Uses `intfloat/multilingual-e5-small` (a free local multilingual model) to turn text into a 384-dim vector, adding the `query: ` / `passage: ` markers E5 expects. Nearby vectors = similar meaning, across languages. |
| **Vector store**   | `backend/app/rag/store.py`                          | Qdrant (a vector DB). One vector per chunk + the original text as payload, with tenant and group fields for access control. |
| **Dense search**   | `backend/app/rag/retrieve.py:dense_search`          | Embed the question → return the top `DENSE_TOP_K` chunks by cosine similarity. |
| **Sparse search**  | `backend/app/rag/sparse.py:sparse_search`           | BM25 over every chunk the caller may read, in memory, one model per access scope → top `BM25_TOP_K`. Catches exact terms (codes, names) that embeddings blur. |
| **Fusion**         | `backend/app/rag/fusion.py`                         | Reciprocal Rank Fusion: `score = Σ 1 / (RRF_K + rank)`. Uses ranks, because cosine and BM25 scores are on different scales. |
| **Hybrid search**  | `backend/app/rag/retrieve.py:hybrid_search`         | Runs the pipeline: preprocess → dense + BM25 (concurrently) → RRF → rerank → `FINAL_CONTEXT_K` chunks, with per-stage diagnostics. `RETRIEVAL_MODE` picks `dense`, `sparse` or `hybrid` (default). |
| **Rerank**         | `backend/app/rag/rerank.py:rerank_scored_async`     | Re-scores the candidates down to `RERANK_TOP_K` with a cross-encoder (`RERANKER_MODEL`), which reads the question and chunk together instead of comparing two independent vectors. Falls back to BM25, then to plain truncation, if the model cannot load; the fallback is reported as `reranker: "bm25-fallback"`. |
| **Grounding**      | `backend/app/rag/grounding.py`                      | Citation handles, the `submit_answer` answer contract, citation validation and the evidence score. See "Grounded answers" below. |
| **Provider layer** | `backend/app/rag/providers/base.py` → `anthropic_provider.py`, `openai_provider.py`, `local.py` | A dispatcher over three providers. The Anthropic module has `generate_with_usage`, `generate_structured` (forced tool call) and `tool_use_loop` (the agentic core). |

---

## 2. Strategy #1: Classic RAG

**File:** `backend/app/rag/strategies/classic.py`

The textbook pipeline. The diagram:

```
question
   │
   ▼
dense (top-50) + BM25 (top-50) ──► RRF fusion
                       │
                       ▼
            cross-encoder rerank (top-8)
                       │
                       ▼
   system = grounding rules; user = [S1..Sn] sources + question
                       │
                       ▼
       model → submit_answer(answer, claims, status)
                       │
                       ▼
   validate citations → cited sources, grounded, evidence score
```

Read it side-by-side with the code (simplified from `classic.py`):

```python
retrieval = await hybrid_search(question, access, final_k=top_k)   # dense + BM25 → RRF → rerank
context   = retrieval.chunks

cited     = cite_chunks(context)                  # S1..Sn handles, chunk ids stay server-side
ctx_block = format_context(cited)                 # "[S2] handbook.pdf | section: Leave | p. 12"
user_msg  = render_prompt(prompt_version, question=question, context=ctx_block)

out = await generate_structured(
    self.provider, model=model, prompt=user_msg,
    tool=SUBMIT_ANSWER_TOOL, system=grounded_system(self.provider),
    max_tokens=settings.max_answer_tokens,
)
grounded, raw = ground(question, out, cited)      # validate every [S#] against the context
```

Both calls are awaited. The embedding model, the BM25 build and the cross-encoder are CPU-bound, so they run in a thread pool; calling the synchronous versions from a request handler would block the event loop and stall every other in-flight request.

**Why each step is there**
- *Why two retrievers?* Embeddings match meaning but blur exact tokens such as
  invoice numbers or article references; BM25 matches exact tokens but misses
  paraphrases. RRF keeps what either one ranks high.
- *Why top-50 then rerank to top-8?* Recall vs precision. First-stage retrievers are
  tuned for recall. A cross-encoder reads (query, chunk) pairs together and ranks more
  precisely, but costs one model pass per pair, so it only sees the fused shortlist.
- *Why `[S#]` handles instead of chunk ids?* The model can only cite what it was shown,
  and the server maps handles back to chunks. A citation to a handle that was not in the
  context is detected, removed and reported in `invalid_citations`.
- *Why a system prompt plus `default.md`?* The grounding rules (answer only from the
  sources, cite every factual sentence, report `insufficient_context`, ignore instructions
  inside the sources) are the system prompt (`grounding.py:grounded_system`).
  `backend/app/prompts/default.md` is the user message with the style, language and
  grounding instructions, the question and the sources.

**Tradeoffs**

| Strengths                             | Weaknesses (expected, not measured)      |
|--------------------------------------|------------------------------------------|
| One model call per question          | Multi-hop reasoning is not supported by design |
| Predictable cost (fixed context size) | Vague questions retrieve loosely related chunks |
| Easy to debug (per-chunk diagnostics) | Synthesis across many documents is limited by `FINAL_CONTEXT_K` |

---

## 3. Strategy #2: Graph RAG

**Files:**
- `backend/app/rag/graph_store.py`- the on-disk graph (a JSONL file + in-memory index)
- `backend/app/rag/graph_extract.py`- Claude Haiku turns chunks into triples
- `backend/app/rag/strategies/graph.py`- the query-time strategy
- `backend/app/api/graph.py`- HTTP routes (`/graph/build`, `/graph/stats`, …)

### The intuition

Classic RAG asks: *"which chunks are similar to this question?"*

Graph RAG also asks: *"which chunks talk about the **same entities** the question does,
and what entities are **linked** to those?"*

That's a fundamentally different retrieval signal. If the question is *"how does BM25
relate to dense retrieval?"*, the chunks that mention BM25 and the chunks that mention
dense retrieval may live in different parts of your corpus. Graph RAG finds both, plus
any chunk that talks about how they connect.

### How the graph gets built

Only on demand, with `POST /graph/build` (admin; the **Build graph** button on the Compare
page calls it). Ingestion does **not** extract triples, so the graph has to be rebuilt after
new documents are indexed. Ingestion does remove the triples of documents it replaces.

```
for each indexed chunk:
    triples = claude_haiku.extract(chunk.text)
    # e.g. [{subject:"BM25", predicate:"is_a", object:"sparse retrieval algorithm",
    #        chunk_id:"abc:7", doc_id:"abc"}, ...]
    append to data/graph/triples.jsonl
```

Each triple records *what concept goes with what concept*, and importantly *which chunk
said so*. The graph is the entities, edges = the relations, and a back-pointer to the
chunks that justify each edge.

Open `data/graph/triples.jsonl` after building, it's plain JSONL, designed to be human-readable.

### How a question gets answered

```python
# 1. Ask Claude Haiku: what entities appear in this question?
entities = await extract_question_entities(question)

# 2. For each entity, walk 1 hop in the graph
graph_chunks = set()
for e in entities:
    chunks, neighbors = graph_store.neighbors(e, hops=1)
    graph_chunks |= chunks

# 3. Also run the hybrid search (dense + BM25, fused, not yet reranked)
retrieval = await hybrid_search(question, access, final_k=settings.dense_top_k, rerank=False)

# 4. Union with the graph-only chunks, rerank the whole pool, and build a context that
#    *also* includes a text rendering of the subgraph
ranked = await rerank_async(question, retrieval.chunks + graph_only_chunks, top_k=top_k)
augmented_ctx = (
    "# Knowledge graph (extracted from your documents)\n"
    + describe_subgraph(entities)
    + "\n\n# Relevant passages\n"
    + format_context(cite_chunks(ranked))
)

# 5. Grounded generation and citation validation, exactly as in Classic
```

The key idea is **dual-channel context**: the LLM sees both raw passages *and* the
structured relationships extracted from those passages. That nudges it toward reasoning
about *connections*, not just summarizing nearby text.

### Tradeoffs

| Strengths                                                  | Weaknesses                                               |
|------------------------------------------------------------|---------------------------------------------------------|
| Can reach chunks that share an entity but not vocabulary    | Build step costs one extraction call per chunk          |
| Subgraph in the prompt gives the model a "map"              | Entities are linked by exact (normalised) string match  |
| Inspectable: open `triples.jsonl` to debug retrieval        | One extra model call per question (entity extraction)   |

Whether the graph improves multi-hop recall on this corpus has not been measured on the
current pipeline. In the historical baseline (previous pipeline, see README), graph and
classic retrieval scores were identical on `golden_v1`.

---

## 4. Strategy #3: Agentic RAG

**File:** `backend/app/rag/strategies/agentic.py`
**Engine:** `backend/app/rag/providers/anthropic_provider.py:tool_use_loop`

### The intuition

Classic and Graph RAG do retrieval **once**, before the LLM speaks. Agentic RAG hands
the steering wheel to Claude. The model gets a **toolbox** and decides, turn by turn-
what to search, what to read in full, when it has enough, and how to phrase the answer.

This is built on **Anthropic tool use** (a.k.a. function calling). Claude returns
either text *or* a structured `tool_use` block; you execute the tool; you pass the
result back; the loop repeats until Claude says it's done.

### The tools we give the model

```python
[
  search(query, top_k<=12)       # hybrid search (dense + BM25, fused, not reranked);
                                 # returns handles (S3...), documents and 300-char previews
  fetch_chunk(source)            # full text of a source it has seen, by handle
  finish(answer, claims,         # ends the loop; same schema as submit_answer,
         status, unsupported_notes)  # validated against the sources it retrieved
]
```

### What a run looks like

```
user → "How is BM25 different from dense retrieval, and when should I use each?"

Claude (step 1):  tool_use search(query="BM25 dense retrieval comparison", top_k=5)
  → previews of 5 chunks

Claude (step 2):  tool_use fetch_chunk(source="S2")   # this preview is truncated, read it all
  → full text

Claude (step 3):  tool_use search(query="when to use BM25 sparse retrieval", top_k=4)
  → previews S6..S9

Claude (step 4):  tool_use finish(
                    answer="BM25 is a sparse, term-frequency-based scorer [S2]…",
                    claims=[{"text": "...", "citations": ["S2"], "supported": true}, ...],
                    status="answered", unsupported_notes="")
  → loop ends; citations are validated against S1..S9
```

The orchestration loop is `tool_use_loop` in `anthropic_provider.py`. It:
- sends the conversation + tool schemas to Claude,
- when Claude calls a tool, runs the local handler,
- packages the result into a `tool_result` message,
- loops until `stop_reason != "tool_use"` or the iteration cap (`AGENTIC_MAX_ITERS`, default 15) is reached.

Every step is recorded in a `trace` array → that's what powers the "Reasoning trace"
disclosure in the Compare UI.

### Tradeoffs

| Strengths                                         | Weaknesses                                        |
|--------------------------------------------------|--------------------------------------------------|
| Designed for multi-step / ambiguous questions     | Several model calls per question, so more tokens than Classic (not measured on the current pipeline) |
| Adapts retrieval on the fly                       | Higher latency (multiple round-trips to Claude)  |
| Can recover from a retrieval miss by re-querying  | Can spin in circles on sparse corpora            |
| Naturally produces "show your work" traces        | Harder to reproduce, same question may take different paths |

---

## 5. How `/compare` ties it together

**File:** `backend/app/api/compare.py:compare_strategies`

The endpoint is `POST /compare/strategies`. It runs the requested strategies **one after
another** (the agentic strategy makes many model calls, and running all three at once trips
provider rate limits):

```python
for name in req.strategies:
    result, err = await run_strategy_raw(req.question, strategy=name, model=req.model,
                                         access=principal.scope())
```

With `evaluate=true`, each row is also scored by the judge and, when the relevant documents
are known (from a reference or a golden example), by the ranked retrieval metrics. The
response contains, for each strategy:
- `answer`, `sources`, `refusal`, `confidence`- what the user asked for
- `grounded`, `status`, `claims`, `invalid_citations`, `citation_count`- how well the
  answer is backed by the retrieved chunks (see "Grounded answers" below)
- `latency_ms`, `input_tokens`, `output_tokens`, `iterations`- telemetry
- `retrieval`- retrieval diagnostics (mode, reranker used, per-stage counts and latency, per-chunk ranks and scores)
- `metrics`- tokens, estimated cost, model calls and per-stage latency
- `trace`- the per-strategy reasoning steps
- `extra`- strategy-specific debug info (entities + subgraph for Graph; explored chunks for Agentic)

The frontend (`frontend/src/pages/Compare.tsx`) renders these as side-by-side cards, with a
summary table that marks the best measured value per metric (ties included, failed rows
ignored) rather than naming an overall winner.

### Grounded answers

All three strategies end in `app/rag/grounding.py`. Each context chunk is shown to the
model under a handle (`[S1]`, `[S2]`...; the agent gets handles from its `search` tool),
and the model answers through a `submit_answer` tool (the agent's `finish`): `answer`
with inline `[S#]` markers, `claims` (each with citations and `supported`), `status`
(`answered` | `partial` | `insufficient_context`) and `unsupported_notes`. The server then
validates every citation against the chunks the model was actually given: unknown handles
are dropped from the text and listed in `invalid_citations`, `sources` becomes the cited
chunks only (the full context stays in `extra.context_sources`), and an answer with no
surviving citation, or with status `insufficient_context`, becomes the localized "cannot
answer from the retrieved documents" refusal.

`confidence` is an **evidence score**, not a calibrated probability:
`0.5 * coverage + 0.2 * status + 0.3 * relevance`, where coverage is the share of supported
claims citing a valid source, status is 1 for `answered` and 0.5 for `partial`, and
relevance is the mean cited-chunk retrieval score mapped into [0, 1] (the first two terms
are rescaled to fill [0, 1] when no score is available). Invalid citations scale it down by
up to half; a refusal scores 0.

---

## 6. The Anthropic API surface, demystified

You'll see three patterns in this codebase:

```python
# 1. COMPLETION + USAGE  (graph extraction, the judge; token counts feed the cost estimate)
resp = client.messages.create(model=..., max_tokens=..., messages=[...])
resp.usage.input_tokens, resp.usage.output_tokens

# 2. STRUCTURED TOOL CALL  (Classic, Graph: the final grounded answer)
client.messages.create(..., tools=[SUBMIT_ANSWER_TOOL], **tool_request_options(model, "submit_answer"))
# forced tool_choice on models that accept it; newer models get "auto" and the system
# prompt names the tool. The tool input *is* the structured answer: answer, claims,
# status, unsupported_notes

# 3. TOOL-USE LOOP  (Agentic: Claude decides what to do next)
client.messages.create(..., tools=[search, fetch_chunk, finish])
# resp.content may include `tool_use` blocks; you reply with `tool_result` blocks
# and call again. Loop until stop_reason != "tool_use".
```

All three patterns are in `backend/app/rag/providers/anthropic_provider.py`.

---

## 7. Running it locally

### 7.1 Put your Anthropic key in `.env`

```bash
# .env  (at the project root)
ANTHROPIC_API_KEY=sk-ant-...
```

Without it the backend refuses to start (`GENERATOR_PROVIDER is 'anthropic' but
ANTHROPIC_API_KEY is not set`). Classic can also run on OpenAI or on a local Ollama model
(`GENERATOR_PROVIDER=openai` or `local`), but **Graph, Agentic and the judge require
Anthropic** (Claude Haiku for extraction, Claude's tool-use API for the agent).

### 7.2 Start the stack

```bash
# from the project root; POSTGRES_PASSWORD must be set in .env
docker compose --env-file .env -f infra/docker-compose.yml up -d
```

Ports are published on 127.0.0.1 only. The embedding model and the cross-encoder
download on first use into `.hf_cache/`; `python scripts/download_models.py
--cache-dir .hf_cache` fetches them up front.

Frontend: http://localhost:5173 · Backend: http://localhost:8011/health

### 7.3 The flow

1. **Ingest**: upload a few `.md` or `.pdf` files on the Upload page (`/ingest`).
2. **Build the graph**: open Compare, click **Build graph**. This loops through the
   chunks Qdrant just stored and asks Claude Haiku to extract triples. Open
   `data/graph/triples.jsonl` to see what came out.
3. **Ask**: go to Compare, type a question (or pick one from a golden dataset), and run
   the comparison. One card per strategy appears, with its cited sources, grounding status,
   retrieval diagnostics, latency breakdown and cost. The reasoning trace shows how each
   one found its evidence.
4. **Tinker**: try the same question with each strategy on the Ask page and compare
   the answers (and the citations).

### 7.4 Good demo questions

These exercise different strategies by design; which one does better on them is exactly
what the Compare page measures, so no outcome is promised here.

- *"Compare BM25 and dense retrieval."* (multi-entity relation, the case Graph is built for)
- *"What is the chunk size used in this project?"* (single-fact lookup)
- *"Why might my eval scores regress after a prompt change?"* (synthesis, may need a
  follow-up query)

---

## 8. A reading order for the code

If you want to *really* understand it, open these in this order, each builds on the last:

1. `backend/app/config.py`- what's tunable
2. `backend/app/schemas/__init__.py`- request/response shapes
3. `backend/app/rag/embed.py`, `store.py`, `sparse.py`, `fusion.py`, `retrieve.py`- retrieval
4. `backend/app/rag/grounding.py` and `backend/app/prompts/default.md`- the grounded-answer contract
5. `backend/app/rag/providers/anthropic_provider.py`- every way we talk to Claude
6. `backend/app/rag/strategies/classic.py`- the baseline
7. `backend/app/rag/graph_store.py` → `graph_extract.py` → `strategies/graph.py`
8. `backend/app/rag/strategies/agentic.py`
9. `backend/app/rag/generate.py`- the dispatcher
10. `backend/app/api/compare.py`- the 3-way fan-out
11. `frontend/src/pages/Compare.tsx`- the UI that makes the comparison legible

---

## 9. What the project demonstrates

On one codebase, three approaches to RAG run against the same indexed corpus:

- **Classic RAG** with hybrid retrieval (dense + BM25, RRF) and cross-encoder reranking
- **Graph RAG** with LLM-driven triple extraction (Claude Haiku) and entity-walk retrieval
- **Agentic RAG** built on Anthropic tool use (search / fetch / finish loop)
- **Grounded answers** with server-side citation validation for all three
- A **compare UI** with cited sources, retrieval diagnostics, latency, cost and reasoning traces
- An **evaluation harness**: deterministic retrieval metrics at K, a versioned LLM judge,
  refusal scoring, failure accounting and a threshold-based regression gate

The measured results so far, and what remains unmeasured, are in the README
("Current benchmark") and in `docs/CASE_STUDY.md`.

---

## 10. Glossary (skim this if a term threw you off)

- **Chunk**: a small slice of a document, here at most 300 words by default (kept under the
  models' 512-token window).
- **Embedding**: a vector of floats that represents the meaning of a chunk.
- **Vector DB (Qdrant)**: stores embeddings and finds nearest neighbors fast.
- **Hybrid retrieval**: combine vector (semantic) and keyword (BM25) retrieval. Here the two
  ranked lists are fused with Reciprocal Rank Fusion (`retrieve.py:hybrid_search`), which
  is the default `RETRIEVAL_MODE`.
- **Reranker**: a model that scores (query, chunk) pairs together. More accurate than
  embeddings, more expensive, used on the shortlist only.
- **Triple**: `(subject, predicate, object)`, the atomic unit of a knowledge graph.
- **Tool use / function calling**: model returns a structured request to run a function;
  you run it and pass the result back. The basis of "agentic" behavior.
- **Faithfulness**: does the answer say only things supported by the retrieved context? The
  judge scores it against the full context the generator saw, not only the cited chunks.
- **Refusal**: when the system says "I don't know from the corpus" instead of hallucinating.
