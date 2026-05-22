# LEARN — Three flavors of RAG, side by side

> A practical, file-by-file tour of how this project does **Classic RAG**, **Graph RAG**,
> and **Agentic RAG** — all powered by the **Anthropic API** (Claude).

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

The three flavors differ only in **step 1 — how we find the right pieces**:

| Flavor      | Retrieval = "find by…"           | Best at                          | Worst at                |
|-------------|----------------------------------|----------------------------------|-------------------------|
| **Classic** | Semantic similarity              | Single-fact lookups, FAQ         | Multi-hop, synthesis    |
| **Graph**   | Entity → relation walks          | "How is X connected to Y?"       | Vague questions         |
| **Agentic** | The LLM decides, step by step    | Multi-step, ambiguous queries    | Cheap simple lookups    |

---

## 1. The shared backbone (used by all three)

These pieces are the *plumbing*. Whichever strategy you pick, this code runs.

```
data/ ─► ingest ─► chunk ─► embed ─► Qdrant (vector DB)
                                          │
                                          └──► retrieved at query time
```

| Piece              | File                                                | What it does |
|--------------------|-----------------------------------------------------|--------------|
| **Chunking**       | `backend/app/rag/ingest.py:chunk_text`              | Splits documents into ~600-token windows with 80-token overlap. Smaller chunks = sharper retrieval; overlap stops sentences being chopped mid-thought. |
| **Embeddings**     | `backend/app/rag/embed.py`                          | Uses `BAAI/bge-small-en-v1.5` (a free local model) to turn text into a 384-dim vector. Nearby vectors = similar meaning. |
| **Vector store**   | `backend/app/rag/store.py`                          | Qdrant (a vector DB). One vector per chunk + the original text as payload. |
| **Vector search**  | `backend/app/rag/retrieve.py:hybrid_search`         | Embed the question → return top-50 chunks by cosine similarity. |
| **Rerank**         | `backend/app/rag/rerank.py:rerank`                  | Re-orders the top-50 down to top-8 with a more expensive model (currently a stub — swap in `bge-reranker-large`). |
| **Provider layer** | `backend/app/rag/providers/anthropic_provider.py`   | Three functions: `generate` (plain text), `generate_with_usage` (returns tokens too), and `tool_use_loop` (the agentic core). |

---

## 2. Strategy #1 — Classic RAG

**File:** `backend/app/rag/strategies/classic.py`

The textbook pipeline. The diagram:

```
question
   │
   ▼
embed_query  ──► vector_search (top-50)
                       │
                       ▼
                 rerank (top-8)
                       │
                       ▼
   prompt = system_rules + chunks + question
                       │
                       ▼
              Claude → answer + citations
```

Read it side-by-side with the code:

```python
candidates = hybrid_search(question, top_k=50)   # 50 nearest neighbors by meaning
reranked   = rerank(question, candidates, top_k=8)  # rerank to top 8
context    = compress_context(question, reranked)   # (stub) optional compression

ctx_block = "\n\n".join(f"[{c.id}]\n{c.text}" for c in context)
user_msg  = prompt.template.replace("{question}", question).replace("{context}", ctx_block)

out = await generate_with_usage(model=model, prompt=user_msg, max_tokens=1024)
```

**Why each step is there**
- *Why top-50 then rerank to top-8?* Recall vs precision. Embeddings are great at recall
  (don't miss anything) but mediocre at precision. A cross-encoder reranker reads
  (query, chunk) pairs together and is far more precise — but quadratically more expensive,
  so we only let it look at 50, not the whole corpus.
- *Why include `[chunk_id]` markers in the prompt?* So the model can cite a specific
  chunk by id, making the answer auditable.
- *Why a system prompt with strict rules?* See `backend/app/prompts/default.md` —
  forbids outside knowledge, requires citations, hardens against prompt-injection in
  retrieved content.

**Tradeoffs**

| ✅                                    | ❌                                        |
|--------------------------------------|------------------------------------------|
| Fast (one Claude call)               | Multi-hop reasoning falls apart          |
| Predictable cost (fixed token budget) | Vague questions retrieve junk            |
| Easy to debug                         | Synthesis across many docs is weak       |

---

## 3. Strategy #2 — Graph RAG

**Files:**
- `backend/app/rag/graph_store.py` — the on-disk graph (a JSONL file + in-memory index)
- `backend/app/rag/graph_extract.py` — Claude Haiku turns chunks into triples
- `backend/app/rag/strategies/graph.py` — the query-time strategy
- `backend/app/api/graph.py` — HTTP routes (`/graph/build`, `/graph/stats`, …)

### The intuition

Classic RAG asks: *"which chunks are similar to this question?"*

Graph RAG also asks: *"which chunks talk about the **same entities** the question does,
and what entities are **linked** to those?"*

That's a fundamentally different retrieval signal. If the question is *"how does BM25
relate to dense retrieval?"*, the chunks that mention BM25 and the chunks that mention
dense retrieval may live in different parts of your corpus. Graph RAG finds both, plus
any chunk that talks about how they connect.

### How the graph gets built

Once per document upload (or via `POST /graph/build`):

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

Open `data/graph/triples.jsonl` after building — it's plain JSONL, designed to be human-readable.

### How a question gets answered

```python
# 1. Ask Claude Haiku: what entities appear in this question?
entities = await extract_question_entities(question)

# 2. For each entity, walk 1 hop in the graph
graph_chunks = set()
for e in entities:
    chunks, neighbors = graph_store.neighbors(e, hops=1)
    graph_chunks |= chunks

# 3. Also do a regular vector search — entities aren't everything
vector_chunks = hybrid_search(question)

# 4. Union, rerank, build a context that *also* includes a text rendering of the subgraph
augmented_ctx = (
    "# Knowledge graph (extracted from your documents)\n"
    + describe_subgraph(entities)
    + "\n\n# Relevant passages\n"
    + chunks_as_text
)

# 5. Send to Claude as normal
answer = await claude.generate(augmented_ctx + question)
```

The key idea is **dual-channel context**: the LLM sees both raw passages *and* the
structured relationships extracted from those passages. That nudges it toward reasoning
about *connections*, not just summarizing nearby text.

### Tradeoffs

| ✅                                                          | ❌                                                       |
|------------------------------------------------------------|---------------------------------------------------------|
| Multi-hop queries get vastly better recall                  | Build step costs Claude calls upfront (~1¢ per chunk)   |
| Subgraph in the prompt gives the model a "map"              | Entity extraction is noisy on unusual domains           |
| Inspectable: open `triples.jsonl` to debug retrieval        | Doesn't help with single-fact lookups (overkill)        |

---

## 4. Strategy #3 — Agentic RAG

**File:** `backend/app/rag/strategies/agentic.py`
**Engine:** `backend/app/rag/providers/anthropic_provider.py:tool_use_loop`

### The intuition

Classic and Graph RAG do retrieval **once**, before the LLM speaks. Agentic RAG hands
the steering wheel to Claude. The model gets a **toolbox** and decides — turn by turn —
what to search, what to read in full, when it has enough, and how to phrase the answer.

This is built on **Anthropic tool use** (a.k.a. function calling). Claude returns
either text *or* a structured `tool_use` block; you execute the tool; you pass the
result back; the loop repeats until Claude says it's done.

### The tools we give the model

```python
[
  search(query, top_k)           # vector search, returns previews
  fetch_chunk(chunk_id)          # full text of one chunk
  finish(answer, citations,      # ends the loop with the final answer
         refusal=False)
]
```

### What a run looks like

```
user → "How is BM25 different from dense retrieval, and when should I use each?"

Claude (step 1):  tool_use search(query="BM25 dense retrieval comparison", top_k=5)
  → previews of 5 chunks

Claude (step 2):  tool_use fetch_chunk("doc4:12")   # this one looks juicy, read it all
  → full text

Claude (step 3):  tool_use search(query="when to use BM25 sparse retrieval", top_k=4)
  → previews

Claude (step 4):  tool_use finish(
                    answer="BM25 is a sparse, term-frequency-based scorer…",
                    citations=["doc4:12","doc2:3"])
  → loop ends, answer returned
```

The orchestration loop is `tool_use_loop` in `anthropic_provider.py`. It:
- sends the conversation + tool schemas to Claude,
- when Claude calls a tool, runs the local handler,
- packages the result into a `tool_result` message,
- loops until `stop_reason != "tool_use"` or the iteration cap (`settings.agentic_max_iters = 6`) is reached.

Every step is recorded in a `trace` array → that's what powers the "Reasoning trace"
disclosure in the Compare UI.

### Tradeoffs

| ✅                                                | ❌                                                |
|--------------------------------------------------|--------------------------------------------------|
| Best for multi-step / ambiguous questions         | 3–6× the tokens of Classic for trivial questions |
| Adapts retrieval on the fly                       | Higher latency (multiple round-trips to Claude)  |
| Catches retrieval misses by re-querying           | Can spin in circles on sparse corpora            |
| Naturally produces "show your work" traces        | Harder to reproduce — same question may take different paths |

---

## 5. How `/compare` ties it together

**File:** `backend/app/api/compare.py:compare_strategies`

The new endpoint is `POST /compare/strategies`. It fans out:

```python
async def _one(name):
    return await run_strategy_raw(question, strategy=name)

results = await asyncio.gather(*[_one(s) for s in ["classic", "graph", "agentic"]])
```

So all three strategies run in parallel. The response contains, for each strategy:
- `answer`, `sources`, `refusal`, `confidence` — what the user asked for
- `latency_ms`, `input_tokens`, `output_tokens`, `iterations` — telemetry
- `trace` — the per-strategy reasoning steps
- `extra` — strategy-specific debug info (entities + subgraph for Graph; explored chunks for Agentic)

The frontend (`frontend/src/pages/Compare.tsx`) renders these as three side-by-side cards
with a "Winners" bar at the bottom that tags the fastest and cheapest result.

---

## 6. The Anthropic API surface, demystified

You'll see three patterns in this codebase:

```python
# 1. PLAIN COMPLETION  (Classic, Graph — final generation step)
client.messages.create(model=..., max_tokens=..., messages=[{"role":"user","content":...}])

# 2. COMPLETION + USAGE  (so we can report cost in the Compare UI)
resp.usage.input_tokens, resp.usage.output_tokens

# 3. TOOL USE  (Agentic — Claude decides what to do next)
client.messages.create(..., tools=[{name, description, input_schema}, ...])
# resp.content may include `tool_use` blocks; you reply with `tool_result` blocks
# and call again. Loop until stop_reason != "tool_use".
```

All three patterns are in `backend/app/rag/providers/anthropic_provider.py` — read that
file once and you've seen 90% of what real production code does with Claude.

---

## 7. Running it locally

### 7.1 Put your Anthropic key in `.env`

```bash
# .env  (at the project root)
ANTHROPIC_API_KEY=sk-ant-...
```

The system also keeps OpenAI as a backup provider for Classic, but **Graph and Agentic
require Anthropic** (they use Claude Haiku for cheap extraction + Claude's tool-use API).

### 7.2 Start the stack

```bash
cd infra
docker compose up -d
```

Frontend: http://localhost:5173 · Backend: http://localhost:8011/health

### 7.3 The flow

1. **Ingest** — upload a few `.md` or `.pdf` files on the Ingest page.
2. **Build the graph** — open Compare, click **Build graph**. This loops through the
   chunks Qdrant just stored and asks Claude Haiku to extract triples. Open
   `data/graph/triples.jsonl` to see what came out.
3. **Ask** — go to Compare, type a question, hit *Compare 3 strategies*. Three cards
   appear. Click *Reasoning trace* on each to see how each one found its evidence.
4. **Tinker** — try the same question with each strategy on the Ask page and compare
   the answers (and the citations).

### 7.4 Good demo questions

- *"Compare BM25 and dense retrieval."* → expect Graph to shine (multi-entity relation).
- *"What is the chunk size used in this project?"* → Classic wins (single-fact lookup).
- *"Why might my eval scores regress after a prompt change?"* → Agentic often wins
  (needs synthesis + a follow-up query).

---

## 8. A reading order for the code

If you want to *really* understand it, open these in this order — each builds on the last:

1. `backend/app/config.py` — what's tunable
2. `backend/app/schemas/__init__.py` — request/response shapes
3. `backend/app/rag/embed.py`, `store.py`, `retrieve.py` — vector layer
4. `backend/app/prompts/default.md` — the grounded-answer contract
5. `backend/app/rag/providers/anthropic_provider.py` — every way we talk to Claude
6. `backend/app/rag/strategies/classic.py` — the baseline
7. `backend/app/rag/graph_store.py` → `graph_extract.py` → `strategies/graph.py`
8. `backend/app/rag/strategies/agentic.py`
9. `backend/app/rag/generate.py` — the dispatcher
10. `backend/app/api/compare.py` — the 3-way fan-out
11. `frontend/src/pages/Compare.tsx` — the UI that makes the comparison legible

---

## 9. What to put on LinkedIn / GitHub

This project demonstrates, on one codebase, three different production-grade approaches
to RAG, all running against the same indexed corpus, all using the Anthropic API:

- **Classic RAG** with hybrid retrieval + cross-encoder reranking
- **Graph RAG** with LLM-driven triple extraction (Claude Haiku) and entity-walk retrieval
- **Agentic RAG** built on Anthropic tool use (search / fetch / finish loop)
- A **3-way compare UI** with latency, token cost, and reasoning traces side by side
- A built-in **evaluation harness** (faithfulness, answer relevance, context precision/recall)
  so improvements can be measured, not asserted.

That last point matters — most "I built a RAG" projects skip evaluation. The whole reason
EvalRAG exists is that *a RAG system is only as good as the eval that catches it drifting.*

---

## 10. Glossary (skim this if a term threw you off)

- **Chunk** — a small slice of a document, typically ~600 tokens.
- **Embedding** — a vector of floats that represents the meaning of a chunk.
- **Vector DB (Qdrant)** — stores embeddings and finds nearest neighbors fast.
- **Hybrid retrieval** — combine vector (semantic) + keyword (BM25) scores. Our current
  retrieve.py is dense-only; the name is aspirational and a great next exercise.
- **Reranker** — a model that scores (query, chunk) pairs together. More accurate than
  embeddings, more expensive — used on the shortlist only.
- **Triple** — `(subject, predicate, object)`, the atomic unit of a knowledge graph.
- **Tool use / function calling** — model returns a structured request to run a function;
  you run it and pass the result back. The basis of "agentic" behavior.
- **Faithfulness** — does the answer say only things actually supported by the cited chunks?
- **Refusal** — when the system says "I don't know from the corpus" instead of hallucinating.
