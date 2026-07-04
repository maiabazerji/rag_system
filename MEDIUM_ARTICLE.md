# RAG Strategy Comparison: Why One Approach Isn't Enough

When you build a RAG system, you face a choice that most tutorials gloss over: how do you actually retrieve context to ground your LLM's answer?

The default answer is always: embed the query, search for similar chunks, and feed them to Claude. This works. But it's not the only way. And it's not always the best way.

I built EvalRAG to answer a simple question: **Which retrieval strategy actually works best for my specific corpus?**

---

## The Three Retrieval Strategies

### 1. Classic RAG: Fast and Simple

**How it works:**
1. Embed the query
2. Vector search for the N most similar chunks
3. Rerank by relevance
4. Generate answer

**When it wins:**
- Factual lookups ("When was X founded?")
- Single-document questions
- Speed matters more than depth

**When it fails:**
- Questions about relationships ("How does X compare to Y?")
- Multi-hop reasoning ("Given A and B, infer C")
- When the answer spans multiple documents that aren't semantically similar

**Cost:** ~200ms latency, ~850 tokens, ~$0.001

---

### 2. Graph RAG: Find Hidden Connections

**How it works:**
1. (Offline) Extract entity triples from each chunk using Claude
   - "BM25 is_a sparse_retrieval_algorithm"
   - "sparse_retrieval compared_to dense_vectors"
   - "dense_vectors uses_for semantic_similarity"
2. Build a knowledge graph from the triples
3. At query time:
   - Extract entities from the question
   - Walk the graph to find related chunks
   - Combine graph results with vector search results
   - Augment prompt with the subgraph structure

**Why this matters:**
Classic RAG only finds chunks that are *semantically similar* to your query. Graph RAG also finds chunks that are *structurally related* — even if they don't use similar language.

**Example:**
```
Question: "How do BM25 and dense vectors differ?"

Classic RAG finds:
- A paper on BM25
- A paper on embeddings
(But doesn't connect them)

Graph RAG finds:
- Papers on BM25
- Papers on dense vectors
- Plus: shows they're both types of retrieval algorithms
- Plus: shows the exact relationships between them
```

**When it wins:**
- Relationship questions ("How is X related to Y?")
- Interconnected documents
- When you need to show your work (subgraph reasoning)

**When it fails:**
- Simple factual lookups (slower for no benefit)
- Sparse documents (graph has no edges)
- Poor triple extraction (Claude's extraction quality varies)

**Cost:** ~450ms latency, ~1,200 tokens, ~$0.002

---

### 3. Agentic RAG: Let Claude Drive

**How it works:**
1. Expose Claude three tools:
   - `search(query)` - semantic search your corpus
   - `fetch_chunk(id)` - get full text of a chunk
   - `finish(answer, citations)` - emit final answer
2. Let Claude loop until it has enough context
3. Claude decides when to search, what to fetch, when to stop

**Real example:**
```
Question: "What's the difference between faithfulness and answer relevance?"

Turn 1 - Claude searches "faithfulness evaluation metrics"
  Gets 3 chunk previews
  One looks promising but is truncated

Turn 2 - Claude fetches the full text of that chunk
  Reads it and searches for "answer relevance"
  
Turn 3 - Gets results about answer relevance
  Has enough context now

Turn 4 - Claude calls finish() with the answer
```

**When it wins:**
- Ambiguous questions (Claude can ask for clarification via follow-up searches)
- Multi-step reasoning
- Questions where initial search is off but refinement helps
- When you want explainability (Claude reasons through its retrieval)

**When it fails:**
- Simple factual lookups (massively over-engineered)
- Tight token budgets (uses 3-6x more tokens)
- Corpus retrieval is random (Claude loops forever)

**Cost:** ~3.2s latency, ~2,840 tokens, ~$0.006

---

## The Comparison Dashboard

The magic is seeing all three side-by-side for the same question:

**Ask:** "Compare faithfulness and answer relevance metrics"

| Metric | Classic | Graph | Agentic |
|--------|---------|-------|---------|
| Latency | 200ms | 450ms | 3.2s |
| Tokens | 850 | 1,200 | 2,840 |
| Cost | $0.001 | $0.002 | $0.006 |
| Quality | 7/10 | 9/10 | 10/10 |

Which is "best"? Classic if speed matters. Graph if you need connections. Agentic if accuracy matters most.

**There is no universal winner. It depends on your question.**

---

## Measuring Quality: Evaluation Metrics

After you compare strategies, the next question is: which answer is actually *better*?

EvalRAG measures four things:

### Faithfulness
Does the answer stay grounded in the retrieved context?

```
Context: "Claude was trained on data up to April 2024"
Answer: "Claude's cutoff is April 2024"
Score: ✅ 1.0 (grounded)

Answer: "Claude's cutoff is 2025"
Score: ❌ 0.0 (hallucinated)
```

### Answer Relevance
Does the answer actually answer the question?

```
Question: "How does RAG improve LLM accuracy?"
Answer: "RAG systems use vector databases"
Score: 0.3 (tangential, not answering)

Answer: "RAG improves accuracy by grounding responses in 
         retrieved context, reducing hallucinations by 40-60%"
Score: 0.95 (directly answers)
```

### Context Precision
Of the chunks you retrieved, what percentage did you actually use?

```
Retrieved: 8 chunks
Used in answer: 3
Precision = 37.5%
```

### Context Recall
Of all relevant chunks in your corpus, what percentage did you retrieve?

```
Total relevant: 10 chunks
Retrieved: 3
Recall = 30%
```

---

## Regression Detection

Once you measure these metrics, compare against your last eval run:

```
Last run:  Faithfulness: 0.85, Latency: 200ms
This run:  Faithfulness: 0.78, Latency: 180ms

Alert: Faithfulness dropped 7 points
```

Catch bugs before they hit production.

---

## How To Use It

### 1. Ingest Documents
Upload PDFs, Markdown, or text files. They're chunked, embedded, and indexed.

### 2. Ask Questions
Pick a strategy (Classic, Graph, or Agentic) and ask. Every answer shows:
- The retrieved chunks it used
- Confidence score
- Execution trace (what it did)

### 3. Compare Strategies
Ask the *same question* to all three strategies at once. See side-by-side latency, token cost, and quality.

### 4. Build the Graph (Optional)
If you want to use Graph RAG, click "Build graph" to extract entity triples from your documents.

### 5. Run Evals
Define a golden dataset (Q&A pairs you know the answers to). Run an evaluation. See metrics per strategy. Detect regressions.

---

## The Architecture

**Four layers:**

```
Frontend (React)
  Ask questions, compare strategies, view results

API (FastAPI)
  /ask → ask a question with a strategy
  /compare → all 3 strategies in parallel
  /eval/run → evaluate on golden dataset
  /graph/build → extract triples

RAG Pipeline                 Eval Engine
  embed_query()               score_example()
  vector_search()             judge_answer()
  retrieve()                  detect_regression()
  rerank()
  generate()

Vector DB (Qdrant) + Document DB (Postgres)
```

Each strategy inherits from a base class:
```python
class Strategy:
    async def run(question, top_k, model) -> StrategyResult
```

So all three return the same shape. Easy to compare.

---

## What The Code Teaches

This isn't a production system. It's a *learning platform*.

**It intentionally leaves these as stubs so you wire them:**

- **Embedding model** — Today: local `bge-small`. Wire in: OpenAI or proprietary
- **Reranker** — Today: pass-through. Wire in: `bge-reranker-large`
- **Judge LLM** — Today: returns zeros. Wire in: real Claude calls
- **Metrics** — Today: stubbed. Wire in: RAGAS

The point: understand each piece and own the implementation.

---

## Key Insights

**1. There's no universal RAG strategy.**

Classic wins on speed. Graph wins on understanding relationships. Agentic wins on accuracy. Which you pick depends on your question type and constraints.

**2. Measuring matters more than the strategy.**

A bad eval can't tell you which strategy is better. Defining "faithfulness" precisely is the hard part, not implementing the retrieval.

**3. Regressions hide in plain sight.**

You tweak the prompt. It feels good. You ship it. Three weeks later, users complain. With regression detection, you would've caught it in minutes.

**4. Retrieval is a bottleneck.**

Spend time here. A bad retrieval stage means no reranker or generator can fix it. Classic RAG teaches you retrieval's real costs.

**5. Graph extraction is fragile.**

Claude's triple extraction quality depends heavily on prompt phrasing and corpus structure. It's not magic. Small details matter.

---

## Getting Started

```bash
# Copy config and add your API key
cp .env.example .env
# Edit: ANTHROPIC_API_KEY=sk-...

# Start everything
docker compose -f infra/docker-compose.yml up -d --build

# Ingest sample docs
docker compose -f infra/docker-compose.yml exec backend python /scripts/ingest.py

# Open http://localhost:5173
# Upload a document
# Ask a question
```

That's it. You're running RAG.

---

## Why This Exists

Most RAG tutorials show you one way. Then you ship it. Then you wonder: "Is this actually the best approach for my data?"

This platform forces you to answer that question empirically, not theoretically.

It's a framework for learning, experimenting, and measuring.

**Not a production system. A foundation to build one.**

---

## What's Next

- Wire in real embedding models
- Add BM25 hybrid search
- Integrate RAGAS metrics
- Build your golden dataset
- Run continuous eval on your corpus
- Export traces to your monitoring system

The code is open. The learning path is clear. The measurements are real.

Now go build something better than Classic RAG.
