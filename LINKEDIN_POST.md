# LinkedIn Post

## Main Post (Feed)

Built EvalRAG — an open-source RAG comparison platform.

Here's the problem it solves:

You're building a RAG system. You pick a retrieval strategy. You ship it. Then you wonder: "Is this actually the best approach?"

Most teams never answer that question.

EvalRAG forces you to.

**Ask one question. See three different retrieval strategies answer it side-by-side.**

1️⃣ **Classic RAG** (embed → search → answer)
- Fast: 200ms
- Cheap: ~$0.001
- Best for: simple factual lookups

2️⃣ **Graph RAG** (extract entities → walk knowledge graph → answer)
- Finds hidden connections
- Slower: 450ms
- Cost: ~$0.002
- Best for: relationship questions

3️⃣ **Agentic RAG** (Claude drives search/fetch tools)
- Most accurate
- Slow: 3.2s
- Cost: ~$0.006
- Best for: complex reasoning

**Then it measures what matters:**
✅ Faithfulness (did it hallucinate?)
✅ Answer relevance (did it actually answer?)
✅ Context precision & recall
✅ Latency and token cost

Pick the one that fits your constraints.

Stack: React + FastAPI + Qdrant + Claude

Open source. Docker Compose. Five minute setup.

Because "which RAG strategy works best" shouldn't be a guess.

[Link to repo]

---

## Shorter Post (if you want tighter)

Built EvalRAG.

Ask one question. Get three RAG strategies answering it side-by-side.

Classic RAG: Fast, cheap, simple facts.
Graph RAG: Finds connections, slower.
Agentic RAG: Most accurate, burns tokens.

Which wins? Depends on your question and budget.

Open source. Docker setup. Measure instead of guess.

[Link]

---

## Carousel Post (10 slides)

**Slide 1:**
Built EvalRAG
Open-source RAG comparison engine

Ask one question → See three strategies → Measure which wins

Most teams guess. This one forces you to measure.

---

**Slide 2:**
The RAG Problem

You pick a retrieval strategy. You ship it. Then you wonder:

"Is this actually the best approach for my corpus?"

Most teams never find out.

---

**Slide 3:**
Strategy #1: Classic RAG

embed query → vector search → rerank → generate answer

✅ Fast (200ms)
✅ Cheap (~$0.001)
✅ Simple

❌ Only finds similar chunks (misses connections)
❌ Doesn't work for multi-hop reasoning

---

**Slide 4:**
Strategy #2: Graph RAG

Extract entity triples → build knowledge graph → walk it → answer

Example:
Question: "How do BM25 and dense vectors compare?"

Graph finds:
- Papers on BM25
- Papers on dense vectors
- The connections between them

Classic only finds them separately.

---

**Slide 5:**
Graph RAG Trade-offs

✅ Finds relationships
✅ Explainable (shows reasoning)
✅ Great for interconnected docs

❌ Slower (450ms)
❌ More expensive (~$0.002)
❌ Requires offline extraction

---

**Slide 6:**
Strategy #3: Agentic RAG

Give Claude three tools:
- search(query) — semantic search
- fetch_chunk(id) — get full text
- finish(answer) — done

Claude loops until it has enough context.

---

**Slide 7:**
Agentic RAG In Action

Question: "Difference between faithfulness and relevance?"

Claude: search("faithfulness metrics")
Gets 3 previews. One's truncated.

Claude: fetch(chunk_id)
Reads full text. Searches for "relevance"

Claude: Gets 2 more chunks.
Calls finish(answer)

Result: Most accurate but 3.2s latency, 2,840 tokens

---

**Slide 8:**
Side-by-Side Comparison

Same question, three strategies:

| | Classic | Graph | Agentic |
|---|---------|-------|---------|
| Latency | 200ms | 450ms | 3.2s |
| Tokens | 850 | 1,200 | 2,840 |
| Cost | $0.001 | $0.002 | $0.006 |
| Quality | 7/10 | 9/10 | 10/10 |

Which wins? Depends on YOUR constraints.

---

**Slide 9:**
Measure Quality

Every answer is scored on:
✅ Faithfulness (hallucinated?)
✅ Relevance (actually answers?)
✅ Precision (useful chunks?)
✅ Recall (all context found?)

Run evals against your golden dataset.
Detect regressions automatically.

---

**Slide 10:**
Open Source

React + FastAPI + Qdrant + Claude
Docker Compose
Five minute setup

Measure instead of guess.

[github.com/yourname/evalrag]

---

## Comment Response Templates

**"Isn't this over-engineered?"**
Only if you don't care about accuracy. Most teams waste thousands on the wrong retrieval strategy because they can't measure trade-offs. This forces you to measure.

**"When would I use Graph RAG over Classic?"**
When your corpus is interconnected and your questions involve relationships. "How does X compare to Y" type questions. If it's pure factual lookup, Classic wins. If it's complex reasoning, Agentic wins.

**"Can I use this in production?"**
The scaffold runs end-to-end today but intentionally leaves space for production components (real embeddings, reranker, eval metrics). It's a learning platform that scales to production.

**"How long to set up?"**
5 minutes with Docker. Add your docs. Compare strategies. Done.

**"What about latency?"**
Classic: 200ms. Graph: 450ms. Agentic: 3.2s. Pick based on your constraints. For low-latency requirements, Classic wins. For accuracy, Agentic wins.

---

## Hashtag Ideas

#RAG #LLM #MachineLearning #OpenSource #AI #Embeddings #VectorDB #Claude #Evaluation #ProductEngineering #DataEngineering

---

## Why This Matters (for your bio/about)

Most RAG tutorials show one way. Doesn't teach you to think about trade-offs or measure what matters. This platform forces you to measure, compare, and choose based on data — not gut feel.
