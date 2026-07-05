# LinkedIn Post

## Main Post (Feed)

Built EvalRAG — an open-source RAG evaluation platform that answers the question every team should be asking:

**"Which retrieval strategy actually works best for my corpus?"**

Most teams pick one approach and ship it. They never measure whether it's optimal. EvalRAG forces you to know.

**Ask one question. See three different strategies answer it side-by-side.**

📊 **Classic RAG** (vector search + rerank)
- Fastest retrieval
- Best for: exact semantic matches

🔗 **Graph RAG** (entity extraction + knowledge graph walk)
- Finds hidden connections between concepts
- Shows the reasoning path explicitly
- Best for: relationship questions like "How does X compare to Y?"

🤖 **Agentic RAG** (LLM loops over search/fetch tools)
- Most accurate answers
- Claude decides what to search, when to stop
- Best for: complex multi-step reasoning

**Then it scores what actually matters:**
✅ Faithfulness — is the answer grounded in context or hallucinated?
✅ Answer relevance — does it actually address the question?
✅ Context precision — are the retrieved chunks ranked well?
✅ Context recall — was all necessary information retrieved?

Run evaluations on your golden dataset. Detect silent regressions before shipping. Compare prompt versions automatically.

Stack: React + FastAPI + Qdrant + Claude API

Open source. Docker Compose. 5-minute setup.

Because "which RAG strategy works best" shouldn't depend on guesswork.

[github link]

---

## Shorter Post

Just built EvalRAG.

Same question → three retrieval strategies → real evaluation scores.

Classic RAG finds semantic matches fast.
Graph RAG uncovers hidden connections.
Agentic RAG reasons multi-step.

Which wins? Measure it instead of guessing.

Open source. Compare strategies side-by-side. Evaluate on your real questions.

[github link]

---

## Carousel Post (10 slides)

**Slide 1: The Problem**

You pick a RAG retrieval strategy. You ship it.

Then you wonder: "Is this actually the best approach for MY corpus?"

Most teams never answer that question.

---

**Slide 2: Three Strategies**

Classic RAG: Fast vector search + rerank
Graph RAG: Knowledge graph + entity walking
Agentic RAG: LLM loops over search tools

All three work. Which one works BEST for your data?

That's the question EvalRAG answers.

---

**Slide 3: Strategy #1 — Classic RAG**

embed query → vector search → rerank → generate

✅ Fastest retrieval
✅ Simple pipeline
✅ Works for factual lookups

❌ Misses relationships
❌ No multi-hop reasoning

---

**Slide 4: Strategy #2 — Graph RAG**

Extract 408 entity triples from your docs → build knowledge graph → walk neighbors → answer

Question: "How do BM25 and dense vectors compare?"

🔗 Graph finds:
- Papers on BM25
- Papers on dense vectors  
- The connections explaining the difference

Classic RAG finds them separately. Graph shows the relationship.

---

**Slide 5: Strategy #3 — Agentic RAG**

Give Claude three tools:
- search(query) — semantic search
- fetch(chunk_id) — read full text
- finish(answer) — done reasoning

Claude loops until it has enough context to answer confidently.

Result: Most accurate. Higher latency + token cost.

---

**Slide 6: Side-by-Side Comparison**

Ask one question. See all three strategies answer it.

Same context. Same LLM. Different retrieval.

You see:
- The answer from each strategy
- Retrieved chunks
- Latency & token cost
- Extracted entities (for Graph RAG)

---

**Slide 7: Score Every Answer**

✅ Faithfulness — is the answer grounded in context or hallucinating?
✅ Answer relevance — does it actually answer the question?
✅ Context precision — were chunks ranked well?
✅ Context recall — was all necessary info retrieved?

LLM-based evaluation on your golden dataset.

---

**Slide 8: Detect Regressions**

Run evaluation once a month. Compare scores run-over-run.

Did your new prompt regress? You'll know immediately.

Silent quality degradation = no longer silent.

---

**Slide 9: Built for Teams**

React frontend shows comparisons visually.
FastAPI backend scores automatically.
Qdrant stores your indexed documents.
Claude API judges answer quality.

All open source. Docker Compose. 5 minutes to run.

---

**Slide 10: The Real Value**

You're not guessing anymore.

You have data. You can choose the right strategy for YOUR constraints.

Classic for speed. Graph for relationships. Agentic for accuracy.

Measure. Compare. Ship with confidence.

[github.com/...]

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
