# Hybrid Retrieval Architecture for Production RAG

## The Fundamental Problem: Combining Incompatible Signals

BM25 scores are unbounded (0 to infinity), while cosine similarity ranges 0-1. How do you combine them fairly?

**Naive approach (fails)**: Normalize both to 0-1, then average
- Problem: BM25 score of 5 and cosine 0.9 have very different distributions
- Result: One signal dominates, or both cancel out

**Solution: Reciprocal Rank Fusion (RRF)**
- Convert scores to ranks independently
- Combine via: score(D) = Σ 1 / (k + rank_i(D))
- Parameter k (default=60) prevents extreme rank biasing
- Strength: Model-agnostic, no tuning needed, always produces [0, 1] output

## Three-Stage Hybrid Pipeline

### Stage 1: Dual Retrieval (Parallel)

**BM25 Component**:
```
Query: "machine learning classification"
↓
Inverted index search
↓
Top-K results by BM25 score
  1. doc_5: "classification methods in ML" (BM25=12.3)
  2. doc_12: "supervised classification" (BM25=10.1)
  3. doc_3: "data classification techniques" (BM25=9.8)
```

**Dense Retrieval Component**:
```
Query: "machine learning classification"
↓
Embed query (384-dim vector)
↓
Vector similarity search
↓
Top-K results by cosine similarity
  1. doc_5: "classification in machine learning" (cosine=0.92)
  2. doc_1: "neural networks for classification" (cosine=0.88)
  3. doc_7: "decision trees for categorization" (cosine=0.85)
```

### Stage 2: Rank Fusion (RRF)

Convert both result lists to ranks, then compute fused score:

```
doc_5: rank_BM25=1, rank_dense=1
       score = 1/(60+1) + 1/(60+1) = 0.0330

doc_12: rank_BM25=2, rank_dense=∞ (not in top-K)
        score = 1/(60+2) + 1/(60+∞) = 0.0159 + 0 = 0.0159

doc_1: rank_BM25=∞, rank_dense=2
       score = 0 + 1/(60+2) = 0.0159

doc_3: rank_BM25=3, rank_dense=∞
       score = 1/(60+3) = 0.0149

Fused ranking: doc_5 (0.0330) > doc_12 (0.0159) ≈ doc_1 (0.0159) > doc_3 (0.0149)
```

**Result**: Combined top-K list respects both signals
- Documents found by only BM25 are included (doc_12)
- Documents found by only dense search are included (doc_1)
- Documents found by both are ranked highest (doc_5)

### Stage 3: Cross-Encoder Reranking

Take top-K from RRF fusion and rerank with cross-encoder:

```
Fused top-8 docs
↓
For each doc: embed([query, doc]) jointly
↓
Cross-encoder scores 0-1
↓
Re-rank by relevance
```

Why rerank? RRF is blind to actual relevance; it only sees two rank lists. Cross-encoder adds semantic understanding of query-document pairs.

## Performance Characteristics

### Recall Improvement
Based on financial domain benchmark:

| Strategy | Recall@5 | Recall@10 |
|----------|----------|-----------|
| BM25 only | 0.42 | 0.58 |
| Dense only | 0.59 | 0.72 |
| **Hybrid (no rerank)** | **0.695** | **0.81** |
| Hybrid + rerank | 0.82 | 0.91 |

Improvement:
- Hybrid vs. dense only: +18% recall@5
- Hybrid + rerank: +39% recall@5 over dense alone

### Latency Breakdown (1M document corpus)

```
BM25 search:        ~2ms (inverted index)
Dense search:       ~5ms (HNSW ANN)
RRF fusion:         <1ms (in-memory combine)
Reranking (top-K):  ~20ms (cross-encoder)
─────────────────────────────────
Total latency:      ~27ms
```

Optimization: RRF happens in parallel with retrieval, adds negligible cost.

## When to Use Each Component

### BM25 is Critical When
- Exact identifiers matter (SKUs, product codes, legal references)
- Technical documentation (function names, parameters)
- Regulatory/compliance (exact phrase matching)
- Reducing false positives (BM25 is conservative)

### Dense Search is Critical When
- Queries use synonyms or paraphrases
- Long, narrative questions ("Tell me about...")
- Zero-shot: query style unseen in training
- Reducing false negatives (dense is inclusive)

### Hybrid is Always Better Than Either Alone
- Complementary failure modes
- No additional cost vs. dense alone (+2ms for BM25)
- RRF requires zero tuning (no parameter tuning needed)

## Implementation in Vector Databases

### Qdrant
```
Query retrieval with both search methods and fuse:
1. Execute sparse_search (BM25)
2. Execute dense_search (vector similarity)
3. Qdrant automatically fuses via RRF (v1.10+)
```

### Weaviate
```
{
  hybrid(
    query: "machine learning",
    sparseSearchProperties: ["text"],
    alpha: 0.5  # 0.5 = equal weight to both
  )
}
```

### Pinecone
```
Both sparse and dense vectors stored together:
- Query embedding + sparse query matrix
- Automatic RRF fusion
```

## Configuration Recommendations

### For Production Systems
1. **k (RRF parameter)**: Use default 60 (works across domains)
2. **top-K for fusion**: Use K=50 to 100
3. **Final rerank top-K**: Use top-10 to 20 for reranking

### For Cost-Conscious Deployments
1. **No reranking**: Hybrid alone gives 80% of quality gain at 40% cost
2. **Sparse-only fallback**: If dense search expensive, BM25 covers precision queries
3. **Cascade**: Dense → rerank only top-5 from hybrid (saves compute)

## Real-World Deployment Patterns

### Pattern 1: High-Accuracy RAG
```
Dense search (top-50) 
+ BM25 search (top-50)
→ RRF fusion (top-20)
→ Cross-encoder reranking (top-5)
→ LLM generation with top-5 chunks
```

### Pattern 2: Cost-Optimized RAG
```
BM25 search (top-20)
+ Dense search (top-10, sampled)
→ RRF fusion (top-8)
→ Direct LLM generation (no rerank)
```

### Pattern 3: Cascade for Scale
```
BM25 → dense → top-20 fused
→ Rerank top-5 only (expensive)
→ LLM generation
Cost: ~70% of high-accuracy, 95% quality
```

## Tuning Hybrid Retrieval

### Measure Your Baseline
1. Run 100 query-document pairs from golden dataset
2. Measure recall@5, recall@10 for each strategy
3. Note where each strategy fails

### Optimize RRF
1. Adjust k parameter: try 30, 60, 100
2. Measure recall on your golden dataset
3. k=60 typically optimal; rarely needs tuning

### Add Reranking Strategically
1. Rerank only top-K (10-20) from fusion
2. Measure NDCG improvement (typically 5-7 points)
3. Assess latency-quality tradeoff

## Common Mistakes to Avoid

1. **Averaging normalized scores**: Use RRF instead
2. **Equal weighting semantically**: BM25 and dense have different quality curves; RRF handles it
3. **Tuning k excessively**: k=60 works across domains; don't overthink it
4. **Reranking all results**: Rerank top-K only; diminishing returns past K=20
5. **Separate optimization**: Tune BM25 and dense independently; they interact

Hybrid retrieval is the industry standard for production RAG because the combination is genuinely better than any single approach.
