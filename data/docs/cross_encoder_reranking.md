# Cross-Encoder Reranking for RAG

## Bi-Encoders vs. Cross-Encoders

### Bi-Encoders (Used in Dense Retrieval)
- Encode query and documents **separately** and independently
- Compute similarity as dot product or cosine
- Fast: O(1) similarity for pre-embedded documents
- Limited: Cannot model deep query-document interactions
- Used for: Initial retrieval (top-K)

```
Query: "What is machine learning?"
↓ [embed independently]
Query vector: [0.5, -0.2, 0.8, ...]
Doc vector:   [0.49, -0.19, 0.79, ...]
↓
Similarity: dot product ≈ 0.99
```

### Cross-Encoders (Used for Reranking)
- Take **[query, document] pair** as joint input
- Use transformer self-attention to model interactions
- Slower: Must encode each (query, doc) pair
- More accurate: Can detect subtle relevance signals
- Used for: Reranking top-K candidates

```
Query: "What is machine learning?"
Doc: "Machine learning is a subfield of artificial intelligence..."
↓ [encode jointly]
[CLS] What is machine learning? [SEP] Machine learning is a subfield...
↓
Transformer self-attention computes:
  Query tokens ↔ Document tokens interactions
↓
Relevance score: 0.92 (higher precision than bi-encoder)
```

## How Cross-Encoders Work

### Architecture
1. Tokenize [query, document] pair
2. Pass through transformer encoder (e.g., BERT base)
3. Pool CLS token output
4. Dense layer → sigmoid → score 0-1

### Key Difference: Self-Attention
Self-attention in transformers allows each token to "look at" every other token, enabling the model to detect:
- Semantic alignment between query and document
- Relevant vs. irrelevant passages within documents
- Complex relationships (negation, condition, multipart queries)

## Popular Cross-Encoder Models

### BGE-Reranker (BAAI/BGE Models)
- **BGE-Reranker-base**: 278M params, 512 seq length
  - Speed: ~200 pairs/second on single GPU
  - Quality: 56.2 MTEB score
  - Cost: Free (open-source)
- **BGE-Reranker-large**: 560M params, 1024 seq length
  - Speed: ~100 pairs/second
  - Quality: 59.1 MTEB score
  - Recommended for production

### Cohere Reranker (Commercial)
- Fine-tuned on millions of query-document pairs
- Zero-shot domain performance
- Cost: $0.001 per 1,000 query-document pairs
- Latency: ~50ms for top-10 reranking

### Jina Reranker
- Open-source, 570M params
- Supports 8,192 token passages (long documents)
- Speed: 150-200 pairs/second
- Quality: Comparable to BGE

### Proprietary (Expensive but Fast)
- **OpenAI Reranker** (hypothetical future): Would be API-based
- **Anthropic (via Claude)**: Can use LLM-as-judge for reranking (very slow)

## Performance Impact

### Quality Improvement (NDCG)
```
Baseline (no rerank):    0.52
After reranking top-10:  0.59 (+7.0 NDCG points)
After reranking top-20:  0.58 (+6.0 NDCG points)
After reranking top-50:  0.56 (+4.0 NDCG points)
```

**Key insight**: Reranking top-10 gives 80% of the benefit; diminishing returns past top-20.

### Latency & Cost
```
Dense retrieval (top-50): 5ms, $0.00001 (embedding cost)
Rerank top-10:          10ms, $0.000001 (inference cost)
─────────────────────────────
Total latency: 15ms
Total cost: $0.000011 per query

vs.

Rerank top-50:          50ms, $0.000005
Total latency: 55ms
ROI: Better to rerank top-10 and do 3x more queries for cost
```

## When to Rerank

### Use Reranking When
1. **Precision matters more than recall** (e.g., law, finance)
   - Top document must be highly relevant
   - False positives costly (hallucination risk)
2. **Retrieval precision is low** (<80% for top-10)
   - Dense search has many near-misses
   - Reranking disambiguates
3. **Small budget of tokens for LLM** (<5 chunks)
   - Must pass only highest-quality passages
   - Reranking filters out marginal results
4. **Domain shift from training data**
   - Pre-trained retrieval scores unreliable
   - Reranking adapts to your domain

### Skip Reranking When
1. **Speed is critical** (real-time search)
   - Reranking adds 10-50ms latency
2. **Dense search already precise** (>85% recall@5)
   - Marginal benefit doesn't justify cost
3. **Large context window for LLM** (100K tokens)
   - Can afford to pass top-50 results unfiltered
   - LLM selection better than reranker filtering

## Implementing Reranking in RAG

### Option 1: Local Open-Source (Recommended for Production)

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-large')

# Retrieve candidates (e.g., top-50 from hybrid search)
candidates = retrieve_candidates(query, top_k=50)

# Prepare pairs
pairs = [[query, doc.text] for doc in candidates]

# Rerank
scores = reranker.predict(pairs)

# Sort by score descending
reranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)[:10]  # Take top-10
```

### Option 2: API-Based (Cohere)

```python
import cohere

client = cohere.Client(api_key="...")

results = client.rerank(
    query="What is machine learning?",
    documents=[doc.text for doc in candidates],
    model="rerank-english-v2.0",
    top_n=10
)

# results.results[i].index, results.results[i].relevance_score
```

### Option 3: Cascade (Cost-Optimized)

```python
# Cheap retrieval (hybrid)
candidates = hybrid_search(query, top_k=50)

# Expensive reranking (only top-10)
final = rerank_cascade(candidates[:10], model='bge-reranker-large')
```

## Multi-Stage Reranking

For extremely high-precision requirements (e.g., legal):

### Stage 1: Fast Reranker
- Use lightweight model (bge-reranker-base)
- Filter top-50 → top-20

### Stage 2: Expensive Reranker
- Use large model (bge-reranker-large or LLM)
- Filter top-20 → top-5

### Stage 3: Manual Review
- For critical documents, human reviewer checks top-3

Cost: ~5x more than single reranking, but precision is extreme.

## Common Mistakes to Avoid

1. **Reranking before filtering**: Rerank top-K only, not all results
2. **Reranking passages separately**: Use full document context in reranker input
3. **Ignoring domain shift**: Open-source rerankers may not match your domain; fine-tune if budget allows
4. **Not measuring ROI**: Some queries don't benefit from reranking; measure NDCG lift first
5. **Over-reranking**: Reranking all results wastes compute; top-10 to 20 is optimal

## Reranking in Your RAG Pipeline

### Recommended Architecture
```
1. Query embedding        (1ms)
2. Dense retrieval        (5ms)  → top-50
3. BM25 retrieval        (2ms)  → top-20
4. RRF fusion            (1ms)  → top-20
5. Cross-encoder rerank  (10ms) → top-5
6. LLM generation        (varies)
─────────────────────────────────
Total retrieval: ~19ms
Quality gain: +5-7 NDCG points
```

### Cost-Quality Tradeoff
| Strategy | Latency | Quality | Cost |
|----------|---------|---------|------|
| Dense only | 5ms | 0.70 NDCG | $0.00001 |
| Dense + rerank top-10 | 15ms | 0.77 NDCG | $0.00002 |
| Hybrid + rerank top-5 | 10ms | 0.75 NDCG | $0.000015 |
| Dense + no rerank | 5ms | 0.70 NDCG | $0.00001 |

**Winner for production**: Hybrid + rerank top-5 (best tradeoff)

## Measuring Reranker Effectiveness

### Metrics to Track
1. **NDCG@5**: Normalized discounted cumulative gain (0-1)
2. **Recall@10**: Did top-10 contain the relevant doc?
3. **Precision@1**: Is the top result relevant?

### Baseline Test
1. Run hybrid search (no rerank) on golden dataset
2. Run hybrid + reranking on same queries
3. Measure NDCG improvement
4. If <5 points, reranking may not be worth it

### Domain-Specific Tuning
If reranker underperforms:
1. Fine-tune open-source reranker on your domain (1k-10k labeled pairs)
2. Or use Cohere API (learns from all data)
3. Or use LLM-as-judge (very accurate but expensive)

Cross-encoder reranking is the simplest, highest-ROI improvement to add to any RAG system after establishing hybrid retrieval.
