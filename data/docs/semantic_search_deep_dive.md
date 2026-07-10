# Semantic Search Deep Dive: Architecture and Implementation

## Introduction

Semantic search represents a fundamental shift from keyword-based retrieval to meaning-based search. This document explores the architecture, algorithms, and practical implementation of semantic search systems for RAG.

## 1. Semantic vs. Lexical Search

### Conceptual Differences

**Lexical Search:**
- Matches keywords and phrases exactly
- Fast and predictable
- Limited to explicit terms in documents
- Poor handling of synonyms and paraphrasing
- Examples: BM25, TF-IDF

**Semantic Search:**
- Matches meaning and intent
- Understands synonyms and concept relationships
- Captures implicit information through embeddings
- Better for conceptual queries
- Examples: Dense retrieval, embedding-based search

### Example Comparison

```python
Query: "What is the capital of France?"

Lexical Results:
- Documents containing "capital", "France", "Paris"
- May return: "Paris is a city" (contains keywords)
- May miss: "The city of Paris serves as France's center of government"

Semantic Results:
- Documents about French capital/government
- Returns: Highly relevant results about Paris and France
- Understanding: Query intent is about geography/government
```

## 2. Semantic Search Architecture

### End-to-End Pipeline

```
User Query
    ↓
Query Encoding (Embedding Model)
    ↓
Vector Lookup (Vector DB)
    ↓
Candidate Ranking (Similarity Scoring)
    ↓
Post-processing (Filtering/Reranking)
    ↓
Results Presentation
```

### Detailed Implementation

```python
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearchEngine:
    def __init__(self, embedding_model='intfloat/e5-large'):
        self.model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = []
    
    def index_documents(self, documents: List[str]):
        """Index documents by computing embeddings"""
        print(f"Indexing {len(documents)} documents...")
        self.documents = documents
        
        # Batch encoding for efficiency
        self.embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Semantic search returning top-k results"""
        # Encode query
        query_embedding = self.model.encode(query)
        
        # Compute similarity scores
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return documents with scores
        results = [
            (self.documents[i], float(similarities[i]))
            for i in top_indices
        ]
        
        return results
    
    def explain_result(self, query: str, doc_idx: int):
        """Explain why a document was retrieved"""
        query_emb = self.model.encode(query)
        doc_emb = self.embeddings[doc_idx]
        
        # Similarity score
        similarity = cosine_similarity(
            [query_emb],
            [doc_emb]
        )[0][0]
        
        # Token-level analysis (simplified)
        return {
            'document': self.documents[doc_idx][:100] + "...",
            'similarity_score': similarity,
            'explanation': f"Semantic similarity: {similarity:.3f}"
        }

# Example usage
search_engine = SemanticSearchEngine()

documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand text.",
    "Computer vision allows machines to interpret images.",
    "Reinforcement learning trains agents through rewards."
]

search_engine.index_documents(documents)

# Search
results = search_engine.search("What is AI and its applications?", top_k=3)
for doc, score in results:
    print(f"Score: {score:.3f} - {doc}")
```

## 3. Similarity Metrics in Semantic Search

### Cosine Similarity

Most common metric for embedding-based search. Measures angle between vectors.

```
Formula: cos(θ) = (A · B) / (|A| × |B|)
Range: [-1, 1]
Characteristics:
  - Invariant to magnitude (only direction matters)
  - Fast to compute
  - Works well for normalized embeddings
  - Best for: General semantic search
```

### Dot Product

Faster than cosine for pre-normalized embeddings (many modern models).

```python
similarity = np.dot(query_embedding, doc_embedding)
# For normalized embeddings, equivalent to cosine but faster
```

### Euclidean Distance

Alternative metric, less common for text embeddings.

```python
distance = np.linalg.norm(query_embedding - doc_embedding)
# Inversely related to similarity
# Use when: Working with models that optimize for distance
```

### Comparison Framework

```python
class SimilarityMetricComparison:
    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @staticmethod
    def dot_product(a, b):
        return np.dot(a, b)
    
    @staticmethod
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b)
    
    @staticmethod
    def manhattan_distance(a, b):
        return np.sum(np.abs(a - b))
    
    @staticmethod
    def compare_metrics(query_emb, doc_embs):
        """Compare different metrics"""
        results = {
            'cosine': [],
            'dot_product': [],
            'euclidean': [],
            'manhattan': []
        }
        
        for doc_emb in doc_embs:
            results['cosine'].append(
                SimilarityMetricComparison.cosine_similarity(
                    query_emb, doc_emb
                )
            )
            results['dot_product'].append(
                SimilarityMetricComparison.dot_product(query_emb, doc_emb)
            )
            results['euclidean'].append(
                SimilarityMetricComparison.euclidean_distance(
                    query_emb, doc_emb
                )
            )
            results['manhattan'].append(
                SimilarityMetricComparison.manhattan_distance(
                    query_emb, doc_emb
                )
            )
        
        return results
```

## 4. Approximate Nearest Neighbor Search

Exact similarity search scales poorly (O(n)). Approximate methods enable fast search.

### HNSW (Hierarchical Navigable Small World)

```python
import hnswlib

class HNSWSemanticSearch:
    def __init__(self, dimension=1536, max_elements=1000000):
        self.index = hnswlib.Index(
            space='cosine',
            dim=dimension
        )
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        self.documents = []
    
    def index_documents(self, docs, embeddings):
        """Index documents for fast search"""
        self.documents = docs
        
        # Add items to index
        doc_ids = np.arange(len(embeddings))
        self.index.add_items(embeddings, doc_ids)
    
    def search(self, query_embedding, k=10, ef=100):
        """Fast approximate nearest neighbor search"""
        self.index.set_ef(ef)  # Tradeoff: higher ef = more accurate but slower
        
        labels, distances = self.index.knn_query(
            [query_embedding],
            k=k
        )
        
        results = []
        for idx, distance in zip(labels[0], distances[0]):
            # Convert distance to similarity (cosine distance → similarity)
            similarity = 1 - distance
            results.append((self.documents[idx], similarity))
        
        return results

# Benchmark: Exact vs HNSW
print("Exact Search:")
results_exact = search_engine.search(query, top_k=10)

print("\nHNSW Search:")
hnsw = HNSWSemanticSearch()
hnsw.index_documents(documents, search_engine.embeddings)
results_hnsw = hnsw.search(query_embedding, k=10)

# HNSW typically returns 95%+ of exact top-k with 100-1000x speedup
```

## 5. Advanced Semantic Search Techniques

### Contextual Search

```python
class ContextualSemanticSearch:
    def __init__(self, embedding_model):
        self.model = embedding_model
    
    def search_with_context(self, query, documents, user_context):
        """Search considering user context"""
        # Add context to query
        augmented_query = f"""
        User Context: {user_context}
        Query: {query}
        """
        
        # Encode augmented query
        query_emb = self.model.encode(augmented_query)
        
        # Also encode documents with context
        enriched_docs = [
            f"Context: {user_context}\nDocument: {doc}"
            for doc in documents
        ]
        
        doc_embs = self.model.encode(enriched_docs)
        
        # Similarity search
        similarities = cosine_similarity([query_emb], doc_embs)[0]
        
        return similarities
```

### Query Understanding and Expansion

```python
class QueryUnderstandingEngine:
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_query(self, query: str):
        """Understand query intent and entities"""
        prompt = f"""Analyze this query:
        Query: {query}
        
        Provide:
        1. Query intent (search, question, exploration, etc.)
        2. Key entities mentioned
        3. Implicit concepts
        4. Alternative phrasings
        """
        
        analysis = self.llm.generate(prompt)
        return analysis
    
    def expand_query_semantically(self, query: str):
        """Generate semantically similar queries"""
        prompt = f"""Generate 3 alternative ways to ask this:
        {query}
        
        Keep the same meaning but vary vocabulary and phrasing."""
        
        expansions = self.llm.generate(prompt, num_variants=3)
        return [query] + expansions
```

## 6. Semantic Search Evaluation

### Metrics

```python
from sklearn.metrics import ndcg_score, mean_reciprocal_rank

class SemanticSearchEvaluator:
    def __init__(self, search_engine):
        self.search_engine = search_engine
    
    def evaluate_recall(self, queries, relevant_docs_map, k=10):
        """Recall@k: % of relevant docs in top-k"""
        recalls = []
        
        for query, relevant_doc_ids in relevant_docs_map.items():
            results = self.search_engine.search(query, top_k=k)
            retrieved_ids = [self.search_engine.documents.index(doc)
                           for doc, _ in results]
            
            recall = len(set(retrieved_ids) & set(relevant_doc_ids)) / len(relevant_doc_ids)
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def evaluate_precision(self, queries, relevant_docs_map, k=10):
        """Precision@k: % of top-k that are relevant"""
        precisions = []
        
        for query, relevant_doc_ids in relevant_docs_map.items():
            results = self.search_engine.search(query, top_k=k)
            retrieved_ids = [self.search_engine.documents.index(doc)
                           for doc, _ in results]
            
            precision = len(set(retrieved_ids) & set(relevant_doc_ids)) / k
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def evaluate_mrr(self, queries, relevant_docs_map):
        """Mean Reciprocal Rank"""
        mrrs = []
        
        for query, relevant_doc_ids in relevant_docs_map.items():
            results = self.search_engine.search(query, top_k=100)
            
            for rank, (doc, _) in enumerate(results, 1):
                doc_id = self.search_engine.documents.index(doc)
                if doc_id in relevant_doc_ids:
                    mrrs.append(1 / rank)
                    break
            else:
                mrrs.append(0)
        
        return np.mean(mrrs)
    
    def evaluate_ndcg(self, queries, relevance_scores_map, k=10):
        """Normalized Discounted Cumulative Gain"""
        ndcgs = []
        
        for query, ideal_scores in relevance_scores_map.items():
            results = self.search_engine.search(query, top_k=k)
            
            # Assign relevance scores
            actual_scores = []
            retrieved_docs = [doc for doc, _ in results]
            
            for doc in retrieved_docs:
                doc_idx = self.search_engine.documents.index(doc)
                actual_scores.append(
                    ideal_scores.get(doc_idx, 0)
                )
            
            # Compute NDCG
            actual_scores = np.array(actual_scores + [0] * (k - len(actual_scores)))
            ideal_scores_sorted = sorted(ideal_scores.values(), reverse=True)
            ideal_scores_arr = np.array(ideal_scores_sorted + [0] * (k - len(ideal_scores)))
            
            ndcg = ndcg_score([ideal_scores_arr], [actual_scores])
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs)
```

## 7. Production Optimization

### Caching Layer

```python
from functools import lru_cache
import hashlib

class CachedSemanticSearch:
    def __init__(self, search_engine, cache_size=10000):
        self.search_engine = search_engine
        self.query_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def search(self, query: str, top_k=10):
        """Search with query result caching"""
        cache_key = f"{query}:{top_k}"
        
        if cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]
        
        # Cache miss
        results = self.search_engine.search(query, top_k)
        
        if len(self.query_cache) >= self.cache_size:
            # Evict oldest entry (simple FIFO)
            self.query_cache.pop(next(iter(self.query_cache)))
        
        self.query_cache[cache_key] = results
        self.cache_misses += 1
        
        return results
    
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.query_cache)
        }
```

## Best Practices

1. **Choose embeddings carefully** - most variation in quality comes from embedding model, not retrieval algorithm
2. **Use HNSW or similar** for production scale (> 1M documents)
3. **Monitor similarity score distributions** - indicates if your corpus matches query types
4. **Implement caching** - typical hit rates 60-80% for production systems
5. **Evaluate with real user queries** - synthetic queries don't capture actual difficulty

## Conclusion

Semantic search powered by modern embedding models enables powerful information retrieval. Success depends on choosing appropriate embeddings, implementing efficient similarity search algorithms, and continuously evaluating on actual use cases. Most improvements come from better embeddings and query understanding rather than algorithmic innovations.
