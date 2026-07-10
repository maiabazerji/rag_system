# Sparse vs Dense Retrieval: Trade-offs and Hybrid Approaches

## Introduction

Retrieval systems use different strategies: sparse (keyword-based) and dense (semantic). This guide covers when to use each and how to combine them effectively.

## 1. Sparse Retrieval (BM25)

### Algorithm Overview

```python
from collections import Counter
import math

class BM25Retriever:
    """BM25 sparse retrieval"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization
        self.documents = []
        self.doc_freqs = {}  # Document frequencies
        self.avg_doc_length = 0
    
    def index_documents(self, documents: list):
        """Index documents for BM25"""
        self.documents = documents
        
        total_length = 0
        vocabulary = set()
        
        # Calculate document frequencies and length
        doc_lengths = {}
        
        for doc_id, doc_text in enumerate(documents):
            words = self._tokenize(doc_text)
            doc_lengths[doc_id] = len(words)
            total_length += len(words)
            
            # Count word occurrences in document
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word not in self.doc_freqs:
                    self.doc_freqs[word] = 0
                self.doc_freqs[word] += 1
                vocabulary.add(word)
        
        self.avg_doc_length = total_length / len(documents) if documents else 0
        self.doc_lengths = doc_lengths
        self.num_docs = len(documents)
    
    def retrieve(self, query: str, top_k: int = 10):
        """Retrieve documents using BM25"""
        query_words = self._tokenize(query)
        
        scores = {}
        
        for doc_id, doc_text in enumerate(self.documents):
            score = self._compute_bm25_score(doc_id, query_words, doc_text)
            scores[doc_id] = score
        
        # Sort and return top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'doc_id': doc_id,
                'text': self.documents[doc_id],
                'score': score
            }
            for doc_id, score in ranked[:top_k]
        ]
    
    def _compute_bm25_score(self, doc_id: int, query_words: list, doc_text: str):
        """Compute BM25 score for document"""
        score = 0
        doc_length = self.doc_lengths.get(doc_id, 0)
        
        for word in query_words:
            # IDF calculation
            word_doc_freq = self.doc_freqs.get(word, 0)
            idf = math.log((self.num_docs - word_doc_freq + 0.5) / 
                          (word_doc_freq + 0.5) + 1)
            
            # Term frequency in document
            doc_words = self._tokenize(doc_text)
            term_freq = doc_words.count(word)
            
            # BM25 formula
            tf_component = (self.k1 + 1) * term_freq / \
                          (self.k1 * (1 - self.b + self.b * 
                          (doc_length / self.avg_doc_length)) + term_freq)
            
            score += idf * tf_component
        
        return score
    
    def _tokenize(self, text: str):
        """Simple tokenization"""
        return text.lower().split()
```

### BM25 Characteristics

```
Strengths:
- Fast and efficient
- Works well for exact phrase matching
- No training required
- Interpretable scores

Weaknesses:
- Poor for synonyms
- Fails with paraphrasing
- No semantic understanding
- Sensitive to vocabulary
```

## 2. Dense Retrieval

### Vector-Based Retrieval

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    """Dense vector-based retrieval"""
    
    def __init__(self, model_name: str = 'intfloat/e5-large'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def index_documents(self, documents: list):
        """Index documents with embeddings"""
        self.documents = documents
        
        # Embed all documents
        self.embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True
        )
    
    def retrieve(self, query: str, top_k: int = 10):
        """Retrieve using semantic similarity"""
        query_embedding = self.model.encode(query)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                'doc_id': idx,
                'text': self.documents[idx],
                'score': float(similarities[idx])
            }
            for idx in top_indices
        ]
```

### Dense Characteristics

```
Strengths:
- Semantic understanding
- Handles synonyms and paraphrasing
- Works across languages
- Robust to word order changes

Weaknesses:
- Slower inference
- Requires embeddings model
- Memory intensive for large collections
- Poor for exact matching
```

## 3. Hybrid Retrieval

### Combining Sparse and Dense

```python
class HybridRetriever:
    """Combine sparse and dense retrieval"""
    
    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
    
    def retrieve_hybrid(self, query: str, top_k: int = 10, 
                       alpha: float = 0.5):
        """Retrieve using both methods and combine"""
        
        # Retrieve with both methods
        bm25_results = self.bm25.retrieve(query, top_k=top_k*2)
        dense_results = self.dense.retrieve(query, top_k=top_k*2)
        
        # Normalize scores to 0-1
        bm25_normalized = self._normalize_scores(bm25_results)
        dense_normalized = self._normalize_scores(dense_results)
        
        # Combine scores
        combined_scores = {}
        
        for result in bm25_normalized:
            doc_id = result['doc_id']
            if doc_id not in combined_scores:
                combined_scores[doc_id] = 0
            combined_scores[doc_id] += (1 - alpha) * result['score']
        
        for result in dense_normalized:
            doc_id = result['doc_id']
            if doc_id not in combined_scores:
                combined_scores[doc_id] = 0
            combined_scores[doc_id] += alpha * result['score']
        
        # Rank by combined score
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'doc_id': doc_id,
                'text': self.dense.documents[doc_id],
                'score': score
            }
            for doc_id, score in ranked[:top_k]
        ]
    
    def _normalize_scores(self, results: list):
        """Normalize scores to 0-1"""
        if not results:
            return results
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        normalized = []
        for result in results:
            normalized_score = (result['score'] - min_score) / (max_score - min_score) if max_score > min_score else 1
            normalized.append({
                **result,
                'score': normalized_score
            })
        
        return normalized
```

## 4. Choosing Retrieval Strategy

### Selection Criteria

```python
class StrategySelectorclass StrategySelector:
    """Select optimal retrieval strategy"""
    
    @staticmethod
    def select_strategy(use_case: str, corpus_size: int, latency_requirement: int):
        """Select retrieval strategy"""
        
        strategy = {
            'name': '',
            'retriever': '',
            'reasoning': ''
        }
        
        # Small corpus, exact matching needed
        if corpus_size < 10000 and use_case == 'FAQ':
            strategy['name'] = 'sparse'
            strategy['retriever'] = 'BM25'
            strategy['reasoning'] = 'Small corpus, exact matching important'
        
        # Large corpus, semantic matching
        elif corpus_size > 1000000 and use_case == 'general':
            strategy['name'] = 'dense_with_caching'
            strategy['retriever'] = 'Dense vectors with caching'
            strategy['reasoning'] = 'Large corpus, semantic understanding needed'
        
        # Real-time requirement, small corpus
        elif latency_requirement < 50 and corpus_size < 100000:
            strategy['name'] = 'sparse'
            strategy['retriever'] = 'BM25'
            strategy['reasoning'] = 'Ultra-low latency needed'
        
        # Balance needed
        else:
            strategy['name'] = 'hybrid'
            strategy['retriever'] = 'BM25 + Dense (equal weight)'
            strategy['reasoning'] = 'Balance between exact and semantic matching'
        
        return strategy
    
    @staticmethod
    def benchmark_strategies(query: str, corpus: list, ground_truth: set):
        """Benchmark different strategies"""
        
        # Initialize retrievers
        bm25 = BM25Retriever()
        bm25.index_documents(corpus)
        
        dense = DenseRetriever()
        dense.index_documents(corpus)
        
        hybrid = HybridRetriever(bm25, dense)
        
        results = {}
        
        # BM25
        bm25_results = bm25.retrieve(query, top_k=10)
        results['bm25'] = StrategySelector._evaluate_retrieval(
            bm25_results, ground_truth
        )
        
        # Dense
        dense_results = dense.retrieve(query, top_k=10)
        results['dense'] = StrategySelector._evaluate_retrieval(
            dense_results, ground_truth
        )
        
        # Hybrid
        hybrid_results = hybrid.retrieve_hybrid(query, top_k=10)
        results['hybrid'] = StrategySelector._evaluate_retrieval(
            hybrid_results, ground_truth
        )
        
        return results
    
    @staticmethod
    def _evaluate_retrieval(results: list, ground_truth: set):
        """Evaluate retrieval results"""
        retrieved_ids = {r['doc_id'] for r in results}
        
        precision = len(retrieved_ids & ground_truth) / len(retrieved_ids)
        recall = len(retrieved_ids & ground_truth) / len(ground_truth)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }
```

## 5. Advanced Hybrid Techniques

### Learned Fusion

```python
from sklearn.ensemble import GradientBoostingRegressor

class LearnedHybridRetriever:
    """Learn optimal weight for combining sparse and dense"""
    
    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.fusion_model = None
    
    def train_fusion(self, training_queries: list, training_labels: list):
        """Train model to fuse scores"""
        
        features = []
        labels = []
        
        for query, label in zip(training_queries, training_labels):
            bm25_results = self.bm25.retrieve(query, top_k=10)
            dense_results = self.dense.retrieve(query, top_k=10)
            
            # Extract features for each result
            for i in range(10):
                bm25_score = bm25_results[i]['score'] if i < len(bm25_results) else 0
                dense_score = dense_results[i]['score'] if i < len(dense_results) else 0
                
                features.append([bm25_score, dense_score])
                labels.append(label.get(i, 0))
        
        # Train fusion model
        self.fusion_model = GradientBoostingRegressor()
        self.fusion_model.fit(features, labels)
    
    def retrieve_learned(self, query: str, top_k: int = 10):
        """Retrieve using learned fusion"""
        
        if self.fusion_model is None:
            raise ValueError("Must train fusion model first")
        
        bm25_results = self.bm25.retrieve(query, top_k=top_k*2)
        dense_results = self.dense.retrieve(query, top_k=top_k*2)
        
        # Prepare features
        doc_scores = {}
        
        for i, result in enumerate(bm25_results):
            doc_id = result['doc_id']
            dense_score = next(
                (r['score'] for r in dense_results if r['doc_id'] == doc_id),
                0
            )
            
            bm25_score = result['score']
            
            # Predict combined score
            combined = self.fusion_model.predict([[bm25_score, dense_score]])[0]
            doc_scores[doc_id] = combined
        
        # Rank by combined score
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
```

## Performance Comparison

```
Method          Latency    Exact Match    Semantic    Memory    Cost
─────────────────────────────────────────────────────────────────
BM25            10ms       95%            20%         Low       Low
Dense           100ms      30%            95%         High      High
Hybrid (50/50)  60ms       65%            75%         High      Medium
Learned Fusion  70ms       70%            80%         High      Medium
```

## Best Practices

1. **Start with hybrid** - balances precision and recall
2. **Use BM25 for exact match** - complement dense retrieval
3. **Tune alpha (weight)** - 0.3-0.7 works for most cases
4. **Monitor performance** - track both types of queries
5. **Cache dense embeddings** - expensive to recompute

## Conclusion

Hybrid retrieval combining sparse (BM25) and dense (semantic) approaches provides best of both worlds. Sparse is fast and precise for exact matching, dense is semantic and robust. Most production RAG systems use hybrid with tuned weights. For extremely high recall, start with dense then use sparse for diversity.
