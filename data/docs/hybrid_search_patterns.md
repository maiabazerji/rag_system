# Hybrid Search Patterns: Combining Multiple Retrieval Strategies

## Introduction

No single retrieval strategy is optimal for all queries. Hybrid search patterns combine multiple methods to achieve better overall performance. This guide covers architectural patterns and implementation strategies.

## 1. Semantic + Lexical Hybrid

### Combined Dense and Sparse

```python
import numpy as np
from scipy.sparse import csr_matrix

class SemanticLexicalHybrid:
    """Combine semantic (dense) and lexical (sparse) retrieval"""
    
    def __init__(self, dense_retriever, sparse_retriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
    
    def retrieve_hybrid(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """Retrieve using hybrid approach"""
        
        # Get results from both methods
        dense_results = self.dense.retrieve(query, top_k=top_k*2)
        sparse_results = self.sparse.retrieve(query, top_k=top_k*2)
        
        # Normalize scores to [0, 1]
        dense_normalized = self._normalize_scores(dense_results)
        sparse_normalized = self._normalize_scores(sparse_results)
        
        # Combine using weighted sum
        combined_scores = {}
        
        # Add dense scores
        for result in dense_normalized:
            doc_id = result['id']
            combined_scores[doc_id] = alpha * result['score']
        
        # Add sparse scores
        for result in sparse_normalized:
            doc_id = result['id']
            if doc_id not in combined_scores:
                combined_scores[doc_id] = 0
            combined_scores[doc_id] += (1 - alpha) * result['score']
        
        # Rank by combined score
        ranked = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {
                'id': doc_id,
                'score': score,
                'text': self._get_doc_text(doc_id)
            }
            for doc_id, score in ranked
        ]
    
    def _normalize_scores(self, results: list):
        """Normalize scores to [0, 1]"""
        if not results:
            return []
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        normalized = []
        for result in results:
            if max_score > min_score:
                norm_score = (result['score'] - min_score) / (max_score - min_score)
            else:
                norm_score = 1
            
            normalized.append({
                'id': result['id'],
                'score': norm_score
            })
        
        return normalized
    
    def _get_doc_text(self, doc_id: str):
        """Get document text"""
        return self.dense.get_document(doc_id)['text']
```

## 2. Query Routing

### Route Queries to Optimal Strategy

```python
class QueryRouter:
    """Route queries to optimal retrieval strategy"""
    
    def __init__(self, retrievers: dict):
        self.retrievers = retrievers
    
    def route_and_retrieve(self, query: str):
        """Route query to best strategy"""
        
        # Analyze query
        analysis = self._analyze_query(query)
        
        # Select strategy
        strategy = self._select_strategy(analysis)
        
        # Retrieve using selected strategy
        retriever = self.retrievers[strategy]
        results = retriever.retrieve(query)
        
        return {
            'strategy': strategy,
            'results': results,
            'analysis': analysis
        }
    
    def _analyze_query(self, query: str):
        """Analyze query characteristics"""
        
        analysis = {
            'length': len(query.split()),
            'has_entities': self._has_entities(query),
            'specificity': self._estimate_specificity(query),
            'complexity': self._estimate_complexity(query)
        }
        
        return analysis
    
    def _select_strategy(self, analysis: dict):
        """Select best strategy for query"""
        
        # Short, specific queries -> lexical
        if analysis['length'] < 5 and analysis['specificity'] > 0.7:
            return 'lexical'
        
        # Long, complex queries -> semantic
        elif analysis['length'] > 15 or analysis['complexity'] > 0.7:
            return 'semantic'
        
        # Default -> hybrid
        else:
            return 'hybrid'
    
    def _has_entities(self, query: str):
        """Check for named entities"""
        from transformers import pipeline
        ner = pipeline("ner")
        entities = ner(query)
        return len(entities) > 0
    
    def _estimate_specificity(self, query: str):
        """Estimate query specificity"""
        # Heuristic: presence of specific terms
        specific_words = ['exactly', 'specifically', 'precisely', 'the']
        count = sum(1 for word in specific_words if word in query.lower())
        return min(count / 2, 1.0)
    
    def _estimate_complexity(self, query: str):
        """Estimate query complexity"""
        # Heuristic: query length and logical operators
        complexity = len(query.split()) / 20
        if any(op in query for op in ['and', 'or', 'not']):
            complexity += 0.2
        return min(complexity, 1.0)
```

## 3. Result Fusion

### Combining Results from Multiple Sources

```python
from scipy.stats import rankdata

class ResultFusion:
    """Fuse results from multiple retrieval systems"""
    
    @staticmethod
    def reciprocal_rank_fusion(result_lists: list, top_k: int = 10):
        """RRF: Reciprocal Rank Fusion"""
        
        scores = {}
        
        for results in result_lists:
            for rank, result in enumerate(results, 1):
                doc_id = result['id']
                
                # RRF score
                rrf_score = 1.0 / (60 + rank)
                
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += rrf_score
        
        # Rank by RRF score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return ranked
    
    @staticmethod
    def weighted_sum_fusion(result_lists: list, weights: list, top_k: int = 10):
        """Weighted sum of scores"""
        
        if len(weights) != len(result_lists):
            raise ValueError("Weights must match result lists")
        
        scores = {}
        
        for results, weight in zip(result_lists, weights):
            # Normalize scores
            max_score = max([r['score'] for r in results]) if results else 1
            
            for result in results:
                doc_id = result['id']
                normalized = result['score'] / max_score if max_score > 0 else 0
                
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += weight * normalized
        
        # Rank by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return ranked
    
    @staticmethod
    def learned_fusion(result_lists: list, model, top_k: int = 10):
        """Use ML model to fuse results"""
        
        # Create feature vectors for each document
        feature_vectors = {}
        
        for list_idx, results in enumerate(result_lists):
            for rank, result in enumerate(results):
                doc_id = result['id']
                
                if doc_id not in feature_vectors:
                    feature_vectors[doc_id] = {}
                
                feature_vectors[doc_id][f'list{list_idx}_rank'] = rank
                feature_vectors[doc_id][f'list{list_idx}_score'] = result['score']
        
        # Use model to predict final score
        final_scores = {}
        
        for doc_id, features in feature_vectors.items():
            score = model.predict([list(features.values())])[0]
            final_scores[doc_id] = score
        
        # Rank by predicted score
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return ranked
```

## 4. Diversity-Aware Hybrid

### Balancing Relevance and Diversity

```python
from sklearn.metrics.pairwise import cosine_similarity

class DiverseHybridRetrieval:
    """Hybrid retrieval with diversity optimization"""
    
    def __init__(self, retriever, embedding_model):
        self.retriever = retriever
        self.embedding_model = embedding_model
    
    def retrieve_diverse(self, query: str, top_k: int = 10, 
                        diversity_ratio: float = 0.3):
        """Retrieve with diversity consideration"""
        
        # Get initial results
        candidates = self.retriever.retrieve(query, top_k=top_k*3)
        
        # Select diverse subset
        selected = self._select_diverse(candidates, top_k, diversity_ratio)
        
        return selected
    
    def _select_diverse(self, candidates: list, top_k: int, diversity_ratio: float):
        """Select top-k with diversity constraint"""
        
        # Get embeddings for diversity measurement
        embeddings = [
            self.embedding_model.encode(c['text'][:500])
            for c in candidates
        ]
        
        selected = []
        selected_embeddings = []
        
        for i, candidate in enumerate(candidates):
            if len(selected) == 0:
                # Always select first result
                selected.append(candidate)
                selected_embeddings.append(embeddings[i])
            else:
                # Compute diversity score
                diversity_score = min([
                    1 - np.dot(embeddings[i], emb)
                    for emb in selected_embeddings
                ])
                
                # Combined score: relevance + diversity
                relevance_weight = 1 - diversity_ratio
                diversity_weight = diversity_ratio
                
                combined_score = (
                    relevance_weight * candidate['score'] +
                    diversity_weight * diversity_score
                )
                
                # Select if better than worst selected
                worst_selected_score = min([
                    (1 - diversity_ratio) * s['score'] + 0
                    for s in selected
                ])
                
                if combined_score > worst_selected_score or len(selected) < top_k:
                    selected.append(candidate)
                    selected_embeddings.append(embeddings[i])
                    
                    # Keep top-k
                    if len(selected) > top_k:
                        # Remove least diverse
                        selected = selected[:-1]
                        selected_embeddings = selected_embeddings[:-1]
        
        return selected[:top_k]
```

## 5. Cross-Lingual Hybrid

### Handling Multiple Languages

```python
class CrossLingualHybrid:
    """Hybrid retrieval for multilingual queries"""
    
    def __init__(self, dense_retriever, sparse_retriever, translator=None):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.translator = translator
    
    def retrieve_cross_lingual(self, query: str, query_language: str, top_k: int = 10):
        """Retrieve across languages"""
        
        # Retrieve in original language
        results_original = self.retrieve_hybrid(query, top_k=top_k)
        
        # If translation available, try other languages
        if self.translator:
            other_languages = ['en', 'es', 'fr', 'de']
            other_languages.remove(query_language)
            
            for lang in other_languages[:2]:  # Try 2 other languages
                translated_query = self.translator.translate(
                    query, query_language, lang
                )
                
                # Retrieve in other language
                results_translated = self.retrieve_hybrid(
                    translated_query, top_k=top_k
                )
                
                results_original.extend(results_translated)
        
        # Deduplicate and rerank
        unique_results = {}
        for result in results_original:
            doc_id = result['id']
            if doc_id not in unique_results:
                unique_results[doc_id] = result
        
        ranked = sorted(
            unique_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        return ranked
    
    def retrieve_hybrid(self, query: str, top_k: int = 10):
        """Standard hybrid retrieval"""
        dense_results = self.dense.retrieve(query, top_k=top_k*2)
        sparse_results = self.sparse.retrieve(query, top_k=top_k*2)
        
        # Combine using RRF
        return ResultFusion.reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=top_k
        )
```

## Best Practices

1. **Use RRF for fusion** - robust to different score scales
2. **Route complex queries** - semantic better for questions, lexical for keywords
3. **Consider diversity** - reduce result redundancy
4. **Monitor hybrid effectiveness** - measure accuracy gain vs complexity
5. **A/B test strategies** - validate improvements on real traffic

## Conclusion

Hybrid search combining semantic and lexical retrieval provides 10-20% accuracy improvement over single methods. Strategic routing and result fusion maximize benefits while managing complexity. Most production RAG systems use hybrid retrieval as default.
