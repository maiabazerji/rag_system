# Reranking Deep Dive: Advanced Ranking Techniques

## Introduction

Reranking refines initial retrieval results using more sophisticated models. This guide covers reranking architectures, strategies, and implementation patterns.

## 1. Reranking Fundamentals

### Why Reranking Matters

```python
class ReRankingBenchmark:
    """Demonstrate impact of reranking"""
    
    @staticmethod
    def compare_with_without_reranking(query: str, retrieved_docs: list, 
                                       reranker, ground_truth: set):
        """Compare performance with and without reranking"""
        
        # Without reranking
        top_10_without = retrieved_docs[:10]
        precision_without = len(
            {d['id'] for d in top_10_without} & ground_truth
        ) / 10
        
        # With reranking
        reranked = reranker.rerank(query, retrieved_docs, top_k=10)
        precision_with = len(
            {d['id'] for d in reranked} & ground_truth
        ) / 10
        
        improvement = (precision_with - precision_without) / max(precision_without, 0.01)
        
        return {
            'precision_without_reranking': precision_without,
            'precision_with_reranking': precision_with,
            'improvement_percent': improvement * 100
        }
```

## 2. Reranking Architectures

### Cross-Encoder Architecture

```python
from sentence_transformers import CrossEncoder
import numpy as np

class CrossEncoderReranker:
    """Rerank using cross-encoder"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: list, top_k: int = 10):
        """Rerank documents using cross-encoder"""
        
        if not documents:
            return []
        
        # Prepare pairs
        pairs = [[query, doc['text']] for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Combine scores with documents
        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_docs.append({
                **doc,
                'reranking_score': float(score)
            })
        
        # Sort by reranking score
        reranked = sorted(
            scored_docs,
            key=lambda x: x['reranking_score'],
            reverse=True
        )
        
        return reranked[:top_k]
    
    def rerank_batch(self, query: str, document_batches: list, top_k: int = 10):
        """Rerank multiple batches efficiently"""
        all_scores = []
        
        for batch in document_batches:
            pairs = [[query, doc['text']] for doc in batch]
            batch_scores = self.model.predict(pairs)
            all_scores.extend(batch_scores)
        
        # Flatten and score
        all_docs = []
        for batch, batch_idx in zip(document_batches, range(len(document_batches))):
            start_idx = batch_idx * len(batch)
            for doc_idx, doc in enumerate(batch):
                all_docs.append({
                    **doc,
                    'reranking_score': float(all_scores[start_idx + doc_idx])
                })
        
        reranked = sorted(
            all_docs,
            key=lambda x: x['reranking_score'],
            reverse=True
        )
        
        return reranked[:top_k]
```

### Listwise Reranking

```python
class ListwiseReranker:
    """Rerank considering all documents together"""
    
    def __init__(self, model):
        self.model = model
    
    def rerank_listwise(self, query: str, documents: list, top_k: int = 10):
        """
        Rerank considering ranking of all documents
        More computationally expensive but higher quality
        """
        
        if len(documents) <= top_k:
            return documents
        
        # Score all documents
        query_doc_scores = []
        for doc in documents:
            # Cross-encoder score
            score = self.model.predict([[query, doc['text']]])[0]
            query_doc_scores.append((doc, score))
        
        # Sort
        sorted_docs = sorted(query_doc_scores, key=lambda x: x[1], reverse=True)
        
        # Consider ranking quality (normalized discount cumulative gain)
        ndcg = self._compute_listwise_ndcg(sorted_docs[:top_k])
        
        return [doc for doc, _ in sorted_docs[:top_k]]
    
    def _compute_listwise_ndcg(self, ranked_docs: list):
        """Compute NDCG for reranked list"""
        dcg = 0
        for rank, (doc, score) in enumerate(ranked_docs, 1):
            dcg += score / np.log2(rank + 1)
        
        return dcg
```

## 3. Hybrid Reranking

### Combining Multiple Signals

```python
class HybridReranker:
    """Combine multiple reranking signals"""
    
    def __init__(self, cross_encoder, lexical_scorer=None, diversity_scorer=None):
        self.cross_encoder = cross_encoder
        self.lexical_scorer = lexical_scorer
        self.diversity_scorer = diversity_scorer
    
    def rerank_hybrid(self, query: str, documents: list, top_k: int = 10):
        """Rerank using multiple signals"""
        
        # Score 1: Cross-encoder relevance
        pairs = [[query, doc['text']] for doc in documents]
        cross_encoder_scores = self.cross_encoder.predict(pairs)
        
        # Score 2: Lexical match (BM25 or similar)
        lexical_scores = None
        if self.lexical_scorer:
            lexical_scores = [
                self.lexical_scorer.score(query, doc['text'])
                for doc in documents
            ]
        
        # Score 3: Diversity (penalize similar documents)
        diversity_scores = None
        if self.diversity_scorer:
            diversity_scores = self.diversity_scorer.score_diversity(documents)
        
        # Normalize and combine
        final_scores = []
        
        for i, doc in enumerate(documents):
            score = cross_encoder_scores[i] * 0.6  # 60% relevance
            
            if lexical_scores:
                score += lexical_scores[i] * 0.2  # 20% lexical
            
            if diversity_scores:
                score += diversity_scores[i] * 0.2  # 20% diversity
            
            final_scores.append((doc, score))
        
        # Rank and return
        reranked = sorted(final_scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[:top_k]]
```

## 4. Context-Aware Reranking

### Considering User/Domain Context

```python
class ContextAwareReranker:
    """Rerank considering context"""
    
    def __init__(self, cross_encoder):
        self.cross_encoder = cross_encoder
    
    def rerank_with_context(self, query: str, documents: list, 
                           user_context: dict, top_k: int = 10):
        """Rerank considering user context"""
        
        # Enhance query with context
        augmented_query = self._augment_query_with_context(query, user_context)
        
        # Rerank with augmented query
        pairs = [[augmented_query, doc['text']] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        # Bonus for domain/category match
        final_scores = []
        for doc, score in zip(documents, scores):
            final_score = score
            
            # Boost if doc is in user's preferred domain
            if user_context.get('preferred_domain') == doc.get('domain'):
                final_score *= 1.2
            
            # Penalize if explicitly excluded domain
            if user_context.get('excluded_domain') == doc.get('domain'):
                final_score *= 0.5
            
            final_scores.append((doc, final_score))
        
        reranked = sorted(final_scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[:top_k]]
    
    def _augment_query_with_context(self, query: str, context: dict):
        """Add context to query"""
        augmented = query
        
        if context.get('domain'):
            augmented = f"{augmented} in {context['domain']}"
        
        if context.get('intent'):
            augmented = f"{augmented} {context['intent']}"
        
        return augmented
```

## 5. Position Bias Handling

### Correcting Retriever Bias

```python
class PositionBiasCorrector:
    """Correct for position bias in initial retrieval"""
    
    def __init__(self, cross_encoder):
        self.cross_encoder = cross_encoder
        self.position_weights = None
    
    def estimate_position_bias(self, queries: list, ground_truth: dict):
        """Estimate position bias from training data"""
        
        position_counts = {}
        position_correct = {}
        
        for query in queries:
            results = self.initial_retriever.retrieve(query)
            
            for rank, result in enumerate(results):
                if rank not in position_counts:
                    position_counts[rank] = 0
                    position_correct[rank] = 0
                
                position_counts[rank] += 1
                
                if result['id'] in ground_truth.get(query, set()):
                    position_correct[rank] += 1
        
        # Compute position bias
        self.position_weights = {}
        for rank in position_counts:
            correct_rate = position_correct[rank] / position_counts[rank]
            self.position_weights[rank] = 1.0 / max(correct_rate, 0.1)
    
    def rerank_with_bias_correction(self, query: str, documents: list, top_k: int = 10):
        """Rerank correcting for position bias"""
        
        # Get cross-encoder scores
        pairs = [[query, doc['text']] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        # Apply position bias correction
        corrected_scores = []
        
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # Correct for position bias
            position_weight = self.position_weights.get(i, 1.0)
            corrected_score = score * position_weight
            
            corrected_scores.append((doc, corrected_score))
        
        reranked = sorted(corrected_scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[:top_k]]
```

## 6. Efficient Reranking

### Two-Stage Reranking

```python
class EfficientReranker:
    """Rerank efficiently with two stages"""
    
    def __init__(self, bi_encoder, cross_encoder):
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
    
    def rerank_two_stage(self, query: str, documents: list, 
                        stage1_k: int = 50, stage2_k: int = 10):
        """
        Stage 1: Fast bi-encoder screening
        Stage 2: Accurate cross-encoder reranking
        """
        
        # Stage 1: Fast filtering with bi-encoder
        query_emb = self.bi_encoder.encode(query)
        
        stage1_scores = []
        for doc in documents:
            doc_emb = self.bi_encoder.encode(doc['text'][:200])
            score = np.dot(query_emb, doc_emb)
            stage1_scores.append((doc, score))
        
        # Keep top documents
        stage1_ranking = sorted(stage1_scores, key=lambda x: x[1], reverse=True)
        candidates = [doc for doc, _ in stage1_ranking[:stage1_k]]
        
        # Stage 2: Accurate ranking with cross-encoder
        pairs = [[query, doc['text']] for doc in candidates]
        cross_encoder_scores = self.cross_encoder.predict(pairs)
        
        stage2_ranking = []
        for doc, score in zip(candidates, cross_encoder_scores):
            stage2_ranking.append((doc, score))
        
        stage2_ranking.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in stage2_ranking[:stage2_k]]
```

## 7. Reranking Evaluation

### Measuring Reranking Quality

```python
class ReRankingEvaluator:
    """Evaluate reranking quality"""
    
    @staticmethod
    def evaluate_reranking(query: str, initial_ranking: list, 
                          reranked: list, ground_truth: set):
        """Compare rankings"""
        
        initial_ids = [d['id'] for d in initial_ranking[:10]]
        reranked_ids = [d['id'] for d in reranked[:10]]
        
        # Metrics
        initial_precision = len(set(initial_ids) & ground_truth) / 10
        reranked_precision = len(set(reranked_ids) & ground_truth) / 10
        
        # Check if ground truth moved up
        first_relevant_initial = None
        first_relevant_reranked = None
        
        for rank, doc_id in enumerate(initial_ids):
            if doc_id in ground_truth:
                first_relevant_initial = rank
                break
        
        for rank, doc_id in enumerate(reranked_ids):
            if doc_id in ground_truth:
                first_relevant_reranked = rank
                break
        
        return {
            'initial_precision@10': initial_precision,
            'reranked_precision@10': reranked_precision,
            'precision_improvement': reranked_precision - initial_precision,
            'first_relevant_rank_improvement': (
                (first_relevant_initial or 10) - (first_relevant_reranked or 10)
            )
        }
```

## Best Practices

1. **Use cross-encoder for high accuracy** - more expensive but better quality
2. **Implement two-stage reranking** - balance accuracy and latency
3. **Consider diversity** - penalize redundant documents
4. **Account for context** - user preferences matter
5. **Measure P@10** - most important metric for reranking
6. **Cache reranking scores** - expensive to recompute

## Conclusion

Reranking with cross-encoders provides 8-15% precision improvement over raw retrieval. Two-stage (bi-encoder + cross-encoder) balances quality and latency. Most gains come from correcting retriever errors; pure reranking has limits.
