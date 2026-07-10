# Advanced RAG Techniques

## Introduction

Retrieval-Augmented Generation (RAG) has evolved from a simple retrieval + generation pipeline into a sophisticated system requiring careful orchestration of multiple components. This document explores advanced techniques that push RAG beyond basic implementations, enabling better accuracy, efficiency, and scalability for production systems.

## 1. Iterative Refinement and Multi-Hop Retrieval

Traditional RAG performs a single retrieval round, but complex questions often require multiple reasoning steps. Iterative refinement enables the LLM to generate intermediate queries or refine its understanding based on retrieved context.

### Multi-Hop Reasoning

Multi-hop retrieval breaks down complex questions into sub-questions:

```python
class MultiHopRetriever:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.max_hops = 3
    
    def retrieve(self, query: str, context=""):
        all_docs = []
        current_context = context
        
        for hop in range(self.max_hops):
            # Generate sub-queries based on current context
            sub_queries = self.llm.generate_sub_queries(
                query, 
                current_context,
                num_queries=2
            )
            
            # Retrieve for each sub-query
            for sub_query in sub_queries:
                docs = self.retriever.retrieve(sub_query, top_k=3)
                all_docs.extend(docs)
            
            # Update context for next iteration
            current_context = self._summarize_docs(all_docs)
            
            # Early stopping if sufficient information found
            if self._has_sufficient_info(current_context, query):
                break
        
        return self._deduplicate_docs(all_docs)
    
    def _summarize_docs(self, docs):
        return self.llm.summarize([d.content for d in docs])
    
    def _has_sufficient_info(self, context, query):
        score = self.llm.compute_relevance(context, query)
        return score > 0.8
    
    def _deduplicate_docs(self, docs):
        seen = set()
        unique = []
        for doc in docs:
            doc_id = doc.metadata.get('id', doc.content)
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
        return unique
```

### Real-World Example

For the query "What were the key financial impacts of COVID-19 on tech companies?":
1. First hop: Retrieve general COVID-19 economic impact documents
2. Second hop: Based on findings, retrieve specific tech sector impacts
3. Third hop: Retrieve financial metrics (revenue, stock performance) for specific companies

This approach improves answer quality by 23-35% for multi-faceted questions compared to single-round retrieval.

## 2. Reranking and Cross-Encoder Fusion

Initial retrieval often returns false positives. Reranking refines the candidate set using more sophisticated models.

### Cross-Encoder Reranking Strategy

```python
from sentence_transformers import CrossEncoder
import numpy as np

class HybridReranker:
    def __init__(self, bi_encoder, cross_encoder):
        self.bi_encoder = bi_encoder
        self.cross_encoder = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-12-v2'
        )
    
    def rerank(self, query: str, docs, top_k=10):
        # Initial ranking with bi-encoder
        query_emb = self.bi_encoder.encode(query)
        doc_scores = []
        
        for doc in docs:
            doc_emb = self.bi_encoder.encode(doc.content)
            score = np.dot(query_emb, doc_emb)
            doc_scores.append((doc, score))
        
        # Keep top candidates for expensive reranking
        candidates = sorted(
            doc_scores, 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k * 3]
        
        # Cross-encoder reranking (more accurate but expensive)
        pairs = [(query, doc.content) for doc, _ in candidates]
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Combine scores (weighted average)
        final_scores = []
        for (doc, bi_score), cross_score in zip(candidates, cross_scores):
            combined = 0.3 * bi_score + 0.7 * cross_score
            final_scores.append((doc, combined))
        
        return sorted(
            final_scores, 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
```

**Performance Benchmarks:**
- Single bi-encoder: 75% NDCG@10
- Bi-encoder + cross-encoder: 84% NDCG@10
- Computational cost: 3-5x increase in latency
- Recommended for cases where accuracy > speed

## 3. Query Transformation and Expansion

Raw user queries are often suboptimal for retrieval. Query transformation improves search signal.

### Query Expansion Techniques

```python
class QueryExpander:
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder
    
    def expand_query(self, query: str, method='diverse'):
        if method == 'diverse':
            return self._diverse_expansion(query)
        elif method == 'clarification':
            return self._clarification_expansion(query)
        elif method == 'reasoning':
            return self._reasoning_expansion(query)
    
    def _diverse_expansion(self, query):
        """Generate diverse reformulations"""
        prompt = f"""Generate 3 alternative ways to ask this question:
        Original: {query}
        
        Provide concise alternatives that preserve meaning but use different vocabulary."""
        
        expansions = self.llm.generate(prompt, num_variants=3)
        return [query] + expansions
    
    def _clarification_expansion(self, query):
        """Add implicit context and clarifications"""
        prompt = f"""The user asked: {query}
        
        What implicit context or clarifications would make this question more precise?
        Provide 2 clarified versions."""
        
        clarifications = self.llm.generate(prompt, num_variants=2)
        return [query] + clarifications
    
    def _reasoning_expansion(self, query):
        """Break down complex queries"""
        prompt = f"""For the question: {query}
        
        What intermediate knowledge or sub-questions would help answer this?
        Generate 2 reformulations that address these aspects."""
        
        sub_queries = self.llm.generate(prompt, num_variants=2)
        return [query] + sub_queries
    
    def rank_expansions(self, query, expansions, documents):
        """Score expansions by retrieval quality"""
        scores = []
        
        for expansion in expansions:
            # Compute how well this expansion matches available documents
            expansion_emb = self.embedder.encode(expansion)
            avg_similarity = self._compute_avg_similarity(
                expansion_emb, 
                documents
            )
            scores.append(avg_similarity)
        
        return sorted(
            zip(expansions, scores),
            key=lambda x: x[1],
            reverse=True
        )
```

**Impact on Retrieval:**
- Query expansion improves recall by 15-20%
- Diversity-based expansion: +18% recall, minimal latency increase
- Reasoning-based: +25% recall, requires LLM call (100-200ms overhead)
- Best practice: Use for complex queries only, cache results

## 4. Adaptive Retrieval and Early Stopping

Not all questions require extensive retrieval. Adaptive systems determine optimal retrieval depth.

```python
class AdaptiveRetriever:
    def __init__(self, base_retriever, confidence_model):
        self.retriever = base_retriever
        self.confidence_model = confidence_model
    
    def retrieve_adaptive(self, query: str, max_docs=50):
        """Retrieve with early stopping based on confidence"""
        all_docs = []
        batch_size = 10
        confidence_scores = []
        
        for batch_num in range(0, max_docs, batch_size):
            # Retrieve next batch
            docs = self.retriever.retrieve(
                query,
                top_k=batch_size,
                offset=batch_num
            )
            all_docs.extend(docs)
            
            # Evaluate confidence with current set
            confidence = self.confidence_model.evaluate(
                query,
                all_docs
            )
            confidence_scores.append(confidence)
            
            # Stop if confidence plateaued
            if len(confidence_scores) > 2:
                improvement = (
                    confidence_scores[-1] - confidence_scores[-2]
                )
                if improvement < 0.02:  # < 2% improvement
                    break
        
        return all_docs
```

**Efficiency Gains:**
- Average reduction in documents retrieved: 40-50%
- Latency improvement: 30-45% for well-calibrated models
- Accuracy impact: <1% degradation with good confidence model

## 5. Context-Aware Retrieval

Retrieval context significantly impacts relevance. Temporal, geographical, and semantic context improve precision.

```python
class ContextAwareRetriever:
    def __init__(self, base_retriever):
        self.retriever = base_retriever
    
    def retrieve_with_context(self, query, user_context):
        """Retrieve considering user context"""
        # Extract temporal context
        temporal = user_context.get('time_period', None)
        
        # Extract geographical context
        location = user_context.get('location', None)
        
        # Extract domain context
        domain = user_context.get('domain', None)
        
        # Build context filter
        filters = self._build_filters(temporal, location, domain)
        
        # Retrieve with context-aware filtering
        docs = self.retriever.retrieve(
            query,
            filters=filters,
            context_boost=user_context.get('emphasis', {})
        )
        
        # Re-score based on context alignment
        reweighted = self._reweight_by_context(docs, user_context)
        
        return reweighted
    
    def _build_filters(self, temporal, location, domain):
        filters = {}
        
        if temporal:
            filters['date_range'] = temporal
        
        if location:
            filters['location'] = location
        
        if domain:
            filters['domain'] = domain
        
        return filters
    
    def _reweight_by_context(self, docs, context):
        """Apply context-based weighting"""
        weighted_docs = []
        
        for doc in docs:
            score = doc.similarity_score
            
            # Boost if domain matches
            if context.get('domain') == doc.metadata.get('domain'):
                score *= 1.2
            
            # Boost if temporal relevance
            if self._is_temporally_relevant(doc, context):
                score *= 1.15
            
            # Boost if geographic relevance
            if self._is_geographically_relevant(doc, context):
                score *= 1.1
            
            weighted_docs.append((doc, score))
        
        return sorted(weighted_docs, key=lambda x: x[1], reverse=True)
```

## 6. Fusion and Ensemble Methods

Combining multiple retrieval strategies improves robustness and coverage.

```python
class EnsembleRetriever:
    def __init__(self, retrievers: list):
        self.retrievers = retrievers  # Different retrieval strategies
    
    def retrieve_ensemble(self, query: str, top_k=10):
        """Combine results from multiple retrievers"""
        all_results = {}
        scores_by_retriever = {}
        
        # Get results from each retriever
        for name, retriever in self.retrievers:
            docs = retriever.retrieve(query, top_k=top_k*2)
            scores_by_retriever[name] = {}
            
            for rank, doc in enumerate(docs):
                doc_id = doc.metadata.get('id')
                
                # RRF (Reciprocal Rank Fusion) scoring
                rrf_score = 1.0 / (60 + rank)
                
                if doc_id not in all_results:
                    all_results[doc_id] = doc
                    scores_by_retriever[name][doc_id] = rrf_score
                else:
                    scores_by_retriever[name][doc_id] = rrf_score
        
        # Aggregate scores
        final_scores = {}
        for retriever_name, retriever_scores in scores_by_retriever.items():
            weight = self._get_retriever_weight(retriever_name)
            
            for doc_id, score in retriever_scores.items():
                final_scores[doc_id] = final_scores.get(doc_id, 0)
                final_scores[doc_id] += weight * score
        
        # Return top-k by combined score
        sorted_results = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [all_results[doc_id] for doc_id, _ in sorted_results]
    
    def _get_retriever_weight(self, retriever_name):
        """Adaptive weighting based on performance"""
        weights = {
            'semantic': 0.4,
            'lexical': 0.3,
            'dense': 0.2,
            'sparse': 0.1
        }
        return weights.get(retriever_name, 0.25)
```

## Trade-offs and Considerations

| Technique | Accuracy Gain | Latency Impact | Complexity | Cost |
|-----------|---------------|----------------|------------|------|
| Multi-hop | +25-35% | 2-3x | High | Medium |
| Cross-encoder Reranking | +8-12% | 2-5x | Medium | Medium |
| Query Expansion | +15-20% | 1.5-2x | Medium | Low |
| Adaptive Retrieval | 0% | -40% | Medium | Low |
| Ensemble Methods | +10-15% | 2-4x | High | High |

## Best Practices

1. **Start Simple**: Begin with reranking before multi-hop or ensembles
2. **Measure Carefully**: Use offline metrics (NDCG, MAP) and online metrics (user satisfaction)
3. **Monitor Latency**: Advanced techniques compound latency; cache aggressively
4. **Context is Key**: Always consider user and temporal context
5. **Hybrid Approach**: Combine lightweight (query expansion) with expensive (cross-encoder) techniques

## Conclusion

Advanced RAG techniques significantly improve retrieval quality but require careful tuning. The best approach combines multiple techniques strategically, measuring impact at each stage and optimizing for your specific use case's accuracy-latency-cost trade-off.
