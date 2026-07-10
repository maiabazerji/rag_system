# Debugging RAG Failures: Systematic Problem Diagnosis

## Introduction

RAG systems fail in many ways: missing relevant documents, poor ranking, hallucination, irrelevant context. This guide provides systematic approaches to debugging and fixing RAG failures.

## 1. RAG Failure Taxonomy

### Common Failure Modes

```python
from enum import Enum
from typing import Optional

class FailureMode(Enum):
    # Retrieval failures
    NO_RELEVANT_DOCS = "retriever"
    LOW_RANK_RELEVANT = "ranker"
    CONTRADICTORY_DOCS = "corpus_quality"
    
    # Generation failures
    HALLUCINATION = "generation"
    CONTEXT_IGNORED = "generation"
    REASONING_ERROR = "generation"
    OUTDATED_INFO = "corpus_age"
    
    # System failures
    LATENCY_SPIKE = "performance"
    OUT_OF_MEMORY = "resources"
    QUALITY_REGRESSION = "regression"

class FailureAnalyzer:
    def diagnose_failure(self, query: str, expected_answer: str, actual_response: str):
        """Diagnose RAG failure"""
        diagnostics = {
            'retrieved_docs': self._check_retrieval(query),
            'context_quality': self._check_context_relevance(query, actual_response),
            'hallucination_score': self._detect_hallucination(actual_response),
            'reasoning_quality': self._evaluate_reasoning(actual_response),
            'latency': self._measure_latency(query)
        }
        
        return self._determine_root_cause(diagnostics)
```

## 2. Retrieval Debugging

### Tracing Retrieval Pipeline

```python
class RetrievalDebugger:
    def __init__(self, retriever, embedding_model):
        self.retriever = retriever
        self.embedding_model = embedding_model
    
    def debug_retrieval(self, query: str, expected_docs: list):
        """Debug retrieval failure"""
        debug_report = {}
        
        # Step 1: Query encoding
        query_emb = self.embedding_model.encode(query)
        debug_report['query_embedding_norm'] = np.linalg.norm(query_emb)
        
        # Step 2: Raw retrieval
        raw_results = self.retriever.retrieve_raw(query_emb, top_k=100)
        debug_report['raw_retrieval'] = {
            'count': len(raw_results),
            'top_scores': [r['score'] for r in raw_results[:5]],
            'expected_found_at': self._find_expected(raw_results, expected_docs)
        }
        
        # Step 3: Score distribution
        scores = [r['score'] for r in raw_results]
        debug_report['score_distribution'] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'percentile_50': np.percentile(scores, 50),
            'percentile_90': np.percentile(scores, 90)
        }
        
        # Step 4: Embedding similarity analysis
        for doc in expected_docs[:3]:
            doc_emb = self.embedding_model.encode(doc['text'])
            similarity = np.dot(query_emb, doc_emb)
            debug_report[f"similarity_to_expected_{doc['id']}"] = similarity
        
        return debug_report
    
    def _find_expected(self, results, expected_docs):
        """Find rank of expected documents"""
        expected_ids = {d['id'] for d in expected_docs}
        
        for rank, result in enumerate(results):
            if result['doc_id'] in expected_ids:
                return rank
        
        return None
```

### Visualization of Retrieval Performance

```python
import matplotlib.pyplot as plt

class RetrievalVisualizer:
    @staticmethod
    def plot_score_distribution(raw_results, expected_doc_rank):
        """Visualize score distribution"""
        scores = [r['score'] for r in raw_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(scores, bins=50, edgecolor='black')
        ax1.axvline(scores[0], color='red', linestyle='--', label='Best score')
        if expected_doc_rank is not None:
            ax1.axvline(scores[expected_doc_rank], color='green', linestyle='--', label='Expected doc')
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.set_title('Score Distribution')
        
        # Ranking
        ax2.plot(scores[:50], 'b-o')
        if expected_doc_rank is not None and expected_doc_rank < 50:
            ax2.scatter([expected_doc_rank], [scores[expected_doc_rank]], color='green', s=100, marker='X')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Similarity Score')
        ax2.set_title('Score by Rank (Top 50)')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_embedding_space(query_emb, doc_embs, doc_labels, expected_docs):
        """Project embeddings to 2D for visualization"""
        from sklearn.decomposition import PCA
        
        all_embs = np.vstack([query_emb, doc_embs])
        
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_embs)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot documents
        colors = ['red' if label in expected_docs else 'blue' for label in doc_labels]
        ax.scatter(projected[1:, 0], projected[1:, 1], c=colors, alpha=0.6, s=50)
        
        # Plot query
        ax.scatter(projected[0, 0], projected[0, 1], c='green', s=200, marker='X', label='Query')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Embedding Space (2D Projection)')
        ax.legend()
        
        return fig
```

## 3. Context Quality Analysis

### Measuring Context Relevance

```python
class ContextQualityAnalyzer:
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
    
    def analyze_context_quality(self, query: str, retrieved_docs: list):
        """Analyze quality of retrieved context"""
        analysis = {}
        
        # 1. Coverage: Do docs cover main aspects?
        coverage = self._assess_coverage(query, retrieved_docs)
        analysis['coverage_score'] = coverage
        
        # 2. Coherence: Are all docs relevant to query?
        coherence = self._assess_coherence(query, retrieved_docs)
        analysis['coherence_score'] = coherence
        
        # 3. Redundancy: Any duplicate information?
        redundancy = self._assess_redundancy(retrieved_docs)
        analysis['redundancy_ratio'] = redundancy
        
        # 4. Contradictions: Do docs conflict?
        contradictions = self._detect_contradictions(retrieved_docs)
        analysis['has_contradictions'] = len(contradictions) > 0
        analysis['contradictions'] = contradictions
        
        # 5. Relevance distribution: Are scores realistic?
        relevance_dist = self._analyze_relevance_distribution(retrieved_docs)
        analysis['relevance_distribution'] = relevance_dist
        
        return analysis
    
    def _assess_coverage(self, query: str, docs: list):
        """Assess if docs answer all aspects of query"""
        query_aspects = self._extract_query_aspects(query)
        
        doc_text = ' '.join([d['text'] for d in docs])
        
        covered = sum(
            1 for aspect in query_aspects
            if aspect.lower() in doc_text.lower()
        )
        
        return covered / len(query_aspects) if query_aspects else 0
    
    def _assess_coherence(self, query: str, docs: list):
        """Assess how relevant each doc is to query"""
        query_emb = self.embedding_model.encode(query)
        
        coherence_scores = []
        for doc in docs:
            doc_emb = self.embedding_model.encode(doc['text'][:200])
            similarity = np.dot(query_emb, doc_emb)
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0
    
    def _assess_redundancy(self, docs: list):
        """Assess information redundancy"""
        doc_embs = self.embedding_model.encode(
            [d['text'][:200] for d in docs]
        )
        
        # Compute pairwise similarities
        similarities = cosine_similarity(doc_embs)
        
        # Get average off-diagonal similarity
        off_diag = similarities[~np.eye(len(similarities), dtype=bool)]
        redundancy = np.mean(off_diag) if len(off_diag) > 0 else 0
        
        return redundancy
    
    def _detect_contradictions(self, docs: list):
        """Detect contradictory statements"""
        contradictions = []
        
        for i, doc1 in enumerate(docs):
            for doc2 in docs[i+1:]:
                # Simple contradiction detection
                if self._are_contradictory(doc1['text'], doc2['text']):
                    contradictions.append({
                        'doc1_id': doc1.get('id'),
                        'doc2_id': doc2.get('id'),
                        'reason': 'Potential contradiction detected'
                    })
        
        return contradictions
    
    def _are_contradictory(self, text1: str, text2: str):
        """Check if two texts contradict"""
        # Use LLM to detect contradictions
        prompt = f"""Do these two passages contradict each other?
        
        Passage 1: {text1[:200]}
        
        Passage 2: {text2[:200]}
        
        Answer: Yes or No"""
        
        response = self.llm.generate(prompt)
        return 'yes' in response.lower()
    
    def _extract_query_aspects(self, query: str):
        """Extract main aspects of query"""
        # Simple: split on "and" and "or"
        aspects = query.split(' and ')
        return [a.strip() for a in aspects if a.strip()]
    
    def _analyze_relevance_distribution(self, docs: list):
        """Analyze distribution of relevance scores"""
        scores = [d.get('score', 0) for d in docs]
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'range': [np.min(scores), np.max(scores)],
            'num_low_confidence': sum(1 for s in scores if s < 0.5)
        }
```

## 4. Hallucination Detection

### Identifying False Information

```python
class HallucinationDetector:
    def __init__(self, embedding_model, nli_model=None):
        self.embedding_model = embedding_model
        self.nli_model = nli_model or self._init_nli()
    
    def detect_hallucination(self, response: str, context: str):
        """Detect hallucinated information"""
        hallucinations = []
        
        # Method 1: NLI-based detection
        nli_hallucinations = self._detect_via_nli(response, context)
        hallucinations.extend(nli_hallucinations)
        
        # Method 2: Information presence check
        missing_info = self._detect_unsupported_claims(response, context)
        hallucinations.extend(missing_info)
        
        # Method 3: Factuality check
        factual_issues = self._check_factuality(response)
        hallucinations.extend(factual_issues)
        
        return hallucinations
    
    def _detect_via_nli(self, response: str, context: str):
        """Use Natural Language Inference to detect contradictions"""
        # Split response into sentences
        response_sents = response.split('.')
        
        hallucinations = []
        
        for sent in response_sents[:5]:  # Check first 5 sentences
            if len(sent.strip()) < 10:
                continue
            
            # Check entailment with context
            prediction = self.nli_model(sent, context)
            
            if prediction['label'] == 'contradiction':
                hallucinations.append({
                    'type': 'nli_contradiction',
                    'sentence': sent,
                    'confidence': prediction['score']
                })
            elif prediction['label'] == 'neutral':
                hallucinations.append({
                    'type': 'unsupported',
                    'sentence': sent,
                    'confidence': prediction['score']
                })
        
        return hallucinations
    
    def _detect_unsupported_claims(self, response: str, context: str):
        """Detect claims not supported by context"""
        response_emb = self.embedding_model.encode(response)
        context_emb = self.embedding_model.encode(context)
        
        similarity = np.dot(response_emb, context_emb)
        
        unsupported = []
        
        if similarity < 0.5:
            unsupported.append({
                'type': 'low_grounding',
                'similarity': similarity,
                'issue': 'Response not well grounded in context'
            })
        
        return unsupported
    
    def _check_factuality(self, response: str):
        """Check factuality of response"""
        # In production, use fact-checking service
        factual_issues = []
        
        # Simple patterns
        numeric_claims = self._extract_numeric_claims(response)
        
        for claim in numeric_claims:
            # Verify numeric facts
            if not self._verify_fact(claim):
                factual_issues.append({
                    'type': 'factual_error',
                    'claim': claim
                })
        
        return factual_issues
    
    def _extract_numeric_claims(self, text: str):
        """Extract numeric facts from text"""
        import re
        
        patterns = [
            r'(\d+\s*%)',  # Percentages
            r'(\d+\s*(?:billion|million|thousand))',  # Large numbers
            r'(\d{4})',  # Years
        ]
        
        claims = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            claims.extend(matches)
        
        return claims
```

## 5. Regression Detection

### Quality Regression Monitoring

```python
class RegressionDetector:
    def __init__(self, baseline_metrics: dict):
        self.baseline = baseline_metrics
        self.current_metrics = {}
    
    def check_regression(self, new_metrics: dict, threshold: float = 0.05):
        """Check if performance regressed"""
        regressions = []
        
        for metric, new_value in new_metrics.items():
            if metric not in self.baseline:
                continue
            
            baseline_value = self.baseline[metric]
            
            # Calculate percent change
            if baseline_value != 0:
                pct_change = (new_value - baseline_value) / baseline_value
            else:
                pct_change = 0 if new_value == baseline_value else 1
            
            # Check threshold
            if pct_change < -threshold:
                regressions.append({
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': new_value,
                    'pct_change': pct_change
                })
        
        return regressions
    
    def compare_versions(self, model_v1, model_v2, test_queries: list):
        """Compare two model versions"""
        v1_metrics = self._evaluate_model(model_v1, test_queries)
        v2_metrics = self._evaluate_model(model_v2, test_queries)
        
        comparison = {}
        
        for metric in v1_metrics:
            comparison[metric] = {
                'v1': v1_metrics[metric],
                'v2': v2_metrics[metric],
                'improvement': (v2_metrics[metric] - v1_metrics[metric]) / v1_metrics[metric]
            }
        
        return comparison
```

## 6. Debugging Workflow

### Systematic Debugging Process

```python
class RAGDebugger:
    def __init__(self, rag_system):
        self.rag = rag_system
    
    def debug_failure(self, query: str, expected_answer: str):
        """Systematic debugging workflow"""
        debug_report = {
            'query': query,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Test retrieval
        retrieval_debug = self._debug_retrieval(query)
        debug_report['retrieval'] = retrieval_debug
        
        if not retrieval_debug['has_relevant_docs']:
            return debug_report  # Retrieval failure
        
        # Step 2: Test ranking
        ranking_debug = self._debug_ranking(query)
        debug_report['ranking'] = ranking_debug
        
        # Step 3: Test context quality
        context_debug = self._debug_context(query)
        debug_report['context'] = context_debug
        
        # Step 4: Test generation
        generation_debug = self._debug_generation(query)
        debug_report['generation'] = generation_debug
        
        # Step 5: Test for hallucination
        hallucination_debug = self._debug_hallucination(query)
        debug_report['hallucination'] = hallucination_debug
        
        # Produce recommendation
        debug_report['recommendation'] = self._produce_recommendation(debug_report)
        
        return debug_report
```

## Best Debugging Practices

1. **Isolate components** - test retrieval, ranking, generation separately
2. **Use visualizations** - embedding space plots reveal problems
3. **Compare to baseline** - track regressions systematically
4. **Profile latency** - identify bottlenecks
5. **Log everything** - maintain audit trail
6. **Version test cases** - reuse for regression testing
7. **Mock components** - substitute perfect retriever/generator to isolate issues

## Conclusion

Systematic debugging requires understanding the RAG pipeline deeply and instrumenting each component. Start with retrieval (most failures), then context quality, then generation. Always establish baseline metrics before deploying to production.
