# Retrieval Evaluation Metrics: Comprehensive Assessment

## Introduction

Measuring retrieval quality is crucial for optimizing RAG systems. This guide covers standard metrics, their interpretation, and practical implementation.

## 1. Core Retrieval Metrics

### Precision and Recall

```python
from typing import List, Set
import numpy as np

class RetrievalEvaluator:
    """Evaluate retrieval quality"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int = 10):
        """Precision@k: % of top-k that are relevant"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & relevant)
        
        if not retrieved_k:
            return 0
        
        return relevant_retrieved / len(retrieved_k)
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 10):
        """Recall@k: % of relevant docs in top-k"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & relevant)
        
        if not relevant:
            return 0
        
        return relevant_retrieved / len(relevant)
    
    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: Set[str], k: int = 10):
        """F1 score: harmonic mean of precision and recall"""
        p = RetrievalEvaluator.precision_at_k(retrieved, relevant, k)
        r = RetrievalEvaluator.recall_at_k(retrieved, relevant, k)
        
        if p + r == 0:
            return 0
        
        return 2 * (p * r) / (p + r)
    
    def evaluate_retrieval(self, retrieved: List[str], relevant: Set[str]):
        """Comprehensive retrieval evaluation"""
        metrics = {}
        
        for k in [1, 5, 10, 20]:
            metrics[f'precision@{k}'] = self.precision_at_k(retrieved, relevant, k)
            metrics[f'recall@{k}'] = self.recall_at_k(retrieved, relevant, k)
            metrics[f'f1@{k}'] = self.f1_at_k(retrieved, relevant, k)
        
        return metrics
```

### Mean Reciprocal Rank (MRR)

```python
class MRRCalculator:
    """Mean Reciprocal Rank"""
    
    @staticmethod
    def mrr(retrieved: List[str], relevant: Set[str]):
        """MRR: 1 / rank of first relevant item"""
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / rank
        
        return 0  # No relevant items found
    
    @staticmethod
    def mean_mrr(queries_results: List[tuple]):
        """Average MRR across queries"""
        mrr_scores = []
        
        for retrieved, relevant in queries_results:
            mrr = MRRCalculator.mrr(retrieved, relevant)
            mrr_scores.append(mrr)
        
        return np.mean(mrr_scores) if mrr_scores else 0
```

## 2. Ranking Metrics

### NDCG (Normalized Discounted Cumulative Gain)

```python
class NDCGCalculator:
    """NDCG - accounts for ranking quality"""
    
    @staticmethod
    def dcg(retrieved: List[str], relevances: dict, k: int = 10):
        """
        DCG: sum of (relevance / log2(rank + 1))
        """
        dcg_sum = 0
        
        for rank, doc_id in enumerate(retrieved[:k], 1):
            relevance = relevances.get(doc_id, 0)
            dcg_sum += relevance / np.log2(rank + 1)
        
        return dcg_sum
    
    @staticmethod
    def idcg(relevances: dict, k: int = 10):
        """
        IDCG: ideal DCG (best possible ranking)
        """
        sorted_relevances = sorted(relevances.values(), reverse=True)
        
        idcg_sum = 0
        for rank, rel in enumerate(sorted_relevances[:k], 1):
            idcg_sum += rel / np.log2(rank + 1)
        
        return idcg_sum
    
    @staticmethod
    def ndcg(retrieved: List[str], relevances: dict, k: int = 10):
        """NDCG@k: DCG / IDCG"""
        dcg = NDCGCalculator.dcg(retrieved, relevances, k)
        idcg = NDCGCalculator.idcg(relevances, k)
        
        if idcg == 0:
            return 0
        
        return dcg / idcg
```

### MAP (Mean Average Precision)

```python
class MAPCalculator:
    """Mean Average Precision"""
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]):
        """
        AP: average precision at each relevant item
        """
        precisions = []
        
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                precision = RetrievalEvaluator.precision_at_k(
                    retrieved,
                    relevant,
                    rank
                )
                precisions.append(precision)
        
        if not precisions:
            return 0
        
        return np.mean(precisions)
    
    @staticmethod
    def mean_ap(queries_results: List[tuple]):
        """MAP: average of AP across queries"""
        ap_scores = []
        
        for retrieved, relevant in queries_results:
            ap = MAPCalculator.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0
```

## 3. Relevance Assessment Methods

### Graded Relevance

```python
class GradedRelevanceAssessor:
    """Handle graded (multi-level) relevance"""
    
    def __init__(self):
        self.relevance_levels = {
            'not_relevant': 0,
            'somewhat_relevant': 1,
            'relevant': 2,
            'highly_relevant': 3
        }
    
    def compute_relevance_score(self, doc_id: str, query: str, assessor=None):
        """
        Compute relevance score (0-3)
        """
        if assessor:
            return assessor(doc_id, query)
        
        # Default: use embedding similarity
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('intfloat/e5-large')
        
        query_emb = model.encode(query)
        doc_emb = model.encode(doc_id)  # Simplified: assuming doc_id is text
        
        similarity = np.dot(query_emb, doc_emb)
        
        # Map to relevance level
        if similarity > 0.8:
            return 3
        elif similarity > 0.6:
            return 2
        elif similarity > 0.4:
            return 1
        else:
            return 0
    
    def create_relevance_dict(self, doc_ids: List[str], query: str):
        """Create relevance score dict for all documents"""
        relevances = {}
        
        for doc_id in doc_ids:
            relevances[doc_id] = self.compute_relevance_score(doc_id, query)
        
        return relevances
```

## 4. Cross-Validation and Statistical Testing

### K-Fold Cross-Validation

```python
class RetrievalCrossValidator:
    """Cross-validate retrieval metrics"""
    
    def __init__(self, retriever, k_folds: int = 5):
        self.retriever = retriever
        self.k_folds = k_folds
    
    def k_fold_evaluation(self, queries: List[str], relevance_judgments: dict):
        """
        Evaluate using k-fold cross-validation
        """
        fold_size = len(queries) // self.k_folds
        fold_results = []
        
        for fold in range(self.k_folds):
            # Split into train/test
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size
            
            test_queries = queries[start_idx:end_idx]
            
            # Evaluate on test set
            fold_metrics = self._evaluate_fold(test_queries, relevance_judgments)
            fold_results.append(fold_metrics)
        
        # Aggregate results
        aggregated = self._aggregate_fold_results(fold_results)
        
        return aggregated
    
    def _evaluate_fold(self, test_queries: List[str], judgments: dict):
        """Evaluate single fold"""
        metrics = {}
        
        for query in test_queries:
            retrieved = self.retriever.retrieve(query)
            relevant = set(judgments.get(query, []))
            
            query_metrics = {
                'precision@10': RetrievalEvaluator.precision_at_k(
                    retrieved,
                    relevant,
                    10
                ),
                'mrr': MRRCalculator.mrr(retrieved, relevant),
                'ndcg@10': NDCGCalculator.ndcg(
                    retrieved,
                    {doc_id: 1 for doc_id in relevant},
                    10
                )
            }
            
            for key, value in query_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        return {key: np.mean(values) for key, values in metrics.items()}
    
    def _aggregate_fold_results(self, fold_results: List[dict]):
        """Aggregate across folds"""
        aggregated = {}
        
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated
```

## 5. Significance Testing

### Statistical Significance

```python
from scipy import stats

class SignificanceTestor:
    """Test if improvements are statistically significant"""
    
    @staticmethod
    def paired_t_test(baseline_scores: List[float], improved_scores: List[float]):
        """
        Paired t-test for dependent samples
        """
        t_stat, p_value = stats.ttest_rel(improved_scores, baseline_scores)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }
    
    @staticmethod
    def mann_whitney_u_test(baseline: List[float], improved: List[float]):
        """
        Mann-Whitney U test for independent samples
        """
        u_stat, p_value = stats.mannwhitneyu(baseline, improved)
        
        return {
            'u_statistic': u_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05
        }
    
    @staticmethod
    def effect_size(baseline: List[float], improved: List[float]):
        """
        Calculate Cohen's d effect size
        """
        mean_diff = np.mean(improved) - np.mean(baseline)
        
        n1, n2 = len(baseline), len(improved)
        var1, var2 = np.var(baseline, ddof=1), np.var(improved, ddof=1)
        
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'cohens_d': cohens_d,
            'effect_size': 'negligible' if abs(cohens_d) < 0.2 else
                          'small' if abs(cohens_d) < 0.5 else
                          'medium' if abs(cohens_d) < 0.8 else
                          'large'
        }
```

## 6. Real-World Evaluation

### BEIR Benchmark

```python
class BEIREvaluator:
    """Evaluate using BEIR benchmark"""
    
    def __init__(self):
        self.datasets = [
            'trec-covid',
            'nfcorpus',
            'nq',
            'dbpedia-entity',
            'trec-covid',
            'scifact'
        ]
    
    def evaluate_on_beir(self, retriever, dataset_name: str = 'trec-covid'):
        """
        Evaluate retriever on BEIR dataset
        """
        # Load BEIR dataset (simplified)
        queries, corpus, qrels = self._load_beir_dataset(dataset_name)
        
        # Evaluate
        metrics = {}
        
        for metric_name in ['ndcg@10', 'mrr@10', 'map@100']:
            scores = []
            
            for query_id, query in queries.items():
                retrieved_docs = retriever.retrieve(query, top_k=100)
                retrieved_ids = [doc['id'] for doc in retrieved_docs]
                
                relevant_ids = set(qrels.get(query_id, {}).keys())
                
                if metric_name == 'ndcg@10':
                    score = NDCGCalculator.ndcg(
                        retrieved_ids,
                        qrels.get(query_id, {}),
                        10
                    )
                elif metric_name == 'mrr@10':
                    score = MRRCalculator.mrr(retrieved_ids[:10], relevant_ids)
                elif metric_name == 'map@100':
                    score = MAPCalculator.average_precision(retrieved_ids, relevant_ids)
                
                scores.append(score)
            
            metrics[metric_name] = np.mean(scores)
        
        return metrics
    
    def _load_beir_dataset(self, dataset_name: str):
        """Load BEIR dataset"""
        # Simplified - in practice use beir library
        return {}, {}, {}
```

## 7. Metric Selection Guide

```
Metric          When to Use                     Pros                    Cons
─────────────────────────────────────────────────────────────────────────────
Precision@k     User satisfaction focused       Simple, intuitive        Ignores ranking quality
Recall@k        Completeness important         Measures coverage        Ignores ranking
NDCG@k          Ranking quality matters        Accounts for ranking    Complex to explain
MRR             First result critical          Simple, interpretable    Ignores other results
MAP             Average quality across         Balanced metric         Requires judgments
F1@k            Balance precision/recall       Comprehensive          Can mask issues
```

## Best Practices

1. **Use multiple metrics** - single metric can mislead
2. **Test on diverse queries** - easy vs hard queries need both
3. **Statistical significance** - p-value required for claims
4. **Benchmark against baseline** - show improvement not absolute performance
5. **Use graded relevance** - more nuanced than binary
6. **NDCG preferred for ranking** - best reflects user satisfaction

## Conclusion

Comprehensive evaluation requires multiple metrics and statistical rigor. NDCG@10 and MRR@10 are good defaults. Always compare against baseline and test statistical significance. Use k-fold cross-validation to avoid overfitting to specific queries.
