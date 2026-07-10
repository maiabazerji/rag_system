# Few-Shot Learning with RAG: In-Context Examples

## Introduction

Few-shot prompting significantly improves LLM performance. This guide covers strategies for effectively using examples in RAG systems.

## 1. Few-Shot Fundamentals

### Theory and Impact

```python
import json
from typing import List, Dict

class FewShotSelector:
    """Select optimal examples for few-shot prompting"""
    
    def __init__(self, embedding_model, example_pool: List[Dict]):
        self.embedding_model = embedding_model
        self.example_pool = example_pool
        self.pool_embeddings = self._embed_examples()
    
    def _embed_examples(self):
        """Pre-compute embeddings for all examples"""
        embeddings = {}
        
        for example in self.example_pool:
            # Embed the input/query part
            emb = self.embedding_model.encode(example['input'])
            example_id = example.get('id', hash(example['input']))
            embeddings[example_id] = emb
        
        return embeddings
    
    def select_examples(self, query: str, num_examples: int = 3):
        """Select most relevant examples for query"""
        query_emb = self.embedding_model.encode(query)
        
        # Compute similarity to all examples
        similarities = []
        
        for example_id, example_emb in self.pool_embeddings.items():
            similarity = np.dot(query_emb, example_emb)
            similarities.append((example_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top examples
        selected_ids = [s[0] for s in similarities[:num_examples]]
        
        selected_examples = [
            ex for ex in self.example_pool
            if ex.get('id', hash(ex['input'])) in selected_ids
        ]
        
        return selected_examples
```

## 2. Example Composition

### Creating Effective Examples

```python
class ExampleComposer:
    """Create well-structured few-shot examples"""
    
    @staticmethod
    def create_qa_example(question: str, context: str, answer: str, 
                         explanation: str = None):
        """Create Q&A example"""
        return {
            'input': question,
            'context': context,
            'output': answer,
            'explanation': explanation,
            'type': 'qa'
        }
    
    @staticmethod
    def create_classification_example(text: str, category: str, reasoning: str = None):
        """Create classification example"""
        return {
            'input': text,
            'output': category,
            'reasoning': reasoning,
            'type': 'classification'
        }
    
    @staticmethod
    def create_extraction_example(text: str, entities: List[Dict], format_spec: str = None):
        """Create extraction example"""
        return {
            'input': text,
            'output': json.dumps(entities),
            'format': format_spec,
            'type': 'extraction'
        }
    
    @staticmethod
    def format_examples_for_prompt(examples: List[Dict], template: str = None):
        """Format examples for inclusion in prompt"""
        if template is None:
            template = "Example {num}:\nInput: {input}\nOutput: {output}\n"
        
        formatted = []
        for i, example in enumerate(examples):
            formatted.append(template.format(num=i+1, **example))
        
        return "\n".join(formatted)
```

## 3. Example Selection Strategies

### Similarity-Based Selection

```python
class SimilarityBasedSelector:
    """Select examples most similar to current query"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def select_similar_examples(self, query: str, example_pool: List[Dict], 
                               top_k: int = 3):
        """Select most similar examples"""
        query_emb = self.embedding_model.encode(query)
        
        scores = []
        for ex in example_pool:
            ex_emb = self.embedding_model.encode(ex['input'])
            similarity = np.dot(query_emb, ex_emb)
            scores.append((ex, similarity))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:top_k]]
```

### Diversity-Based Selection

```python
class DiversityBasedSelector:
    """Select diverse examples to cover broader patterns"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def select_diverse_examples(self, query: str, example_pool: List[Dict], 
                               num_examples: int = 3):
        """Select diverse set of examples"""
        # First, select initial candidate (most similar to query)
        query_emb = self.embedding_model.encode(query)
        
        selected = []
        remaining = example_pool.copy()
        
        # Greedy selection: pick most similar, then most different from selected
        while len(selected) < num_examples and remaining:
            if not selected:
                # Pick most similar to query
                scores = [
                    (ex, np.dot(query_emb, self.embedding_model.encode(ex['input'])))
                    for ex in remaining
                ]
            else:
                # Pick most different from selected ones
                diversity_scores = []
                
                for ex in remaining:
                    ex_emb = self.embedding_model.encode(ex['input'])
                    
                    # Distance to selected examples
                    min_distance = min([
                        np.linalg.norm(ex_emb - self.embedding_model.encode(sel['input']))
                        for sel in selected
                    ])
                    
                    diversity_scores.append((ex, min_distance))
                
                scores = diversity_scores
            
            best_ex = max(scores, key=lambda x: x[1])[0]
            selected.append(best_ex)
            remaining.remove(best_ex)
        
        return selected
```

## 4. Contrastive Examples

### Negative and Hard Negative Examples

```python
class ContrastiveExampleSelector:
    """Include negative/contrastive examples"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def select_contrastive_examples(self, query: str, positive_examples: List[Dict],
                                    negative_pool: List[Dict], num_negatives: int = 1):
        """Select positive and negative examples"""
        # Positive examples (most similar)
        query_emb = self.embedding_model.encode(query)
        
        positive_scores = [
            (ex, np.dot(query_emb, self.embedding_model.encode(ex['input'])))
            for ex in positive_examples
        ]
        positive_scores.sort(key=lambda x: x[1], reverse=True)
        selected_positives = [s[0] for s in positive_scores[:len(positive_examples)]]
        
        # Negative examples (least similar, but still somewhat relevant)
        negative_scores = [
            (ex, np.dot(query_emb, self.embedding_model.encode(ex['input'])))
            for ex in negative_pool
        ]
        negative_scores.sort(key=lambda x: x[1])
        
        # Select negatives that are in mid-range (not completely dissimilar)
        selected_negatives = [
            s[0] for s in negative_scores[len(negative_scores)//3:len(negative_scores)//3 + num_negatives]
        ]
        
        return {
            'positive_examples': selected_positives,
            'negative_examples': selected_negatives
        }
    
    def format_contrastive_prompt(self, query: str, examples: Dict):
        """Format prompt with contrastive examples"""
        prompt = f"Query: {query}\n\n"
        
        prompt += "CORRECT EXAMPLES:\n"
        for ex in examples['positive_examples']:
            prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
        
        prompt += "INCORRECT EXAMPLES (what NOT to do):\n"
        for ex in examples['negative_examples']:
            prompt += f"Input: {ex['input']}\nOutput (WRONG): {ex['output']}\n\n"
        
        prompt += "Now answer the query correctly."
        
        return prompt
```

## 5. Dynamic Example Selection

### Adaptive Selection Based on Performance

```python
class AdaptiveExampleSelector:
    """Adapt example selection based on performance"""
    
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
        self.performance_history = {}
    
    def select_adaptive_examples(self, query: str, example_pool: List[Dict], 
                                num_examples: int = 3):
        """Select examples adaptively based on historical performance"""
        
        # Check if we've seen similar queries before
        similar_past_queries = self._find_similar_queries(query)
        
        if similar_past_queries:
            # Use examples that worked well for similar queries
            best_examples = self._get_examples_by_performance(
                similar_past_queries,
                num_examples
            )
            return best_examples
        
        else:
            # Fall back to similarity-based selection
            return self._select_by_similarity(query, example_pool, num_examples)
    
    def record_performance(self, query: str, selected_examples: List[Dict], 
                          output_quality: float):
        """Record how well selected examples performed"""
        query_hash = hash(query)
        
        if query_hash not in self.performance_history:
            self.performance_history[query_hash] = {
                'query': query,
                'attempts': []
            }
        
        self.performance_history[query_hash]['attempts'].append({
            'examples': [ex.get('id') for ex in selected_examples],
            'quality': output_quality,
            'timestamp': datetime.now()
        })
    
    def _find_similar_queries(self, query: str, num_similar: int = 3):
        """Find similar queries from history"""
        if not self.performance_history:
            return []
        
        query_emb = self.embedding_model.encode(query)
        
        similarities = []
        for entry in self.performance_history.values():
            hist_emb = self.embedding_model.encode(entry['query'])
            sim = np.dot(query_emb, hist_emb)
            similarities.append((entry['query'], sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in similarities[:num_similar]]
    
    def _get_examples_by_performance(self, queries: List[str], num_examples: int):
        """Get examples that performed well"""
        example_scores = {}
        
        for query in queries:
            query_hash = hash(query)
            if query_hash in self.performance_history:
                for attempt in self.performance_history[query_hash]['attempts']:
                    for ex_id in attempt['examples']:
                        if ex_id not in example_scores:
                            example_scores[ex_id] = []
                        example_scores[ex_id].append(attempt['quality'])
        
        # Score examples by average performance
        ranked = sorted(
            example_scores.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )
        
        return [ex_id for ex_id, _ in ranked[:num_examples]]
```

## 6. Few-Shot RAG Integration

### Complete Few-Shot RAG Pipeline

```python
class FewShotRAG:
    """RAG system with few-shot enhancement"""
    
    def __init__(self, retriever, embedding_model, llm, example_pool: List[Dict]):
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.llm = llm
        self.example_selector = SimilarityBasedSelector(embedding_model)
        self.example_pool = example_pool
    
    def generate_with_few_shot(self, query: str, num_examples: int = 3):
        """Generate with few-shot examples"""
        
        # 1. Select examples
        examples = self.example_selector.select_similar_examples(
            query,
            self.example_pool,
            num_examples
        )
        
        # 2. Retrieve relevant documents
        docs = self.retriever.retrieve(query)
        context = self._format_context(docs)
        
        # 3. Build prompt with examples
        prompt = self._build_few_shot_prompt(query, examples, context)
        
        # 4. Generate
        response = self.llm.generate(prompt)
        
        return response
    
    def _format_context(self, docs):
        """Format retrieved documents"""
        context_parts = []
        for doc in docs[:3]:
            context_parts.append(f"- {doc['text'][:200]}")
        
        return "\n".join(context_parts)
    
    def _build_few_shot_prompt(self, query: str, examples: List[Dict], context: str):
        """Build prompt with few-shot examples"""
        prompt = "EXAMPLES:\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n"
        
        prompt += f"\nCONTEXT:\n{context}\n"
        prompt += f"\nQUESTION: {query}\n"
        prompt += "ANSWER:"
        
        return prompt
```

## 7. Measuring Few-Shot Impact

### Evaluation Framework

```python
class FewShotEvaluator:
    """Evaluate impact of few-shot examples"""
    
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
    
    def compare_with_without_examples(self, query: str, examples: List[Dict],
                                     test_queries: List[str]):
        """Compare performance with and without few-shot"""
        
        results = {
            'without_few_shot': [],
            'with_few_shot': []
        }
        
        for test_query in test_queries:
            # Without examples
            prompt_without = f"Question: {test_query}\nAnswer:"
            response_without = self.llm.generate(prompt_without)
            
            quality_without = self._evaluate_response(
                response_without,
                test_query
            )
            results['without_few_shot'].append(quality_without)
            
            # With examples
            prompt_with = self._build_few_shot_prompt(test_query, examples)
            response_with = self.llm.generate(prompt_with)
            
            quality_with = self._evaluate_response(
                response_with,
                test_query
            )
            results['with_few_shot'].append(quality_with)
        
        return {
            'without_examples': np.mean(results['without_few_shot']),
            'with_examples': np.mean(results['with_few_shot']),
            'improvement': (
                np.mean(results['with_few_shot']) - 
                np.mean(results['without_few_shot'])
            )
        }
    
    def _evaluate_response(self, response: str, query: str):
        """Evaluate response quality"""
        # Simplified: check if response addresses query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        overlap = len(query_words & response_words) / len(query_words)
        
        return overlap
    
    def _build_few_shot_prompt(self, query: str, examples: List[Dict]):
        """Build few-shot prompt"""
        prompt = "EXAMPLES:\n"
        for ex in examples:
            prompt += f"Q: {ex['input']}\nA: {ex['output']}\n\n"
        
        prompt += f"Q: {query}\nA:"
        return prompt
```

## Best Practices

1. **Select examples similar to target query** - semantic similarity works well
2. **Include diverse examples** - cover different patterns
3. **Use high-quality examples** - model learns from example quality
4. **Keep examples concise** - long examples waste tokens
5. **Include explanations** - helps model understand reasoning
6. **Use contrastive examples** - show what NOT to do
7. **Adapt dynamically** - track what works for your use case

## Conclusion

Few-shot prompting provides 5-15% quality improvement with minimal token overhead. Effective example selection is crucial - semantic similarity is a good starting point. Combine with retrieval for best results.
