# Query Optimization: Techniques for Better Retrieval

## Introduction

Query optimization improves retrieval precision and recall through intelligent query processing. This guide covers techniques for transforming raw queries into optimized search signals.

## 1. Query Understanding

### Intent Classification

```python
from enum import Enum

class QueryIntent(Enum):
    FACTUAL = "factual"  # "What is...?" "When did...?"
    COMPARATIVE = "comparative"  # "Compare X vs Y"
    PROCEDURAL = "procedural"  # "How to...?"
    EXPLORATORY = "exploratory"  # "Tell me about..."
    NAVIGATIONAL = "navigational"  # "Find document about..."
    OPINION = "opinion"  # "What do you think...?"

class QueryUnderstandingEngine:
    def __init__(self, intent_classifier):
        self.classifier = intent_classifier
    
    def classify_intent(self, query: str):
        """Classify query intent"""
        return self.classifier.predict(query)
    
    def extract_entities(self, query: str):
        """Extract named entities from query"""
        # Use NER model
        return self.ner_model.extract(query)
    
    def identify_query_type(self, query: str):
        """Determine query type and structure"""
        if "vs" in query.lower() or "compare" in query.lower():
            return "comparative"
        elif query.endswith("?"):
            return "question"
        elif any(word in query.lower() for word in ["how to", "how do"]):
            return "procedural"
        else:
            return "statement"
```

## 2. Query Expansion

### Synonym Expansion

```python
class QueryExpander:
    def __init__(self, synonym_db=None):
        self.synonym_db = synonym_db or self._load_synonym_db()
    
    def expand_with_synonyms(self, query: str):
        """Expand query with synonyms"""
        expanded_queries = [query]
        words = query.split()
        
        for word in words:
            synonyms = self.synonym_db.get(word.lower(), [])
            
            for syn in synonyms[:2]:  # Limit expansions
                new_query = query.replace(word, syn)
                expanded_queries.append(new_query)
        
        return expanded_queries
    
    def expand_with_hypernyms(self, query: str):
        """Expand with more general terms"""
        # "dog" -> "animal", "pet", "mammal"
        expanded = [query]
        
        for word in query.split():
            hypernyms = self._get_hypernyms(word)
            for hypernym in hypernyms[:1]:
                new_query = query.replace(word, hypernym)
                expanded.append(new_query)
        
        return expanded
    
    def expand_with_hyponyms(self, query: str):
        """Expand with more specific terms"""
        # "animal" -> "dog", "cat", "bird"
        expanded = [query]
        
        for word in query.split():
            hyponyms = self._get_hyponyms(word)
            for hyponym in hyponyms[:2]:
                new_query = query.replace(word, hyponym)
                expanded.append(new_query)
        
        return expanded
```

### LLM-Based Expansion

```python
class LLMQueryExpander:
    def __init__(self, llm):
        self.llm = llm
    
    def expand_query(self, query: str, num_expansions=3):
        """Generate query variations using LLM"""
        prompt = f"""Generate {num_expansions} alternative ways to ask this question:
        Original: {query}
        
        Requirements:
        - Each alternative should have the same intent
        - Use different vocabulary and phrasing
        - Keep them concise
        
        Format: One alternative per line"""
        
        expansions = self.llm.generate(prompt, num_variants=num_expansions)
        return [query] + expansions
    
    def clarify_query(self, query: str):
        """Add implicit context to clarify query"""
        prompt = f"""The user asked: {query}
        
        What clarifications or additional context would make this question more specific?
        Provide 2 clarified versions that maintain the original intent."""
        
        clarifications = self.llm.generate(prompt, num_variants=2)
        return [query] + clarifications
    
    def decompose_complex_query(self, query: str):
        """Break down complex queries into sub-queries"""
        prompt = f"""For this question: {query}
        
        Break it down into simpler sub-questions that together answer the original.
        Each sub-question should be independent and precise."""
        
        subqueries = self.llm.generate(prompt, num_variants=1)
        return subqueries.split('\n')
```

## 3. Query Rewriting

### Keyword Optimization

```python
class QueryRewriter:
    def __init__(self, llm=None):
        self.llm = llm
    
    def remove_stop_words(self, query: str):
        """Remove common words that add no meaning"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 
            'was', 'were', 'be', 'have', 'has', 'had', 'do', 
            'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = query.split()
        filtered = [w for w in words if w.lower() not in stop_words]
        
        return ' '.join(filtered)
    
    def normalize_query(self, query: str):
        """Normalize query for better matching"""
        # Lowercase
        query = query.lower()
        
        # Remove punctuation
        import string
        query = query.translate(
            str.maketrans('', '', string.punctuation)
        )
        
        # Remove extra spaces
        query = ' '.join(query.split())
        
        return query
    
    def rewrite_for_retrieval(self, query: str):
        """Rewrite query to optimize for document retrieval"""
        if self.llm:
            prompt = f"""Rewrite this query to be more suitable for document search:
            Original: {query}
            
            Guidelines:
            - Add relevant keywords
            - Remove ambiguous pronouns
            - Make implicit concepts explicit
            - Use document-friendly vocabulary"""
            
            rewritten = self.llm.generate(prompt)
            return rewritten
        else:
            # Rule-based rewriting
            return self.remove_stop_words(self.normalize_query(query))
```

### Structural Query Rewriting

```python
class StructuredQueryBuilder:
    """Build structured queries from natural language"""
    
    def __init__(self, schema=None):
        self.schema = schema
    
    def parse_date_range(self, query: str):
        """Extract date ranges from query"""
        import re
        from datetime import datetime
        
        # Patterns: "from 2020 to 2024", "between Jan and Dec 2023"
        date_pattern = r'(\d{4}|\w+ \d{1,2})'
        dates = re.findall(date_pattern, query)
        
        return dates
    
    def parse_filters(self, query: str):
        """Extract filter conditions"""
        filters = {}
        
        # Category filter: "in category X"
        if "in category" in query:
            cat = query.split("in category")[-1].strip()
            filters['category'] = cat.split()[0]
        
        # Date filter: "from X to Y"
        dates = self.parse_date_range(query)
        if dates:
            filters['date_range'] = dates
        
        # Author filter: "by author X"
        if "by author" in query:
            author = query.split("by author")[-1].strip()
            filters['author'] = author.split()[0]
        
        return filters
    
    def build_structured_query(self, query: str):
        """Convert to structured query format"""
        filters = self.parse_filters(query)
        
        # Remove filter keywords from query
        cleaned_query = query
        for keyword in ["in category", "by author", "from", "to"]:
            cleaned_query = cleaned_query.replace(keyword, "")
        
        return {
            'text_query': cleaned_query.strip(),
            'filters': filters
        }
```

## 4. Query Routing

```python
class QueryRouter:
    """Route queries to appropriate retrieval strategy"""
    
    def __init__(self, retriever_configs):
        self.retriever_configs = retriever_configs
    
    def route_query(self, query: str):
        """Determine best retrieval strategy"""
        # Analyze query characteristics
        length = len(query.split())
        has_entities = self._has_named_entities(query)
        is_factual = self._is_factual_query(query)
        complexity = self._estimate_complexity(query)
        
        # Route based on characteristics
        if length < 10 and has_entities:
            return 'keyword_search'  # Short, entity-focused
        elif complexity > 0.7:
            return 'semantic_multi_hop'  # Complex, needs reasoning
        elif is_factual:
            return 'semantic_hybrid'  # Mix semantic + keyword
        else:
            return 'semantic_dense'  # Pure semantic
    
    def _has_named_entities(self, query: str):
        """Check if query contains named entities"""
        # Simplified check
        important_words = query.split()
        capitalized = sum(1 for w in important_words if w[0].isupper())
        return capitalized > 0
    
    def _is_factual_query(self, query: str):
        """Check if query is factual (not opinion)"""
        opinion_words = ['think', 'believe', 'opinion', 'prefer']
        return not any(w in query.lower() for w in opinion_words)
    
    def _estimate_complexity(self, query: str):
        """Estimate query complexity (0-1)"""
        # More words = more complex
        return min(len(query.split()) / 20, 1.0)
```

## 5. Query Weighting and Scoring

```python
class QueryTermWeighter:
    """Weight query terms by importance"""
    
    def __init__(self, idf_scores=None):
        self.idf_scores = idf_scores or {}
    
    def compute_term_weights(self, query: str):
        """Compute weight for each query term"""
        words = query.split()
        weights = {}
        
        for word in words:
            # TF-IDF weighting
            term_freq = words.count(word)
            idf = self.idf_scores.get(word.lower(), 1.0)
            
            weights[word] = term_freq * idf
        
        # Normalize
        max_weight = max(weights.values()) if weights else 1
        return {k: v/max_weight for k, v in weights.items()}
    
    def reweight_retrieval_results(self, query: str, documents):
        """Boost scores for documents matching important terms"""
        term_weights = self.compute_term_weights(query)
        
        reweighted = []
        for doc in documents:
            score = doc['score']
            
            # Boost for important term matches
            for term, weight in term_weights.items():
                if term.lower() in doc['text'].lower():
                    score *= (1 + weight * 0.1)
            
            doc['reweighted_score'] = score
            reweighted.append(doc)
        
        return sorted(reweighted, key=lambda x: x['reweighted_score'], reverse=True)
```

## 6. Query-Document Matching

```python
class AdvancedQueryMatcher:
    """Advanced query-document matching strategies"""
    
    def __init__(self, embedding_model, cross_encoder):
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder
    
    def bidirectional_matching(self, query: str, documents):
        """Match query->doc and doc->query"""
        # Generate query-based match queries
        generated_queries = self._generate_matching_queries(query)
        
        # For each document, check if it would retrieve the query
        scores = []
        for doc in documents:
            # Direct match: query -> doc
            direct_score = self.embedding_model.similarity(query, doc['text'])
            
            # Reverse match: would doc retrieve this query?
            reverse_score = max([
                self.embedding_model.similarity(gq, doc['text'])
                for gq in generated_queries
            ])
            
            # Combine
            combined = 0.6 * direct_score + 0.4 * reverse_score
            scores.append((doc, combined))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def _generate_matching_queries(self, document: str):
        """Generate queries that would match this document"""
        # Simplified: extract key phrases
        important_phrases = self._extract_key_phrases(document)
        return important_phrases[:3]
    
    def _extract_key_phrases(self, text: str):
        """Extract key phrases from text"""
        # Use TF-IDF or embedding-based extraction
        words = text.split()
        # Return most important words
        return words[:5]
```

## 7. Batch Query Optimization

```python
class BatchQueryOptimizer:
    """Optimize multiple queries efficiently"""
    
    def __init__(self, retriever, cache=None):
        self.retriever = retriever
        self.cache = cache or {}
    
    def optimize_queries(self, queries: list):
        """Optimize batch of queries"""
        # Deduplicate similar queries
        unique_queries = self._deduplicate_queries(queries)
        
        # Expand only unique queries
        expanded = {}
        for q in unique_queries:
            expanded[q] = self._expand_query(q)
        
        # Map back to original
        optimized = {}
        for original in queries:
            canonical = self._find_canonical(original, unique_queries)
            optimized[original] = expanded[canonical]
        
        return optimized
    
    def _deduplicate_queries(self, queries: list):
        """Find similar queries"""
        unique = []
        for q in queries:
            if not any(self._are_similar(q, uq) for uq in unique):
                unique.append(q)
        return unique
    
    def _are_similar(self, q1: str, q2: str, threshold=0.9):
        """Check if queries are similar"""
        sim = self.retriever.embedding_model.similarity(q1, q2)
        return sim > threshold
    
    def _expand_query(self, query: str):
        """Expand single query"""
        if query in self.cache:
            return self.cache[query]
        
        expanded = [query]
        # Add expansion logic
        
        self.cache[query] = expanded
        return expanded
```

## 8. Evaluation

```python
class QueryOptimizationEvaluator:
    """Evaluate query optimization effectiveness"""
    
    @staticmethod
    def evaluate_optimization(original_queries, optimized_queries, retriever, ground_truth):
        """Compare optimization impact"""
        metrics = {
            'original_precision': 0,
            'optimized_precision': 0,
            'improvement': 0
        }
        
        # Evaluate original queries
        orig_results = [retriever.retrieve(q) for q in original_queries]
        metrics['original_precision'] = QueryOptimizationEvaluator._compute_precision(
            orig_results, ground_truth
        )
        
        # Evaluate optimized queries
        opt_results = [retriever.retrieve(q) for q in optimized_queries]
        metrics['optimized_precision'] = QueryOptimizationEvaluator._compute_precision(
            opt_results, ground_truth
        )
        
        metrics['improvement'] = (
            metrics['optimized_precision'] - metrics['original_precision']
        ) / (metrics['original_precision'] + 1e-6)
        
        return metrics
    
    @staticmethod
    def _compute_precision(results, ground_truth):
        """Compute precision of retrieval results"""
        correct = sum(1 for result in results if result in ground_truth)
        return correct / len(results) if results else 0
```

## Best Practices

1. **Start with query normalization** - remove noise before processing
2. **Use intent classification** to select retrieval strategy
3. **Implement query expansion carefully** - balance coverage vs noise
4. **Route complex queries differently** - multi-hop retrieval for reasoning tasks
5. **Cache frequently asked questions** - avoid redundant optimization
6. **Monitor optimization impact** - measure precision gains

## Conclusion

Query optimization is an underrated lever for RAG performance. Simple techniques like normalization and expansion can improve retrieval by 15-25%. More sophisticated routing and rewriting provides additional gains, especially for complex queries. Always measure impact empirically on your use case.
