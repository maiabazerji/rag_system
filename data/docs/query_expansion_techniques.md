# Query Expansion Techniques: Improving Retrieval Coverage

## Introduction

Query expansion generates alternative queries to improve retrieval recall and coverage. This guide covers techniques for expanding and transforming queries.

## 1. Query Expansion Methods

### Synonym-Based Expansion

```python
from typing import List, Set

class SynonymExpander:
    """Expand queries with synonyms"""
    
    def __init__(self, synonym_db=None):
        self.synonym_db = synonym_db or self._load_default_synonyms()
    
    def expand_with_synonyms(self, query: str, num_synonyms: int = 2):
        """Expand query with synonyms"""
        words = query.split()
        expanded_queries = [query]
        
        for word in words:
            synonyms = self.synonym_db.get(word.lower(), [])
            
            for syn in synonyms[:num_synonyms]:
                new_query = query.replace(word, syn)
                expanded_queries.append(new_query)
        
        return self._deduplicate(expanded_queries)
    
    def _load_default_synonyms(self):
        """Load basic synonym dictionary"""
        return {
            'car': ['automobile', 'vehicle', 'motor car'],
            'buy': ['purchase', 'acquire', 'obtain'],
            'big': ['large', 'huge', 'massive'],
            'small': ['tiny', 'little', 'compact'],
            'fast': ['quick', 'rapid', 'speedy'],
            'slow': ['gradual', 'sluggish', 'leisurely'],
        }
    
    def _deduplicate(self, queries: List[str]):
        """Remove duplicate queries"""
        return list(dict.fromkeys(queries))
```

### Hypernym/Hyponym Expansion

```python
class HypernymHyponymExpander:
    """Expand queries with more general/specific terms"""
    
    def __init__(self):
        self.word_hierarchy = self._build_hierarchy()
    
    def expand_with_hypernyms(self, query: str):
        """Generalize query with hypernyms (broader terms)"""
        expanded = [query]
        
        for word in query.split():
            hypernyms = self._get_hypernyms(word.lower())
            
            for hypernym in hypernyms[:1]:  # Use top hypernym
                new_query = query.replace(word, hypernym)
                expanded.append(new_query)
        
        return expanded
    
    def expand_with_hyponyms(self, query: str):
        """Specialize query with hyponyms (narrower terms)"""
        expanded = [query]
        
        for word in query.split():
            hyponyms = self._get_hyponyms(word.lower())
            
            for hyponym in hyponyms[:2]:  # Use top 2 hyponyms
                new_query = query.replace(word, hyponym)
                expanded.append(new_query)
        
        return expanded
    
    def _build_hierarchy(self):
        """Build simplified word hierarchy"""
        return {
            'dog': {'hypernym': 'animal', 'hyponym': ['labrador', 'poodle']},
            'animal': {'hypernym': 'organism', 'hyponym': ['dog', 'cat', 'bird']},
            'car': {'hypernym': 'vehicle', 'hyponym': ['sedan', 'suv', 'truck']},
            'vehicle': {'hypernym': 'transportation', 'hyponym': ['car', 'bus', 'train']},
        }
    
    def _get_hypernyms(self, word: str):
        """Get more general terms"""
        if word in self.word_hierarchy:
            hyper = self.word_hierarchy[word].get('hypernym')
            return [hyper] if hyper else []
        return []
    
    def _get_hyponyms(self, word: str):
        """Get more specific terms"""
        if word in self.word_hierarchy:
            return self.word_hierarchy[word].get('hyponym', [])
        return []
```

## 2. NLP-Based Expansion

### Paraphrase Generation

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class ParaphraseExpander:
    """Generate paraphrases of queries"""
    
    def __init__(self, model_name: str = "t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def generate_paraphrases(self, query: str, num_paraphrases: int = 3):
        """Generate paraphrases using seq2seq model"""
        
        input_text = f"paraphrase: {query}"
        encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        
        outputs = self.model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=256,
            num_beams=num_paraphrases,
            num_return_sequences=num_paraphrases,
            temperature=1.5
        )
        
        paraphrases = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return [query] + paraphrases
```

### Semantic Query Expansion

```python
class SemanticExpander:
    """Expand query semantically"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def find_semantically_similar_queries(self, query: str, query_pool: List[str]):
        """Find similar queries from pool"""
        query_emb = self.embedding_model.encode(query)
        
        similarities = []
        for pool_query in query_pool:
            pool_emb = self.embedding_model.encode(pool_query)
            sim = np.dot(query_emb, pool_emb)
            similarities.append((pool_query, sim))
        
        # Return top similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [q for q, _ in similarities[:3]]
    
    def generate_clarification_queries(self, query: str, llm):
        """Generate clarifying questions"""
        prompt = f"""For the query: {query}
        
        Generate 2 clarifying questions that help better understand intent:"""
        
        clarifications = llm.generate(prompt, num_variants=2)
        
        return [query] + clarifications
```

## 3. Term Weighting

### Query Term Importance

```python
class QueryTermWeighter:
    """Weight query terms by importance"""
    
    def __init__(self, idf_scores: dict = None):
        self.idf_scores = idf_scores or {}
    
    def weight_query_terms(self, query: str):
        """Compute importance weight for each term"""
        words = query.split()
        weights = {}
        
        for word in words:
            # IDF-based weighting
            idf = self.idf_scores.get(word.lower(), 1.0)
            
            # Term frequency
            freq = words.count(word)
            
            # Combined weight
            weights[word] = idf * np.log(freq + 1)
        
        return weights
    
    def generate_weighted_queries(self, query: str):
        """Generate queries focusing on important terms"""
        weights = self.weight_query_terms(query)
        
        # Sort by importance
        important_words = sorted(
            weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create focused query with important words
        top_words = [w for w, _ in important_words[:3]]
        focused_query = " ".join(top_words)
        
        return [query, focused_query]
```

## 4. Query Rewriting

### Question-Answering Query Transformation

```python
class QAQueryTransformer:
    """Transform queries for QA systems"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def transform_to_qa_format(self, query: str):
        """Transform query into QA format"""
        
        if query.endswith('?'):
            return query  # Already a question
        
        # Convert statement to question
        prompt = f"""Convert this to a natural question:
        
        Statement: {query}
        
        Question:"""
        
        question = self.llm.generate(prompt)
        
        return question
    
    def extract_query_aspects(self, query: str):
        """Break down query into aspects"""
        
        prompt = f"""Identify the key aspects/entities in this query:
        
        Query: {query}
        
        For each aspect, suggest a focused sub-query:"""
        
        aspects = self.llm.generate(prompt, output_format='json')
        
        return aspects
```

## 5. Query Reformulation

### Document-Centric Reformulation

```python
class DocumentCentricReformulator:
    """Reformulate query based on retrieved documents"""
    
    def __init__(self, retriever, embedding_model):
        self.retriever = retriever
        self.embedding_model = embedding_model
    
    def reformulate_for_retrieved_docs(self, query: str, initial_docs: List[str]):
        """Reformulate query to better match retrieved documents"""
        
        # Extract key phrases from retrieved documents
        key_phrases = self._extract_key_phrases(initial_docs)
        
        # Incorporate into query
        reformulated = f"{query} {' '.join(key_phrases[:3])}"
        
        return reformulated
    
    def iterative_reformulation(self, query: str, num_iterations: int = 3):
        """Reformulate query iteratively"""
        
        current_query = query
        
        for i in range(num_iterations):
            # Retrieve with current query
            docs = self.retriever.retrieve(current_query, top_k=10)
            
            # Extract key terms from top docs
            key_terms = self._extract_key_terms(docs)
            
            # Reformulate query
            current_query = self._combine_terms(query, key_terms)
        
        return current_query
    
    def _extract_key_phrases(self, documents: List[str]):
        """Extract important phrases from documents"""
        # Simple: first few words of each document
        phrases = []
        for doc in documents[:3]:
            words = doc.split()[:3]
            phrases.extend(words)
        
        return phrases
    
    def _extract_key_terms(self, docs: List[dict]):
        """Extract key terms from retrieved documents"""
        # TF-IDF or similar
        terms = []
        for doc in docs[:5]:
            words = doc['text'].split()
            terms.extend(words[:5])
        
        return list(dict.fromkeys(terms))  # Deduplicate
    
    def _combine_terms(self, original_query: str, additional_terms: list):
        """Combine original query with additional terms"""
        return f"{original_query} {' '.join(additional_terms[:3])}"
```

## 6. Expansion Evaluation

### Measuring Expansion Quality

```python
class ExpansionEvaluator:
    """Evaluate query expansion quality"""
    
    @staticmethod
    def evaluate_expansion(original_query: str, expanded_queries: List[str],
                          retriever, ground_truth: Set[str]):
        """Evaluate expansion effectiveness"""
        
        # Retrieve with original
        original_results = retriever.retrieve(original_query, top_k=10)
        original_ids = {r['id'] for r in original_results}
        original_recall = len(original_ids & ground_truth) / len(ground_truth)
        
        # Retrieve with expanded
        expanded_results = []
        for query in expanded_queries:
            expanded_results.extend(retriever.retrieve(query, top_k=10))
        
        # Deduplicate
        expanded_ids = {r['id'] for r in expanded_results}
        expanded_recall = len(expanded_ids & ground_truth) / len(ground_truth)
        
        return {
            'original_recall': original_recall,
            'expanded_recall': expanded_recall,
            'recall_improvement': (expanded_recall - original_recall) / max(original_recall, 0.01),
            'num_expansions': len(expanded_queries),
            'unique_results': len(expanded_ids)
        }
    
    @staticmethod
    def measure_query_diversity(queries: List[str], embedding_model):
        """Measure diversity of expanded queries"""
        embeddings = embedding_model.encode(queries)
        
        # Pairwise similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity
        
        return diversity
```

## 7. Best Practices

1. **Use diverse expansion methods** - different methods catch different issues
2. **Limit expansion size** - too many variants hurt performance
3. **Measure recall gains** - expansion increases recall, may hurt precision
4. **Consider query type** - complex queries benefit more from expansion
5. **Cache expansion results** - same queries used repeatedly

## Conclusion

Query expansion improves recall by 15-25% at the cost of some precision. Most effective when using multiple methods (synonyms + semantic + rewriting). Best results from combining with reranking to recover precision. For most systems, start with synonym expansion and add semantic methods if needed.
