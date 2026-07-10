# Context Window Management for LLMs

## Introduction

Modern LLMs have large but finite context windows. Effectively managing context is crucial for RAG performance. This guide covers strategies for maximizing information density within token limits.

## 1. Context Window Fundamentals

### Understanding Token Limits

```python
from transformers import AutoTokenizer

class ContextWindowManager:
    def __init__(self, model_name='gpt-3.5-turbo'):
        self.model_name = model_name
        self.context_windows = {
            'gpt-3.5-turbo': 4096,
            'gpt-4': 8192,
            'gpt-4-32k': 32768,
            'claude-3-sonnet': 200000,
            'claude-3-opus': 200000,
        }
        self.tokenizer = self._load_tokenizer()
    
    def get_context_limit(self):
        """Get context window for model"""
        return self.context_windows.get(self.model_name, 4096)
    
    def count_tokens(self, text):
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def estimate_tokens(self, text, method='word_ratio'):
        """Fast estimate of token count"""
        if method == 'word_ratio':
            # Rough estimate: 1 token ≈ 1.3 words
            words = len(text.split())
            return int(words / 1.3)
        elif method == 'char_ratio':
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4
```

## 2. Context Allocation Strategy

### Optimal Token Distribution

```python
class ContextAllocator:
    def __init__(self, context_window: int = 8192):
        self.context_window = context_window
    
    def allocate_tokens(self, query_tokens: int, num_documents: int, response_budget: int = 500):
        """Allocate tokens optimally"""
        # Reserve tokens for response
        reserved_for_response = response_budget
        
        # Reserve tokens for prompt structure
        prompt_structure_tokens = 200  # System prompt, formatting, etc.
        
        # Available for context
        available_for_context = self.context_window - reserved_for_response - prompt_structure_tokens - query_tokens
        
        # Allocate equally to documents (with minimum per document)
        min_per_doc = 100
        max_per_doc = max(min_per_doc, available_for_context // num_documents)
        
        return {
            'total_window': self.context_window,
            'reserved_for_response': reserved_for_response,
            'prompt_structure': prompt_structure_tokens,
            'query': query_tokens,
            'context': available_for_context,
            'per_document': max_per_doc,
            'num_documents': min(
                num_documents,
                available_for_context // min_per_doc
            )
        }
    
    def should_compress_context(self, context_tokens: int):
        """Determine if context should be compressed"""
        threshold = self.context_window * 0.7
        return context_tokens > threshold
```

## 3. Context Compression Techniques

### Abstractive Summarization

```python
from transformers import pipeline

class ContextCompressor:
    def __init__(self, compression_ratio: float = 0.3):
        self.compression_ratio = compression_ratio
        self.summarizer = pipeline("summarization", 
                                  model="facebook/bart-large-cnn")
    
    def compress_document(self, document: str, target_ratio: float = None):
        """Compress document via abstractive summarization"""
        if target_ratio is None:
            target_ratio = self.compression_ratio
        
        words = document.split()
        target_words = int(len(words) * target_ratio)
        
        # Chunk if too long for summarizer
        max_chunk_length = 1024  # words
        chunks = [
            ' '.join(words[i:i+max_chunk_length])
            for i in range(0, len(words), max_chunk_length)
        ]
        
        summaries = []
        for chunk in chunks:
            summary = self.summarizer(chunk, max_length=100, min_length=30)
            summaries.append(summary[0]['summary_text'])
        
        compressed = ' '.join(summaries)
        
        return {
            'original_length': len(words),
            'compressed_length': len(compressed.split()),
            'compression_ratio': len(compressed.split()) / len(words),
            'text': compressed
        }
    
    def compress_context(self, documents: list, target_tokens: int):
        """Compress multiple documents to fit token budget"""
        # Proportionally compress each document
        tokens_per_doc = target_tokens // len(documents)
        
        compressed_docs = []
        for doc in documents:
            # Estimate compression ratio needed
            doc_tokens = len(doc['text'].split()) * 1.3  # rough estimate
            needed_ratio = tokens_per_doc / doc_tokens
            
            compressed = self.compress_document(
                doc['text'],
                target_ratio=min(1.0, needed_ratio)
            )
            
            compressed_docs.append({
                **doc,
                'text': compressed['text'],
                'compressed': True,
                'compression_ratio': compressed['compression_ratio']
            })
        
        return compressed_docs
```

### Extractive Summarization

```python
class ExtractiveCompressor:
    """Extract key sentences instead of generating new ones"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def extract_key_sentences(self, document: str, ratio: float = 0.3):
        """Extract most important sentences"""
        sentences = document.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Embed sentences
        embeddings = self.embedding_model.encode(sentences)
        
        # Compute sentence importance via clustering
        from sklearn.cluster import KMeans
        
        n_clusters = max(1, int(len(sentences) * ratio))
        
        if n_clusters >= len(sentences):
            return document  # Keep all
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        
        # Select closest sentence to each cluster center
        selected_indices = []
        for cluster_center in kmeans.cluster_centers_:
            closest_idx = np.argmin(
                np.linalg.norm(embeddings - cluster_center, axis=1)
            )
            selected_indices.append(closest_idx)
        
        # Maintain original order
        selected_indices = sorted(selected_indices)
        
        # Reconstruct
        extracted_sentences = [sentences[i] for i in selected_indices]
        
        return '. '.join(extracted_sentences) + '.'
```

## 4. Hierarchical Context

### Progressive Context Loading

```python
class ProgressiveContextLoader:
    """Load context progressively for token efficiency"""
    
    def __init__(self, context_window: int = 8192):
        self.context_window = context_window
    
    def create_initial_context(self, documents: list, initial_tokens: int = 1000):
        """Create initial minimal context"""
        context_parts = []
        token_count = 0
        
        for doc in sorted(documents, key=lambda x: x.get('score', 0), reverse=True):
            # Add summary/title first
            title_section = f"Source: {doc.get('source', 'Unknown')}"
            title_tokens = len(title_section.split()) * 1.3
            
            if token_count + title_tokens < initial_tokens:
                context_parts.append(title_section)
                token_count += title_tokens
            else:
                break
        
        return '\n'.join(context_parts)
    
    def expand_context(self, documents: list, current_tokens: int, target_tokens: int):
        """Expand context with additional details"""
        context_parts = []
        token_count = current_tokens
        
        for doc in documents:
            # Add more details
            detail_section = f"{doc['text'][:500]}"
            detail_tokens = len(detail_section.split()) * 1.3
            
            if token_count + detail_tokens < target_tokens:
                context_parts.append(detail_section)
                token_count += detail_tokens
            else:
                # Truncate to fit
                remaining = int((target_tokens - token_count) / 1.3)
                if remaining > 50:
                    truncated = ' '.join(doc['text'].split()[:remaining])
                    context_parts.append(truncated)
                break
        
        return '\n'.join(context_parts)
```

## 5. Context Quality vs Quantity

### Precision-Recall Tradeoff

```python
class ContextQualityOptimizer:
    """Optimize context quality within token budget"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def evaluate_context_quality(self, context: str, query: str, embedding_model):
        """Evaluate quality of context"""
        query_emb = embedding_model.encode(query)
        context_emb = embedding_model.encode(context[:500])  # First part
        
        relevance = np.dot(query_emb, context_emb)
        
        # Measure diversity (should not be too redundant)
        sentences = context.split('.')
        sentence_embs = embedding_model.encode(sentences)
        
        pairwise_similarities = np.dot(sentence_embs, sentence_embs.T)
        diversity = 1 - np.mean(pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)])
        
        return {
            'relevance': relevance,
            'diversity': diversity,
            'overall_quality': 0.7 * relevance + 0.3 * diversity
        }
    
    def find_optimal_context_size(self, documents: list, query: str, embedding_model):
        """Find context size that maximizes quality"""
        results = []
        
        for num_docs in range(1, min(len(documents) + 1, 10)):
            context = '\n'.join([d['text'] for d in documents[:num_docs]])
            quality = self.evaluate_context_quality(context, query, embedding_model)
            tokens = len(context.split()) * 1.3
            
            results.append({
                'num_docs': num_docs,
                'tokens': tokens,
                'quality': quality['overall_quality']
            })
        
        # Find best quality-to-token ratio
        best = max(results, key=lambda x: x['quality'] / max(x['tokens'], 1))
        
        return best
```

## 6. Dynamic Context Adjustment

### Feedback-Based Optimization

```python
class DynamicContextOptimizer:
    """Adjust context based on generation quality feedback"""
    
    def __init__(self):
        self.context_history = []
    
    def adjust_context_for_next_query(self, previous_quality_score: float, 
                                     previous_context_tokens: int):
        """Adjust context size based on previous result"""
        
        if previous_quality_score > 0.8:
            # Good quality, can reduce context
            adjustment = 0.8
        elif previous_quality_score > 0.6:
            # Acceptable, maintain
            adjustment = 1.0
        else:
            # Poor quality, increase context
            adjustment = 1.3
        
        recommended_tokens = int(previous_context_tokens * adjustment)
        
        return recommended_tokens
    
    def learn_optimal_context(self, query_context_quality_tuples: list):
        """Learn optimal context size from history"""
        # Group by query complexity
        simple_queries = []
        complex_queries = []
        
        for query, context_tokens, quality in query_context_quality_tuples:
            # Estimate complexity from query length
            if len(query.split()) < 10:
                simple_queries.append((context_tokens, quality))
            else:
                complex_queries.append((context_tokens, quality))
        
        # Find optimal context for each category
        optimal = {}
        
        for category, tuples in [('simple', simple_queries), ('complex', complex_queries)]:
            if tuples:
                # Quality per token
                quality_per_token = [q / max(c, 1) for c, q in tuples]
                best_idx = np.argmax(quality_per_token)
                optimal[category] = tuples[best_idx][0]
        
        return optimal
```

## 7. Multi-Turn Conversation Context

### Managing Conversation History

```python
class ConversationContextManager:
    """Manage context in multi-turn conversations"""
    
    def __init__(self, context_window: int = 8192):
        self.context_window = context_window
        self.conversation_history = []
    
    def add_turn(self, query: str, response: str):
        """Add turn to conversation"""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'tokens': self._estimate_tokens(query + response)
        })
    
    def get_effective_context(self, max_tokens: int = None):
        """Get conversation context within token limit"""
        if max_tokens is None:
            max_tokens = self.context_window // 2  # Reserve half for new content
        
        context_parts = []
        token_count = 0
        
        # Go backwards through history to get recent context
        for turn in reversed(self.conversation_history):
            if token_count + turn['tokens'] < max_tokens:
                context_parts.insert(0, f"Q: {turn['query']}\nA: {turn['response'][:200]}")
                token_count += turn['tokens']
            else:
                break
        
        return '\n'.join(context_parts)
    
    def summarize_history(self, num_turns_to_keep: int = 5):
        """Summarize old history to save tokens"""
        if len(self.conversation_history) <= num_turns_to_keep:
            return
        
        # Keep recent turns, summarize older ones
        old_turns = self.conversation_history[:-num_turns_to_keep]
        recent_turns = self.conversation_history[-num_turns_to_keep:]
        
        if old_turns:
            summary = self._summarize_turns(old_turns)
            
            self.conversation_history = [
                {'query': 'Previous conversation summary', 'response': summary, 'tokens': self._estimate_tokens(summary)},
                *recent_turns
            ]
    
    def _estimate_tokens(self, text):
        """Estimate token count"""
        return len(text.split()) * 1.3
    
    def _summarize_turns(self, turns):
        """Create summary of multiple turns"""
        # Simplified: concatenate key points
        key_points = []
        for turn in turns:
            key_points.append(f"- {turn['query'][:50]}")
        
        return "Previous discussion covered: " + ", ".join(key_points)
```

## Best Practices

1. **Reserve at least 20% for response** - give model room to generate
2. **Use hierarchical context** - summaries + details on demand
3. **Monitor token usage carefully** - costs increase linearly with tokens
4. **Compress intelligently** - extractive better than abstractive for precision
5. **Prioritize relevance over quantity** - fewer relevant tokens > many irrelevant
6. **Test different context sizes** - find sweet spot for your use case

## Conclusion

Effective context window management requires careful planning. Reserve space for response, allocate remaining space optimally among system prompt/structure, query, and context. Most improvements come from compression and selection rather than trying to fit more information.
