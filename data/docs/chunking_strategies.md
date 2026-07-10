# Chunking Strategies: Maximizing Retrieval Quality

## Introduction

Document chunking is often overlooked but critically impacts RAG performance. Chunks that are too large miss fine-grained retrieval, while chunks that are too small create noise and context loss. This guide explores chunking strategies, trade-offs, and implementation patterns.

## 1. Chunking Fundamentals

### The Chunking Problem

Raw documents must be split into retrievable units (chunks). The quality of this process significantly affects:

- **Semantic coherence**: Each chunk should represent a complete thought
- **Retrieval precision**: Chunks too large include irrelevant content
- **Redundancy**: Overlapping chunks improve recall
- **Context loss**: Small chunks lose surrounding context
- **Computational efficiency**: Chunk size affects embedding and storage costs

### Optimal Chunk Size

```
Document Type     Optimal Size  Reasoning
─────────────────────────────────────────────────────
News articles     300-500 words  Single topic per chunk
Technical docs    400-800 words  May span multiple concepts
Scientific papers 500-1000 words High density, inter-related ideas
Chat messages     50-200 words   Already small units
Code files        200-400 words  Function/class level
Long-form essays  400-600 words  Paragraph/section level
```

## 2. Chunking Strategies

### 1. Fixed-Size Chunking with Overlap

Simplest approach: divide into fixed-length chunks with overlap.

```python
class FixedSizeChunker:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata=None):
        """Split text into fixed-size overlapping chunks"""
        chunks = []
        words = text.split()
        
        stride = self.chunk_size - self.overlap
        
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + self.chunk_size]
            
            if len(chunk_words) > 0:
                chunk_text = ' '.join(chunk_words)
                
                chunks.append({
                    'text': chunk_text,
                    'start_idx': i,
                    'end_idx': min(i + self.chunk_size, len(words)),
                    'metadata': metadata
                })
        
        return chunks

# Example
chunker = FixedSizeChunker(chunk_size=256, overlap=32)
chunks = chunker.chunk_text(
    "Long document text...",
    metadata={'source': 'document.pdf', 'page': 1}
)
```

**Advantages:**
- Simple, deterministic
- Predictable performance
- Easy to parallelize

**Disadvantages:**
- Ignores document structure
- May split sentences mid-way
- Overlap wastes storage

### 2. Semantic Chunking

Split documents at semantic boundaries using embeddings.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticChunker:
    def __init__(self, model_name='intfloat/e5-large', threshold=0.5):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
    
    def chunk_text(self, text: str, metadata=None):
        """Split at semantic boundaries"""
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        # Encode sentences
        sentence_embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_embedding = None
        
        for i, (sentence, embedding) in enumerate(
            zip(sentences, sentence_embeddings)
        ):
            if current_embedding is None:
                current_chunk.append(sentence)
                current_embedding = embedding
            else:
                # Compute similarity to current chunk
                similarity = self._cosine_similarity(
                    current_embedding,
                    embedding
                )
                
                if similarity > self.threshold:
                    # Semantically similar, add to chunk
                    current_chunk.append(sentence)
                    
                    # Update running average of embedding
                    current_embedding = (
                        current_embedding * (i - len(current_chunk)) +
                        embedding
                    ) / (i - len(current_chunk) + 1)
                else:
                    # Semantic boundary, start new chunk
                    if current_chunk:
                        chunks.append({
                            'text': ' '.join(current_chunk),
                            'sentence_count': len(current_chunk),
                            'metadata': metadata
                        })
                    
                    current_chunk = [sentence]
                    current_embedding = embedding
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'sentence_count': len(current_chunk),
                'metadata': metadata
            })
        
        return chunks
    
    def _split_into_sentences(self, text):
        """Split text into sentences"""
        import re
        # Simple sentence splitter (use spaCy or NLTK for production)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, a, b):
        """Compute cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Advantages:**
- Respects semantic boundaries
- Better chunk coherence
- Adaptive chunk size

**Disadvantages:**
- Computationally expensive
- Requires embedding model
- May create very small chunks

### 3. Recursive Chunking

Use multiple splitting strategies in order of preference.

```python
class RecursiveChunker:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Splitting strategies in order
        self.separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentences
            " ",     # Words
        ]
    
    def chunk_text(self, text: str, metadata=None):
        """Recursively chunk using separators"""
        chunks = []
        
        # Try each separator
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                
                good_splits = [
                    s for s in splits
                    if len(s) > self.chunk_size * 0.5
                ]
                
                if good_splits:
                    # Recursively chunk the good splits
                    for split in good_splits:
                        new_chunks = self._merge_splits(
                            split,
                            separator,
                            metadata
                        )
                        chunks.extend(new_chunks)
                    
                    return chunks
        
        # No separator found, return as-is
        return [{
            'text': text,
            'metadata': metadata
        }]
    
    def _merge_splits(self, text, separator, metadata):
        """Merge splits to match chunk size"""
        if len(text) <= self.chunk_size:
            return [{'text': text, 'metadata': metadata}]
        
        # Recursively split
        return self.chunk_text(text, metadata)
```

**Advantages:**
- Respects document structure
- Falls back to finer granularity when needed
- Produces coherent chunks

**Disadvantages:**
- More complex logic
- Still somewhat arbitrary

### 4. Document Structure-Aware Chunking

Use document structure (headers, sections) for chunking.

```python
from typing import List, Dict
import re

class StructureAwareChunker:
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
    
    def chunk_markdown(self, text: str, metadata=None):
        """Chunk Markdown documents respecting structure"""
        chunks = []
        current_chunk = []
        current_section = ""
        current_level = 0
        
        lines = text.split('\n')
        
        for line in lines:
            # Detect heading level
            heading_match = re.match(r'^(#+)\s+(.*)', line)
            
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2)
                
                # Save current chunk if exists
                if current_chunk and len(' '.join(current_chunk)) > 100:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'section': current_section,
                        'metadata': metadata
                    })
                    current_chunk = []
                
                current_section = heading_text
                current_level = level
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
                # Check if chunk is large enough
                chunk_size_words = len(' '.join(current_chunk).split())
                if chunk_size_words >= self.chunk_size:
                    chunks.append({
                        'text': '\n'.join(current_chunk),
                        'section': current_section,
                        'metadata': metadata
                    })
                    current_chunk = []
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'section': current_section,
                'metadata': metadata
            })
        
        return chunks
    
    def chunk_html(self, html: str, metadata=None):
        """Chunk HTML documents preserving structure"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        chunks = []
        current_chunk = []
        current_section = ""
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
            if element.name in ['h1', 'h2', 'h3']:
                # Save current chunk
                if current_chunk:
                    chunks.append({
                        'text': '\n'.join(current_chunk),
                        'section': current_section,
                        'metadata': metadata
                    })
                    current_chunk = []
                
                current_section = element.get_text()
            else:
                text = element.get_text().strip()
                if text:
                    current_chunk.append(text)
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'section': current_section,
                'metadata': metadata
            })
        
        return chunks
```

## 3. Advanced Chunking Techniques

### Multi-Resolution Chunking

Create chunks at multiple granularities for flexible retrieval.

```python
class MultiResolutionChunker:
    def __init__(self):
        self.coarse_chunker = FixedSizeChunker(chunk_size=1024, overlap=100)
        self.fine_chunker = FixedSizeChunker(chunk_size=256, overlap=32)
    
    def chunk_text(self, text: str, metadata=None):
        """Create chunks at multiple resolutions"""
        coarse = self.coarse_chunker.chunk_text(text, metadata)
        fine = self.fine_chunker.chunk_text(text, metadata)
        
        return {
            'coarse': coarse,  # Context-rich, fewer chunks
            'fine': fine       # Precise, more chunks
        }
    
    def retrieve_multi_resolution(self, query, similarity_threshold=0.7):
        """Retrieve using appropriate resolution"""
        # Try fine chunks first (more precise)
        fine_results = self.search_fine(query, threshold=similarity_threshold)
        
        if len(fine_results) < 3:
            # Fall back to coarse chunks if sparse
            return self.search_coarse(query, threshold=0.5)
        
        return fine_results
```

### Metadata Preservation

Maintain rich metadata during chunking for better retrieval context.

```python
class MetadataPreservingChunker:
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
    
    def chunk_with_metadata(self, document, metadata):
        """Preserve and propagate metadata"""
        chunks = self._base_chunking(document)
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunks.append({
                'text': chunk['text'],
                'chunk_id': f"{metadata['doc_id']}_chunk_{i}",
                'chunk_num': i,
                'total_chunks': len(chunks),
                'source': metadata.get('source', ''),
                'date': metadata.get('date', ''),
                'author': metadata.get('author', ''),
                'category': metadata.get('category', ''),
                'section': chunk.get('section', ''),
                'offset': chunk.get('start_idx', 0)
            })
        
        return enhanced_chunks
    
    def _base_chunking(self, text):
        # Implementation of chunking
        pass
```

## 4. Chunking Evaluation

### Quality Metrics

```python
class ChunkingEvaluator:
    @staticmethod
    def evaluate_chunks(chunks, query, relevant_docs):
        """Evaluate chunking quality"""
        metrics = {
            'chunk_size_stats': ChunkingEvaluator._size_stats(chunks),
            'coverage': ChunkingEvaluator._coverage_score(chunks, relevant_docs),
            'coherence': ChunkingEvaluator._coherence_score(chunks),
        }
        return metrics
    
    @staticmethod
    def _size_stats(chunks):
        """Compute chunk size statistics"""
        sizes = [len(chunk['text'].split()) for chunk in chunks]
        return {
            'mean': np.mean(sizes),
            'median': np.median(sizes),
            'min': np.min(sizes),
            'max': np.max(sizes),
            'std': np.std(sizes)
        }
    
    @staticmethod
    def _coverage_score(chunks, relevant_docs):
        """% of relevant documents covered by chunks"""
        chunk_texts = ' '.join([c['text'] for c in chunks])
        
        covered = sum(
            1 for doc in relevant_docs
            if any(phrase in chunk_texts for phrase in doc.split()[:5])
        )
        
        return covered / len(relevant_docs) if relevant_docs else 0
    
    @staticmethod
    def _coherence_score(chunks):
        """Average semantic coherence within chunks"""
        model = SentenceTransformer('intfloat/e5-large')
        coherence_scores = []
        
        for chunk in chunks:
            sentences = chunk['text'].split('.')
            if len(sentences) > 1:
                embeddings = model.encode(sentences)
                # Average pairwise similarity
                avg_sim = np.mean([
                    np.dot(embeddings[i], embeddings[i+1])
                    for i in range(len(embeddings)-1)
                ])
                coherence_scores.append(avg_sim)
        
        return np.mean(coherence_scores) if coherence_scores else 0
```

## 5. Best Practices

| Strategy | Best For | Chunk Size | Overlap |
|----------|----------|-----------|---------|
| Fixed-size | Simple use cases | 256-512 | 10-20% |
| Semantic | Complex documents | 300-600 | 0-10% |
| Recursive | Mixed content | 400-800 | 5-15% |
| Structure-aware | Markdown/HTML | 500-1000 | 0% |

## Key Recommendations

1. **Start with recursive chunking** - good default balancing simplicity and quality
2. **Use structure when available** - respect document boundaries
3. **Monitor chunk quality metrics** - track size distribution and coverage
4. **Implement overlap carefully** - 10-20% reduces information loss without excessive storage
5. **Preserve metadata** - enables better filtering and context in generation

## Conclusion

Chunking is a critical but often underestimated component of RAG. The optimal strategy depends on document type, use case, and computational constraints. A combination of structure-aware and semantic approaches typically yields best results. Always evaluate chunking quality empirically on your specific dataset and query patterns.
