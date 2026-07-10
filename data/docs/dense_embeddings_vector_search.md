# Dense Embeddings and Vector Search in RAG

## What Are Dense Embeddings?

Dense embeddings are fixed-length vector representations of text that encode semantic meaning. Unlike sparse representations (BM25), which use vocabulary-indexed binary vectors, dense embeddings compress semantic information into continuous vectors, typically 384-1536 dimensions.

Modern embedding models are trained on large corpora using contrastive learning objectives (e.g., Sentence Transformers), enabling:
- **Semantic similarity**: Similar meaning → similar vectors
- **Compositionality**: Vector arithmetic works (e.g., king - man + woman ≈ queen)
- **Paraphrase matching**: Different words, same meaning, close vectors

## Popular Embedding Models

### General-Purpose Models
- **BGE-Small (BAAI/bge-small-en-v1.5)**: 384 dims, 33M params, fast, good for most use cases
- **BGE-Large**: 768 dims, high quality but slower
- **text-embedding-3-large (OpenAI)**: 3072 dims, state-of-the-art, but costs $0.02/million tokens

### Specialized Models
- **Medical**: MedDPR (medical papers), BioWordVec (biomedical)
- **Legal**: Legal-BERT fine-tuned models
- **Code**: Code Embeddings (GitHub's models)
- **Multilingual**: mBERT, XLM-RoBERTa

## Similarity Metrics

The choice of similarity metric affects retrieval performance and resource requirements.

### Cosine Similarity (Default)
- **Formula**: `cos(u, v) = (u · v) / (||u|| * ||v||)`
- **Range**: -1 to 1 (typically 0 to 1 for normalized embeddings)
- **Computation**: Fastest (just dot product after normalization)
- **Use case**: General-purpose, most embedding models trained with this

### Euclidean Distance (L2)
- **Formula**: `L2(u, v) = sqrt(Σ(ui - vi)²)`
- **Range**: 0 to ∞
- **Computation**: Slower than cosine
- **Use case**: Dense clusters (FAISS, some vector DBs)

### Dot Product (Inner Product)
- **Formula**: `u · v`
- **Range**: Unbounded
- **Computation**: Fastest (no normalization needed)
- **Use case**: When embeddings are pre-normalized

## Vector Database Indexing Algorithms

For fast similarity search over millions of embeddings, vector databases use approximate nearest neighbor (ANN) algorithms instead of brute-force search.

### HNSW (Hierarchical Navigable Small Worlds)
- **Structure**: Graph-based hierarchical layer structure
- **Complexity**: O(log n) search, O(n) indexing
- **Best for**: Low latency (<100ms), reasonable memory
- **Databases**: Qdrant, Weaviate, Pinecone
- **Tradeoff**: Higher RAM than competitors, extremely fast

### IVF (Inverted File Index)
- **Structure**: Coarse quantization into partitions, then search partition
- **Complexity**: O(k * nlist) where k is cluster count
- **Best for**: Large corpus with memory constraints
- **Databases**: FAISS, Milvus
- **Tradeoff**: Can miss neighbors in far partitions; recall-speed tradeoff tunable

### FLAT (Brute Force)
- **Structure**: Linear scan all vectors
- **Complexity**: O(n) search
- **Best for**: <1M vectors or requiring 100% recall
- **Use case**: Small RAG systems, precision-critical

### PQ (Product Quantization)
- **Structure**: Split vectors into segments, compress each segment
- **Compression**: 32-dimensional vector → 1-4 bytes per segment
- **Best for**: Extreme memory constraints (on-device retrieval)
- **Tradeoff**: Accuracy loss (typically 2-5% recall)

## Strengths of Dense Retrieval

1. **Semantic Understanding**: Catches paraphrased queries with same meaning
2. **Short Query Support**: Works on single words or phrases without training
3. **Cross-lingual**: Same vector space for multiple languages (multilingual models)
4. **Composition**: Negation, boolean logic possible via vector arithmetic
5. **Speed**: Modern ANN algorithms achieve sub-millisecond latency at scale

## Limitations of Dense Retrieval

1. **Rare Entity Blindness**: Proper nouns not in training data → poor embeddings
2. **Vocabulary Mismatch Reversed**: Domain-specific terminology can produce inconsistent embeddings
3. **Length Bias**: Longer passages often score higher due to more content
4. **Embedding Space Quality**: Dependent on training corpus; domain shift → poor performance
5. **Similarity Score Meaningless**: Cosine similarity 0.7 ≠ 0.7 units of relevance

## Vector Search Performance Characteristics

### Latency vs. Recall Tradeoff
- HNSW with ef=100: ~5ms latency, 95% recall
- HNSW with ef=300: ~15ms latency, 99% recall
- FLAT: ~50ms latency for 1M vectors, 100% recall

### Memory Consumption
- Raw vectors: ~4 bytes/dimension (float32) × n vectors × dimension
  - 1M vectors × 384 dims = 1.5 GB
- HNSW graph: +10-20% overhead
- IVF with PQ: 1-4 bytes per vector (compressed)

### Scalability Limits
- Single-machine HNSW: ~10-100M vectors
- Sharded HNSW: 1B+ vectors (distributed)
- PQ-based: 1B+ vectors easily on single machine

## Practical Considerations for RAG

### Choosing an Embedding Model
1. **Speed requirement** → smaller model (BGE-small)
2. **Accuracy requirement** → larger model (text-embedding-3-large)
3. **Domain-specific needs** → fine-tune or use specialized model
4. **Cost** → open-source models are free (but self-hosted)

### Indexing Strategy
1. **Chunk documents** into 300-800 token passages with overlap
2. **Embed all chunks** at ingest time (batch processing)
3. **Use ANN** with appropriate recall target (98-99% typical)
4. **Monitor drift** by embedding validation samples monthly

### Query Processing
1. **Embed the query** with the SAME model as documents
2. **Search top-K** (typically 10-50 candidates)
3. **Rerank** with cross-encoder for precision (see next section)

## Hybrid Retrieval: Combining with BM25

Dense retrieval alone misses exact keywords. Production RAG stacks use:
1. Run BM25 (sparse) and dense vector search in parallel
2. Fuse results with Reciprocal Rank Fusion (RRF)
3. Rerank fused results with cross-encoder

This achieves recall on both semantic similarity and exact keywords.
