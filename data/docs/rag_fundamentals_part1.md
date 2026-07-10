# RAG Fundamentals: Core Concepts and Architecture

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) is a technique that combines document retrieval with language model generation to provide answers grounded in external knowledge. The basic workflow:

1. **Ingest**: Process and index documents into a vector database
2. **Retrieve**: Search for relevant documents using the user's query
3. **Rank**: Score and reorder results by relevance
4. **Generate**: Pass top results to an LLM to synthesize an answer

## Why RAG Matters

Traditional LLMs have three fundamental problems:
- **Knowledge cutoff**: Training data stops at a specific date
- **Hallucination**: Models generate plausible-sounding but false information
- **No source attribution**: Hard to verify where answers come from

RAG addresses all three by:
- Retrieving up-to-date information from your corpus
- Grounding answers in retrieved text (reduces hallucination)
- Providing citations that users can verify

## The RAG Pipeline: Five Stages

### Stage 1: Document Ingestion
Input: Raw documents (PDF, TXT, MD, HTML)
Process:
- Parse document content
- Split into chunks (300-800 tokens typical)
- Extract metadata (source, date, author)
Output: List of (chunk, metadata) pairs

### Stage 2: Embedding and Indexing
Input: Chunks from Stage 1
Process:
- Embed each chunk using a dense model (e.g., BAAI/bge-small-en-v1.5)
- Index embeddings in vector database (Qdrant, Pinecone, Milvus)
- Build auxiliary indices (BM25 inverted index for sparse retrieval)
Output: Searchable vector index + inverted index

### Stage 3: Query Processing
Input: User question
Process:
- Embed query with same model as documents
- Optionally expand query (synonyms, reformulations)
Output: Query embedding

### Stage 4: Retrieval and Ranking
Input: Query embedding + indexed documents
Process:
- Dense retrieval: Vector similarity search (top-K nearest neighbors)
- Sparse retrieval: BM25 keyword matching (alternative or complement)
- Hybrid fusion: Combine dense + sparse results via Reciprocal Rank Fusion
- Reranking: Cross-encoder model scores each (query, doc) pair for precision
Output: Top-M ranked documents (usually 3-5 for LLM context)

### Stage 5: Generation and Answer Synthesis
Input: Top-M documents + user question
Process:
- Construct prompt with question + context chunks
- Call LLM with prompt + system instruction
- Parse response and extract citations
Output: Answer + source attribution

## Key Design Decisions

### Chunk Size
- **Too small** (<100 tokens): Loses context, weak embeddings
- **Too large** (>1000 tokens): Dilutes relevance signal, fills context window
- **Sweet spot**: 300-800 tokens with 50-100 token overlap

### Embedding Model Choice
- **Small** (384 dims): Fast, low memory, good for most use cases
- **Large** (1536+ dims): Better accuracy, higher latency/cost
- **Domain-specific**: Fine-tuned models beat general models on specialized content

### Retrieval Strategy
- **Dense-only**: Fast, works on semantic queries, misses exact keywords
- **Sparse-only** (BM25): Fast, precise on technical terms, no semantic understanding
- **Hybrid**: Best of both, standard in production RAG

### Reranking Decision
- **No rerank**: 5x faster, ~70% quality
- **Light rerank** (top-10): Balanced cost/quality (recommended)
- **Heavy rerank** (top-50): Maximum quality, slower

## Common RAG Architectures

### Basic RAG (Naive)
```
Query → Embed → Dense Search → Top-K → LLM → Answer
```
Pros: Simple, fast
Cons: Misses keywords, limited quality

### Hybrid RAG (Industry Standard)
```
Query → Split dense + BM25
       ↓         ↓
    Top-K    Top-K'
       └─→ Fuse (RRF) → Top-M
              ↓
         Rerank (cross-encoder)
              ↓
           Top-5 → LLM → Answer
```
Pros: Catches semantic + keyword, high quality
Cons: More complex, 2-3x slower than basic

### Graph-Based RAG
```
Query → Entity Extraction → Knowledge Graph Walk → Passage Retrieval → LLM
```
Pros: Excellent on multi-hop questions, structured reasoning
Cons: Expensive (entity extraction bottleneck), needs entity-rich corpus

### Agentic RAG (LLM-Driven)
```
Query → LLM Tool Loop
          ├→ search (query reformulation loop)
          ├→ fetch (get full documents)
          └→ finish (answer synthesis)
```
Pros: Adaptive, handles ambiguous queries, great on reasoning questions
Cons: High latency (multiple LLM calls), high token cost, unpredictable

## Measuring RAG Quality

Key metrics:
- **Retrieval Recall@K**: Did top-K results include relevant documents?
- **Reranking NDCG**: How well ordered are the retrieved documents?
- **Answer Faithfulness**: Does answer only use retrieved context?
- **Answer Relevance**: Does answer address the question?
- **Latency**: End-to-end time from query to answer
- **Token Cost**: Input + output tokens consumed

## When RAG Fails

RAG degrades when:
1. **Corpus mismatch**: Question asks about topics not in documents
2. **Ambiguous retrieval**: Multiple documents equally relevant
3. **Long-horizon reasoning**: Answer requires integrating 10+ documents
4. **Structured data queries**: "Show me documents where X > 100"
5. **Contradictory sources**: Retrieved documents disagree

## RAG vs. Fine-tuning vs. Prompting

| Approach | Knowledge | Latency | Cost | Best For |
|----------|-----------|---------|------|----------|
| RAG | Updated docs | Slow | Medium | Dynamic knowledge |
| Fine-tuning | Trained params | Fast | High | Behavior changes |
| Prompting | Context window | Fast | Low | Small questions |
| RAG + FT | Both | Slow | High | Maximum quality |

## The RAG Measurement Mindset

Most RAG projects skip evaluation. This is the biggest mistake.

A strategy that crushes single-fact lookups might waste tokens on synthesis. One that handles ambiguity can overshoot on simple queries. The only way to know what works on YOUR data is to measure empirically on a golden dataset.

Production RAG requires:
1. Golden dataset (hand-labeled Q&A pairs)
2. Multiple strategies tested in parallel
3. Metrics tracked over time (catch regressions)
4. Continuous iteration based on measured results
