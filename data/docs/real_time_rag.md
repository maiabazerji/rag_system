# Real-Time RAG: Building Low-Latency Systems

## Introduction

Many RAG applications require sub-second latency for interactive use cases. This guide covers architectural patterns, optimization techniques, and trade-offs for building real-time RAG systems.

## 1. Latency Analysis

### Latency Budget

A typical RAG system has these components:

```
Total Latency = Query Encoding + Vector Lookup + Reranking + LLM Generation
                   50ms        +    30ms      +   100ms    +     1000ms
                 = ~1180ms (1.2 seconds) baseline

Interactive Target: <500ms
Real-time Target: <100ms
```

### Breaking Down the Pipeline

```python
import time
from contextlib import contextmanager

class LatencyProfiler:
    def __init__(self):
        self.measurements = {}
    
    @contextmanager
    def measure(self, component_name):
        """Context manager for measuring latency"""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000  # ms
            
            if component_name not in self.measurements:
                self.measurements[component_name] = []
            
            self.measurements[component_name].append(elapsed)
    
    def get_stats(self, component_name):
        """Get latency statistics for component"""
        times = self.measurements.get(component_name, [])
        
        if not times:
            return None
        
        return {
            'mean': np.mean(times),
            'p50': np.percentile(times, 50),
            'p99': np.percentile(times, 99),
            'max': np.max(times),
            'count': len(times)
        }
    
    def profile_rag(self, query, retriever, reranker, llm):
        """Profile entire RAG pipeline"""
        profiler = LatencyProfiler()
        
        with profiler.measure('query_encoding'):
            query_emb = self._encode_query(query)
        
        with profiler.measure('vector_lookup'):
            candidates = retriever.retrieve(query_emb, top_k=50)
        
        with profiler.measure('reranking'):
            reranked = reranker.rerank(query, candidates, top_k=5)
        
        with profiler.measure('context_assembly'):
            context = self._assemble_context(reranked)
        
        with profiler.measure('llm_generation'):
            response = llm.generate(query, context)
        
        return response, profiler.measurements
```

## 2. Caching Strategies

### Query Result Caching

```python
class QueryCacheManager:
    def __init__(self, cache_size_mb=500):
        self.cache = {}
        self.max_size = cache_size_mb * 1024 * 1024  # bytes
        self.current_size = 0
        self.hits = 0
        self.misses = 0
    
    def get_cached_result(self, query):
        """Get cached result if available"""
        # Normalize query for consistent caching
        cache_key = self._normalize_query(query)
        
        if cache_key in self.cache:
            self.hits += 1
            result = self.cache[cache_key]
            result['cached'] = True
            result['age'] = time.time() - result['time']
            return result
        
        self.misses += 1
        return None
    
    def cache_result(self, query, result):
        """Cache result for future queries"""
        cache_key = self._normalize_query(query)
        
        result_size = self._estimate_size(result)
        
        # Evict if necessary
        if self.current_size + result_size > self.max_size:
            self._evict_lru()
        
        self.cache[cache_key] = {
            'result': result,
            'time': time.time(),
            'size': result_size,
            'hits': 0
        }
        
        self.current_size += result_size
    
    def _normalize_query(self, query):
        """Normalize query for caching"""
        # Remove extra whitespace, lowercase
        return ' '.join(query.lower().split())
    
    def _estimate_size(self, obj):
        """Estimate object size in bytes"""
        import sys
        return sys.getsizeof(obj)
    
    def _evict_lru(self):
        """Evict least recently used items"""
        if not self.cache:
            return
        
        # Sort by time accessed
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1]['time']
        )
        
        # Remove oldest until sufficient space
        for key, value in sorted_items:
            del self.cache[key]
            self.current_size -= value['size']
            
            if self.current_size < self.max_size * 0.7:
                break
```

### Embedding Cache

```python
class EmbeddingCache:
    def __init__(self, cache_backend='redis'):
        self.cache = self._init_backend(cache_backend)
        self.local_cache = {}  # For fast local access
    
    def get_embedding(self, text):
        """Get cached embedding"""
        # Try local first
        if text in self.local_cache:
            return self.local_cache[text]
        
        # Try distributed cache
        cached = self.cache.get(f"emb:{text}")
        
        if cached is not None:
            # Store in local cache
            self.local_cache[text] = cached
            return cached
        
        return None
    
    def cache_embedding(self, text, embedding):
        """Cache embedding"""
        # Local cache
        self.local_cache[text] = embedding
        
        # Distributed cache with expiry
        self.cache.set(
            f"emb:{text}",
            embedding,
            expire_seconds=86400  # 24 hours
        )
    
    def batch_cache_embeddings(self, text_embedding_pairs):
        """Cache multiple embeddings efficiently"""
        # Local cache
        for text, emb in text_embedding_pairs:
            self.local_cache[text] = emb
        
        # Batch set in distributed cache
        pipe = self.cache.pipeline()
        for text, emb in text_embedding_pairs:
            pipe.set(f"emb:{text}", emb, expire_seconds=86400)
        pipe.execute()
```

## 3. Optimized Retrieval

### Approximate Nearest Neighbor Search

```python
import hnswlib

class OptimizedRetriever:
    def __init__(self, dimension=1536, max_elements=1000000):
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        self.documents = []
    
    def index_documents(self, docs, embeddings):
        """Index documents efficiently"""
        doc_ids = np.arange(len(embeddings))
        self.index.add_items(embeddings, doc_ids)
        self.documents = docs
    
    def retrieve_fast(self, query_embedding, k=10, ef=100):
        """Ultra-fast retrieval with HNSW"""
        self.index.set_ef(ef)
        
        labels, distances = self.index.knn_query(
            query_embedding.reshape(1, -1),
            k=k
        )
        
        results = []
        for idx, distance in zip(labels[0], distances[0]):
            similarity = 1 - distance  # Convert distance to similarity
            results.append({
                'document': self.documents[idx],
                'score': similarity
            })
        
        return results
```

### Quantization for Speed

```python
class QuantizedRetriever:
    """Use quantized embeddings for faster search"""
    
    def __init__(self, original_dimension=1536, quantized_dimension=128):
        self.original_dim = original_dimension
        self.quantized_dim = quantized_dimension
        self.quantizer = None
    
    def fit_quantizer(self, embeddings):
        """Fit quantization on sample embeddings"""
        from sklearn.decomposition import PCA
        
        # Dimensionality reduction
        self.quantizer = PCA(n_components=self.quantized_dim)
        self.quantizer.fit(embeddings)
    
    def quantize_embedding(self, embedding):
        """Quantize embedding for faster search"""
        quantized = self.quantizer.transform(
            embedding.reshape(1, -1)
        )
        return quantized[0]
    
    def retrieve_with_quantized(self, query_embedding, candidates, k=10):
        """Two-stage retrieval: quantized then refined"""
        # Stage 1: Fast search with quantized embeddings
        quantized_query = self.quantize_embedding(query_embedding)
        
        scores = []
        for candidate in candidates:
            quantized_doc = self.quantize_embedding(candidate['embedding'])
            score = np.dot(quantized_query, quantized_doc)
            scores.append((candidate, score))
        
        # Get top 2*k from quantized search
        top_candidates = sorted(scores, key=lambda x: x[1], reverse=True)[:k*2]
        
        # Stage 2: Refine with full embeddings
        refined = []
        for candidate, _ in top_candidates:
            score = np.dot(query_embedding, candidate['embedding'])
            refined.append((candidate, score))
        
        return sorted(refined, key=lambda x: x[1], reverse=True)[:k]
```

## 4. Streaming Context Assembly

### Progressive Context Building

```python
class StreamingContextAssembler:
    """Build context progressively for faster generation start"""
    
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens
    
    def assemble_context_streaming(self, documents):
        """Yield context chunks as they're processed"""
        token_count = 0
        
        for doc in documents:
            doc_tokens = len(doc['text'].split())
            
            if token_count + doc_tokens <= self.max_tokens:
                yield doc['text']
                token_count += doc_tokens
            else:
                # Truncate last document
                remaining_tokens = self.max_tokens - token_count
                truncated = ' '.join(
                    doc['text'].split()[:remaining_tokens]
                )
                yield truncated
                break
    
    def get_initial_context(self, documents, initial_token_count=500):
        """Get initial context quickly for fast LLM start"""
        context_parts = []
        token_count = 0
        
        for doc in documents:
            if token_count >= initial_token_count:
                break
            
            doc_tokens = len(doc['text'].split())
            context_parts.append(doc['text'])
            token_count += doc_tokens
        
        return "\n\n".join(context_parts)
```

## 5. Early Termination

### Adaptive Retrieval Depth

```python
class AdaptiveRetrievalDepth:
    """Retrieve only as much as needed"""
    
    def __init__(self, confidence_threshold=0.85):
        self.threshold = confidence_threshold
    
    def retrieve_adaptive(self, query_embedding, index):
        """Retrieve with early stopping"""
        results = []
        top_k = 1
        
        while top_k <= 100:
            # Retrieve batch
            batch = index.retrieve(query_embedding, top_k=top_k)
            results = batch
            
            # Assess confidence
            if len(results) >= 5:
                confidence = self._assess_confidence(
                    query_embedding,
                    results
                )
                
                if confidence >= self.threshold:
                    return results
            
            # Increment search depth
            top_k = int(top_k * 1.5)
        
        return results[:20]  # Return top 20 max
    
    def _assess_confidence(self, query_emb, results):
        """Assess confidence in result set"""
        if len(results) < 5:
            return 0
        
        # Confidence based on score distribution
        scores = [r['score'] for r in results]
        
        # High confidence if top result much better than others
        if scores[0] - np.mean(scores[1:]) > 0.3:
            return 0.9
        
        # Medium confidence if scores are consistent
        if np.std(scores) < 0.1:
            return 0.7
        
        return 0.5
```

## 6. Batch Processing for Throughput

### Request Batching

```python
from queue import Queue
import threading

class RequestBatcher:
    """Batch multiple requests for better throughput"""
    
    def __init__(self, batch_size=32, batch_timeout_ms=100):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000
        self.queue = Queue()
        self.results = {}
    
    def process_request(self, request_id, query):
        """Process request (may be batched)"""
        self.queue.put((request_id, query))
        
        # Start batch processor if not running
        if not hasattr(self, 'processor_thread') or not self.processor_thread.is_alive():
            self.processor_thread = threading.Thread(
                target=self._process_batches
            )
            self.processor_thread.start()
        
        return self.get_result_async(request_id)
    
    def _process_batches(self):
        """Process batches of requests"""
        batch = []
        last_process_time = time.time()
        
        while True:
            try:
                # Wait for items or timeout
                request_id, query = self.queue.get(
                    timeout=self.batch_timeout
                )
                batch.append((request_id, query))
                
                # Process when batch full or timeout
                if len(batch) >= self.batch_size or \
                   (time.time() - last_process_time) >= self.batch_timeout:
                    self._process_batch(batch)
                    batch = []
                    last_process_time = time.time()
            
            except:
                if batch:
                    self._process_batch(batch)
                    batch = []
    
    def _process_batch(self, batch):
        """Process batch of requests efficiently"""
        # Encode all queries together
        queries = [q for _, q in batch]
        embeddings = self.encode_batch(queries)
        
        # Retrieve for all queries
        for (request_id, query), embedding in zip(batch, embeddings):
            result = self.retrieve(embedding)
            self.results[request_id] = result
    
    def get_result_async(self, request_id):
        """Get result when ready"""
        import asyncio
        
        async def wait_for_result():
            while request_id not in self.results:
                await asyncio.sleep(0.01)
            return self.results.pop(request_id)
        
        return wait_for_result()
```

## 7. Latency Optimization Checklist

- [ ] Enable query caching (typical 60-80% hit rate)
- [ ] Use HNSW or similar ANN for vector search
- [ ] Cache embeddings aggressively
- [ ] Enable streaming/progressive context
- [ ] Implement adaptive retrieval depth
- [ ] Use quantization for initial filtering
- [ ] Batch process requests
- [ ] Monitor P99 latency closely
- [ ] Profile each component regularly
- [ ] Plan for scale (latency increases with index size)

## 8. Measuring Real-Time Performance

```python
class RealtimeMetricsCollector:
    """Collect real-time performance metrics"""
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.latencies = []
        self.throughputs = []
    
    def record_latency(self, latency_ms):
        """Record request latency"""
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
    
    def get_latency_stats(self):
        """Get current latency statistics"""
        if not self.latencies:
            return None
        
        return {
            'p50': np.percentile(self.latencies, 50),
            'p95': np.percentile(self.latencies, 95),
            'p99': np.percentile(self.latencies, 99),
            'mean': np.mean(self.latencies),
            'max': np.max(self.latencies)
        }
```

## Conclusion

Real-time RAG requires careful optimization across all components. The biggest gains come from caching (40-50% latency reduction), approximate search (30-40% reduction), and batch processing (2-3x throughput). Always profile your specific setup and optimize the bottleneck, not the average case.
