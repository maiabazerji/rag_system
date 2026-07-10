# Scaling RAG Systems: From Thousands to Millions of Documents

## Introduction

Scaling RAG to millions of documents introduces challenges in indexing, retrieval speed, memory usage, and computational resources. This guide covers architectural patterns and techniques for scaling RAG systems.

## 1. Scaling Challenges

### Problem Breakdown

```
Challenge              Scale 100K  Scale 1M   Scale 10M
─────────────────────────────────────────────────────
Index Size             100MB       1GB        10GB
Query Latency p99      50ms        100ms      200ms+
Memory Usage           2GB         20GB       200GB+
Embedding Cost         $5          $50        $500
Reindexing Time        minutes     hours      days
Index Fragmentation    Low         Medium     High
```

## 2. Distributed Indexing

### Sharded Vector Database

```python
class ShardedVectorDB:
    """Distribute index across multiple shards"""
    
    def __init__(self, num_shards: int, shard_capacity: int = 1_000_000):
        self.num_shards = num_shards
        self.shard_capacity = shard_capacity
        self.shards = [self._init_shard() for _ in range(num_shards)]
    
    def add_document(self, doc_id: str, embedding: np.ndarray, metadata: dict):
        """Add document to appropriate shard"""
        # Determine shard using consistent hashing
        shard_id = self._get_shard_id(doc_id)
        
        self.shards[shard_id].add(doc_id, embedding, metadata)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        """Search across all shards"""
        results = []
        
        # Search in parallel across shards
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    shard.search,
                    query_embedding,
                    top_k
                )
                for shard in self.shards
            ]
            
            shard_results = [f.result() for f in futures]
        
        # Merge and rerank results
        merged = self._merge_shard_results(shard_results, top_k)
        
        return merged
    
    def _get_shard_id(self, doc_id: str):
        """Map document to shard using consistent hashing"""
        import hashlib
        
        hash_val = int(
            hashlib.md5(doc_id.encode()).hexdigest(),
            16
        )
        return hash_val % self.num_shards
    
    def _merge_shard_results(self, shard_results, top_k):
        """Merge results from multiple shards"""
        all_results = []
        
        for shard_result in shard_results:
            all_results.extend(shard_result)
        
        # Sort by similarity
        sorted_results = sorted(
            all_results,
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def _init_shard(self):
        """Initialize individual shard"""
        # Each shard is independent vector DB
        from pinecone import Index
        return Index(dimension=1536)
```

### Hierarchical Indexing

```python
class HierarchicalIndex:
    """Multi-level index for efficient retrieval"""
    
    def __init__(self, coarse_level_size: int = 1000, fine_level_size: int = 100):
        self.coarse_level = {}  # Top-level index
        self.fine_level = {}    # Fine-grained index
        self.coarse_level_size = coarse_level_size
        self.fine_level_size = fine_level_size
    
    def build_hierarchical_index(self, embeddings: np.ndarray, doc_ids: list):
        """Build multi-level index"""
        from sklearn.cluster import KMeans
        
        # Coarse level: cluster embeddings
        n_coarse_clusters = len(embeddings) // self.coarse_level_size
        
        coarse_kmeans = KMeans(n_clusters=n_coarse_clusters)
        coarse_labels = coarse_kmeans.fit_predict(embeddings)
        
        # Fine level: cluster within each coarse cluster
        for coarse_id in range(n_coarse_clusters):
            cluster_mask = coarse_labels == coarse_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if cluster_mask[i]]
            
            if len(cluster_embeddings) > self.fine_level_size:
                n_fine_clusters = max(
                    1,
                    len(cluster_embeddings) // self.fine_level_size
                )
                
                fine_kmeans = KMeans(n_clusters=n_fine_clusters)
                fine_labels = fine_kmeans.fit_predict(cluster_embeddings)
                
                self.fine_level[coarse_id] = {
                    'kmeans': fine_kmeans,
                    'labels': fine_labels,
                    'doc_ids': cluster_doc_ids
                }
            else:
                self.fine_level[coarse_id] = {
                    'doc_ids': cluster_doc_ids
                }
        
        self.coarse_level = {
            'kmeans': coarse_kmeans,
            'centroids': coarse_kmeans.cluster_centers_
        }
    
    def search_hierarchical(self, query_embedding: np.ndarray, top_k: int = 10):
        """Search using hierarchical index"""
        # Level 1: Find coarse cluster
        coarse_distances = np.linalg.norm(
            self.coarse_level['centroids'] - query_embedding,
            axis=1
        )
        
        coarse_candidates = np.argsort(coarse_distances)[:3]  # Top 3 coarse clusters
        
        # Level 2: Search within fine clusters
        candidates = []
        
        for coarse_id in coarse_candidates:
            fine_data = self.fine_level.get(coarse_id, {})
            
            if 'kmeans' in fine_data:
                # Search within fine clusters
                fine_distances = np.linalg.norm(
                    fine_data['kmeans'].cluster_centers_ - query_embedding,
                    axis=1
                )
                fine_candidates = np.argsort(fine_distances)[:2]
                
                for fine_id in fine_candidates:
                    mask = fine_data['labels'] == fine_id
                    matching_docs = [
                        fine_data['doc_ids'][i]
                        for i in range(len(mask)) if mask[i]
                    ]
                    candidates.extend(matching_docs)
            else:
                candidates.extend(fine_data.get('doc_ids', []))
        
        return candidates[:top_k]
```

## 3. Incremental Indexing

### Online Index Updates

```python
class IncrementalIndexUpdater:
    """Update index without full reindexing"""
    
    def __init__(self, base_index, update_buffer_size: int = 1000):
        self.base_index = base_index
        self.update_buffer = []
        self.update_buffer_size = update_buffer_size
        self.deleted_docs = set()
    
    def add_document(self, doc_id: str, embedding: np.ndarray, metadata: dict):
        """Add document to buffer"""
        self.update_buffer.append({
            'doc_id': doc_id,
            'embedding': embedding,
            'metadata': metadata,
            'operation': 'add'
        })
        
        # Flush buffer if full
        if len(self.update_buffer) >= self.update_buffer_size:
            self._flush_buffer()
    
    def delete_document(self, doc_id: str):
        """Mark document for deletion"""
        self.deleted_docs.add(doc_id)
        
        # Add to buffer as deletion
        self.update_buffer.append({
            'doc_id': doc_id,
            'operation': 'delete'
        })
    
    def _flush_buffer(self):
        """Flush buffer updates to index"""
        for update in self.update_buffer:
            if update['operation'] == 'add':
                self.base_index.add(
                    update['doc_id'],
                    update['embedding'],
                    update['metadata']
                )
            elif update['operation'] == 'delete':
                self.base_index.delete(update['doc_id'])
        
        self.update_buffer = []
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        """Search including buffered updates"""
        # First search base index
        results = self.base_index.search(query_embedding, top_k * 2)
        
        # Filter deleted documents
        filtered_results = [
            r for r in results
            if r['doc_id'] not in self.deleted_docs
        ]
        
        # Rescore with buffer
        for update in self.update_buffer:
            if update['operation'] == 'add':
                similarity = np.dot(
                    query_embedding,
                    update['embedding']
                )
                filtered_results.append({
                    'doc_id': update['doc_id'],
                    'score': similarity,
                    'metadata': update['metadata']
                })
        
        # Resort and return top-k
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        return filtered_results[:top_k]
```

## 4. Caching at Scale

### Multi-Level Cache

```python
class MultiLevelCache:
    """Cache with local, in-memory, and persistent layers"""
    
    def __init__(self):
        self.local_cache = {}  # In-process
        self.redis_cache = self._init_redis()  # Distributed
        self.persistent_cache = self._init_persistent()  # Database
    
    def get_cached_result(self, query_key: str):
        """Get from fastest available level"""
        # Level 1: Local cache
        if query_key in self.local_cache:
            return self.local_cache[query_key]
        
        # Level 2: Redis
        result = self.redis_cache.get(query_key)
        if result is not None:
            self.local_cache[query_key] = result  # Promote to L1
            return result
        
        # Level 3: Persistent
        result = self.persistent_cache.get(query_key)
        if result is not None:
            self.redis_cache.set(query_key, result)  # Promote to L2
            self.local_cache[query_key] = result
            return result
        
        return None
    
    def cache_result(self, query_key: str, result, ttl_seconds: int = 3600):
        """Cache at all levels"""
        # Local cache (no expiry)
        self.local_cache[query_key] = result
        
        # Redis (with TTL)
        self.redis_cache.set(query_key, result, expire_seconds=ttl_seconds)
        
        # Persistent (permanent unless explicitly removed)
        self.persistent_cache.set(query_key, result)
    
    def invalidate_cache(self, query_key: str):
        """Remove from all levels"""
        if query_key in self.local_cache:
            del self.local_cache[query_key]
        
        self.redis_cache.delete(query_key)
        self.persistent_cache.delete(query_key)
```

## 5. Asynchronous Processing

### Background Indexing

```python
import asyncio
from queue import Queue

class AsyncIndexingWorker:
    """Background indexing without blocking queries"""
    
    def __init__(self, num_workers: int = 4):
        self.task_queue = Queue()
        self.num_workers = num_workers
        self.workers = []
        self._start_workers()
    
    def _start_workers(self):
        """Start background workers"""
        for _ in range(self.num_workers):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True
            )
            worker_thread.start()
            self.workers.append(worker_thread)
    
    def _worker_loop(self):
        """Worker thread processing index updates"""
        while True:
            # Get task from queue
            task = self.task_queue.get()
            
            if task is None:  # Shutdown signal
                break
            
            # Process task
            try:
                self._process_indexing_task(task)
            except Exception as e:
                print(f"Indexing error: {e}")
            
            self.task_queue.task_done()
    
    def _process_indexing_task(self, task):
        """Process single indexing task"""
        doc_id = task['doc_id']
        text = task['text']
        
        # Expensive operations in background
        embedding = self.embedding_model.encode(text)
        
        # Update index
        self.index.add_document(doc_id, embedding, task.get('metadata', {}))
    
    def schedule_indexing(self, doc_id: str, text: str, metadata: dict = None):
        """Schedule document for indexing"""
        task = {
            'doc_id': doc_id,
            'text': text,
            'metadata': metadata or {}
        }
        
        self.task_queue.put(task)
    
    def shutdown(self):
        """Shutdown workers gracefully"""
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        for worker in self.workers:
            worker.join()
```

## 6. Resource Management

### Memory-Efficient Embedding Storage

```python
class CompressedEmbeddingStorage:
    """Store embeddings with compression"""
    
    def __init__(self, compression_method='product_quantization'):
        self.compression_method = compression_method
        self.compressor = self._init_compressor()
    
    def _init_compressor(self):
        """Initialize compression"""
        if self.compression_method == 'product_quantization':
            from sklearn.cluster import KMeans
            
            # Use product quantization
            return KMeans(n_clusters=256)
        
        elif self.compression_method == 'binarization':
            return None  # Use sign of values
    
    def compress_embedding(self, embedding: np.ndarray):
        """Compress embedding"""
        if self.compression_method == 'product_quantization':
            # Split into chunks and quantize
            chunk_size = len(embedding) // 8
            chunks = [
                embedding[i:i+chunk_size]
                for i in range(0, len(embedding), chunk_size)
            ]
            
            quantized = []
            for chunk in chunks:
                cluster_id = self.compressor.predict([chunk])[0]
                quantized.append(cluster_id)
            
            return np.array(quantized, dtype=np.uint8)
        
        elif self.compression_method == 'binarization':
            # Convert to binary (sign)
            return (embedding > 0).astype(np.uint8)
    
    def memory_savings(self):
        """Estimate memory savings"""
        original_size = 1536 * 4  # float32
        compressed_size = 192  # quantized to 8 bits
        
        return {
            'original_bytes': original_size,
            'compressed_bytes': compressed_size,
            'compression_ratio': original_size / compressed_size
        }
```

## 7. Monitoring and Auto-Scaling

### Scaling Metrics

```python
class ScalingMonitor:
    """Monitor when to scale"""
    
    def __init__(self, target_query_latency_ms: int = 100):
        self.target_latency = target_query_latency_ms
        self.metrics = {
            'query_times': [],
            'index_sizes': [],
            'qps': 0
        }
    
    def should_scale_out(self):
        """Determine if should add more shards"""
        if not self.metrics['query_times']:
            return False
        
        p99_latency = np.percentile(self.metrics['query_times'], 99)
        
        if p99_latency > self.target_latency * 1.5:
            return True
        
        return False
    
    def should_consolidate(self):
        """Determine if can consolidate shards"""
        if not self.metrics['query_times']:
            return False
        
        p99_latency = np.percentile(self.metrics['query_times'], 99)
        
        if p99_latency < self.target_latency * 0.5:
            return True
        
        return False
    
    def get_scaling_recommendation(self):
        """Get recommended scaling action"""
        if self.should_scale_out():
            return 'scale_out'
        elif self.should_consolidate():
            return 'consolidate'
        else:
            return 'no_action'
```

## Best Practices for Scaling

1. **Use sharding** from the start - easier to scale horizontally
2. **Implement hierarchical indexing** - reduces search space
3. **Cache aggressively** - most queries are repetitive
4. **Async indexing** - don't block queries for updates
5. **Monitor constantly** - latency increases nonlinearly
6. **Plan for 10x growth** - architecture should handle it
7. **Compression matters** - saves 70-80% memory
8. **Test at scale** - synthetic benchmarks mislead

## Scaling Checklist

- [ ] Baseline performance at 100K documents
- [ ] Test with 1M documents
- [ ] Implement sharding strategy
- [ ] Setup distributed caching
- [ ] Configure auto-scaling triggers
- [ ] Monitor P99 latency
- [ ] Plan index maintenance
- [ ] Document scaling procedures

## Conclusion

Scaling RAG requires thinking in terms of shards, hierarchies, and layers of caching. Most scaling issues come from a single bottleneck (usually embedding computation). Identify bottlenecks early, monitor continuously, and scale strategically rather than reactively.
