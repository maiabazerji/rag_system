# Cache Strategies for RAG: Performance Optimization

## Introduction

Caching is one of the most effective performance optimizations for RAG systems. This guide covers caching strategies at different layers of the stack.

## 1. Query Result Caching

### Exact Query Match Caching

```python
from typing import Dict, Optional
import hashlib
from datetime import datetime, timedelta

class QueryResultCache:
    """Cache query results"""
    
    def __init__(self, ttl_seconds: int = 3600, max_size_mb: int = 500):
        self.ttl = ttl_seconds
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.current_size = 0
        self.hits = 0
        self.misses = 0
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached result"""
        cache_key = self._hash_query(query)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check if expired
            if datetime.now() < entry['expires']:
                self.hits += 1
                return entry['result']
            else:
                # Expired, remove
                del self.cache[cache_key]
        
        self.misses += 1
        return None
    
    def set(self, query: str, result: Dict):
        """Cache result"""
        cache_key = self._hash_query(query)
        
        result_size = self._estimate_size(result)
        
        # Evict if necessary
        if self.current_size + result_size > self.max_size_bytes:
            self._evict_lru()
        
        self.cache[cache_key] = {
            'result': result,
            'expires': datetime.now() + timedelta(seconds=self.ttl),
            'created': datetime.now(),
            'size': result_size,
            'hits': 0
        }
        
        self.current_size += result_size
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'current_size_mb': self.current_size / 1024 / 1024,
            'entries': len(self.cache)
        }
    
    def _hash_query(self, query: str):
        """Create hash key from query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _estimate_size(self, obj):
        """Estimate object size"""
        import sys
        return sys.getsizeof(obj)
    
    def _evict_lru(self):
        """Evict least recently used"""
        if not self.cache:
            return
        
        # Sort by creation time
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1]['created']
        )
        
        # Remove oldest until sufficient space
        for key, entry in sorted_items:
            del self.cache[key]
            self.current_size -= entry['size']
            
            if self.current_size < self.max_size_bytes * 0.8:
                break
```

## 2. Embedding Cache

### Caching Vector Embeddings

```python
class EmbeddingCache:
    """Cache text embeddings"""
    
    def __init__(self, backend='memory'):
        self.backend = backend
        self.memory_cache = {}
        
        if backend == 'redis':
            import redis
            self.redis_client = redis.Redis()
    
    def get_embedding(self, text: str):
        """Retrieve cached embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Try memory cache first
        if text_hash in self.memory_cache:
            return self.memory_cache[text_hash]
        
        # Try persistent cache
        if self.backend == 'redis':
            cached = self.redis_client.get(f"emb:{text_hash}")
            if cached:
                import pickle
                return pickle.loads(cached)
        
        return None
    
    def set_embedding(self, text: str, embedding):
        """Cache embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Memory cache
        self.memory_cache[text_hash] = embedding
        
        # Persistent cache
        if self.backend == 'redis':
            import pickle
            self.redis_client.setex(
                f"emb:{text_hash}",
                86400,  # 24 hours
                pickle.dumps(embedding)
            )
    
    def batch_get(self, texts: list):
        """Get multiple embeddings"""
        results = []
        missing_texts = []
        missing_indices = []
        
        for i, text in enumerate(texts):
            cached = self.get_embedding(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                missing_texts.append(text)
                missing_indices.append(i)
        
        return results, missing_texts, missing_indices
```

## 3. Retrieval Result Caching

### Caching Document Retrieval

```python
class RetrievalCache:
    """Cache retrieval results"""
    
    def __init__(self):
        self.query_cache = {}
    
    def get_cached_retrieval(self, query: str, top_k: int):
        """Get cached retrieval results"""
        cache_key = f"{query}:k{top_k}"
        return self.query_cache.get(cache_key)
    
    def cache_retrieval(self, query: str, top_k: int, results: list):
        """Cache retrieval results"""
        cache_key = f"{query}:k{top_k}"
        self.query_cache[cache_key] = {
            'results': results,
            'cached_at': datetime.now()
        }
    
    def invalidate_query(self, query: str):
        """Invalidate query cache"""
        to_delete = [k for k in self.query_cache if k.startswith(query)]
        for key in to_delete:
            del self.query_cache[key]
    
    def invalidate_document(self, doc_id: str):
        """Invalidate all queries that returned this doc"""
        to_delete = []
        
        for cache_key, entry in self.query_cache.items():
            for result in entry['results']:
                if result['id'] == doc_id:
                    to_delete.append(cache_key)
                    break
        
        for key in to_delete:
            del self.query_cache[key]
```

## 4. Multi-Layer Caching

### Hierarchical Cache Architecture

```python
class HierarchicalCache:
    """Multi-layer caching system"""
    
    def __init__(self):
        self.l1_cache = {}  # In-process memory
        self.l2_cache = None  # Redis or memcached
        self.l3_cache = None  # Database
    
    def get(self, key: str, layer=None):
        """Get from appropriate layer"""
        
        # Try L1
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Try L2
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value:
                self.l1_cache[key] = value  # Promote to L1
                return value
        
        # Try L3
        if self.l3_cache:
            value = self.l3_cache.get(key)
            if value:
                self._promote_to_l2(key, value)
                return value
        
        return None
    
    def set(self, key: str, value, ttl_seconds: int = 3600):
        """Set across layers"""
        
        # L1: always
        self.l1_cache[key] = value
        
        # L2: if configured
        if self.l2_cache:
            self.l2_cache.set(key, value, expire_seconds=ttl_seconds)
        
        # L3: if configured
        if self.l3_cache:
            self.l3_cache.set(key, value, ttl_seconds)
    
    def _promote_to_l2(self, key: str, value):
        """Promote value from L3 to L2"""
        if self.l2_cache:
            self.l2_cache.set(key, value, expire_seconds=3600)
```

## 5. Cache Invalidation

### Strategies for Cache Freshness

```python
class CacheInvalidationStrategy:
    """Manage cache invalidation"""
    
    def __init__(self, cache):
        self.cache = cache
    
    def time_based_invalidation(self, ttl_seconds: int):
        """Invalidate based on age"""
        now = datetime.now()
        
        to_delete = []
        for key, entry in self.cache.items():
            age = (now - entry['created']).total_seconds()
            
            if age > ttl_seconds:
                to_delete.append(key)
        
        for key in to_delete:
            del self.cache[key]
    
    def dependency_based_invalidation(self, updated_doc_id: str):
        """Invalidate dependent queries when document updates"""
        
        # Find queries that depend on this document
        dependent_queries = self._find_dependent_queries(updated_doc_id)
        
        for query in dependent_queries:
            # Invalidate all cache entries for this query
            to_delete = [k for k in self.cache if k.startswith(query)]
            for key in to_delete:
                del self.cache[key]
    
    def event_based_invalidation(self, event: str, affected_items: list):
        """Invalidate on specific events"""
        
        if event == 'document_added':
            # Invalidate general queries
            self._invalidate_by_pattern("*")
        
        elif event == 'document_updated':
            # Invalidate specific queries
            for doc_id in affected_items:
                self.dependency_based_invalidation(doc_id)
        
        elif event == 'index_reindexed':
            # Flush all caches
            self.cache.clear()
    
    def _find_dependent_queries(self, doc_id: str):
        """Find queries that returned this document"""
        dependent = []
        
        for key, entry in self.cache.items():
            if 'results' in entry:
                for result in entry['results']:
                    if result.get('id') == doc_id:
                        dependent.append(key)
                        break
        
        return dependent
    
    def _invalidate_by_pattern(self, pattern: str):
        """Invalidate matching patterns"""
        import fnmatch
        
        to_delete = [
            k for k in self.cache
            if fnmatch.fnmatch(k, pattern)
        ]
        
        for key in to_delete:
            del self.cache[key]
```

## 6. Cache Metrics and Monitoring

### Tracking Cache Performance

```python
class CacheMetrics:
    """Monitor cache performance"""
    
    def __init__(self, cache):
        self.cache = cache
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'access_times': []
        }
    
    def record_hit(self, access_time_ms: float):
        """Record cache hit"""
        self.metrics['hits'] += 1
        self.metrics['access_times'].append(access_time_ms)
    
    def record_miss(self):
        """Record cache miss"""
        self.metrics['misses'] += 1
    
    def get_hit_rate(self):
        """Calculate hit rate"""
        total = self.metrics['hits'] + self.metrics['misses']
        return self.metrics['hits'] / total if total > 0 else 0
    
    def get_avg_access_time(self):
        """Average access time for hits"""
        if self.metrics['access_times']:
            return np.mean(self.metrics['access_times'])
        return 0
    
    def get_metrics_report(self):
        """Generate metrics report"""
        return {
            'hit_rate': self.get_hit_rate(),
            'total_hits': self.metrics['hits'],
            'total_misses': self.metrics['misses'],
            'avg_access_time_ms': self.get_avg_access_time(),
            'cache_size': len(self.cache.cache) if hasattr(self.cache, 'cache') else 0
        }
```

## Best Practices

1. **Multi-level caching** - memory + distributed + persistent
2. **TTL-based expiration** - automatic cache refresh
3. **Monitor hit rates** - target 60-80% for production
4. **Invalidation strategy** - proactive > reactive
5. **Size limits** - prevent unbounded growth
6. **Cache warming** - preload frequent queries
7. **Separate concerns** - cache different types separately

## Performance Impact

- Query caching: 40-50% latency reduction
- Embedding caching: 60-70% compute reduction
- Overall: typical 30-40% system latency improvement

## Conclusion

Caching is one of the highest ROI optimizations for RAG systems. Multi-layer caching (memory + Redis + database) provides both speed and persistence. Proper invalidation strategy is critical to maintain correctness.
