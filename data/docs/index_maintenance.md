# Index Maintenance: Keeping RAG Systems Fresh

## Introduction

As RAG systems grow, maintaining index quality and freshness becomes critical. This guide covers strategies for updating, optimizing, and debugging indices.

## 1. Index Updates

### Incremental Indexing

```python
from datetime import datetime
from typing import List, Dict

class IncrementalIndexer:
    """Manage incremental index updates"""
    
    def __init__(self, base_index):
        self.base_index = base_index
        self.pending_updates = []
        self.deleted_docs = set()
        self.update_log = []
    
    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """Queue document for addition"""
        self.pending_updates.append({
            'operation': 'add',
            'doc_id': doc_id,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        })
    
    def remove_document(self, doc_id: str):
        """Queue document for removal"""
        self.deleted_docs.add(doc_id)
        self.pending_updates.append({
            'operation': 'remove',
            'doc_id': doc_id,
            'timestamp': datetime.now()
        })
    
    def update_document(self, doc_id: str, content: str, metadata: dict = None):
        """Queue document for update"""
        self.pending_updates.append({
            'operation': 'update',
            'doc_id': doc_id,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        })
    
    def flush_updates(self, batch_size: int = 100):
        """Apply pending updates to index"""
        
        for i in range(0, len(self.pending_updates), batch_size):
            batch = self.pending_updates[i:i+batch_size]
            self._apply_batch(batch)
        
        # Log flush
        self.update_log.append({
            'timestamp': datetime.now(),
            'updates_applied': len(self.pending_updates)
        })
        
        self.pending_updates = []
    
    def _apply_batch(self, batch: List[Dict]):
        """Apply batch of updates"""
        
        for update in batch:
            if update['operation'] == 'add':
                embedding = self._embed(update['content'])
                self.base_index.add(
                    update['doc_id'],
                    embedding,
                    update['metadata']
                )
            
            elif update['operation'] == 'remove':
                self.base_index.remove(update['doc_id'])
            
            elif update['operation'] == 'update':
                self.base_index.remove(update['doc_id'])
                embedding = self._embed(update['content'])
                self.base_index.add(
                    update['doc_id'],
                    embedding,
                    update['metadata']
                )
```

## 2. Index Optimization

### Defragmentation and Reindexing

```python
class IndexOptimizer:
    """Optimize index performance"""
    
    def __init__(self, index):
        self.index = index
    
    def measure_fragmentation(self):
        """Measure index fragmentation"""
        # Fragmentation: ratio of deleted/total documents
        
        stats = self.index.stats()
        
        deleted = stats.get('deleted_docs', 0)
        total = stats.get('total_docs', 0)
        
        fragmentation = deleted / total if total > 0 else 0
        
        return fragmentation
    
    def should_defragment(self, fragmentation_threshold: float = 0.3):
        """Determine if defragmentation needed"""
        fragmentation = self.measure_fragmentation()
        return fragmentation > fragmentation_threshold
    
    def defragment(self):
        """Defragment index"""
        print("Starting index defragmentation...")
        
        # Extract all documents
        all_docs = self.index.list_all()
        
        # Clear index
        self.index.clear()
        
        # Reindex all documents
        for doc in all_docs:
            self.index.add(doc['id'], doc['embedding'], doc['metadata'])
        
        print(f"Defragmented index with {len(all_docs)} documents")
    
    def optimize_index(self):
        """Run all optimizations"""
        
        optimizations = []
        
        # Check and defragment if needed
        if self.should_defragment():
            self.defragment()
            optimizations.append('defragmentation')
        
        # Rebuild indices
        self.index.rebuild_indices()
        optimizations.append('index_rebuild')
        
        # Compact storage
        self.index.compact()
        optimizations.append('storage_compaction')
        
        return optimizations
```

## 3. Index Monitoring

### Health Checks and Diagnostics

```python
class IndexMonitor:
    """Monitor index health"""
    
    def __init__(self, index):
        self.index = index
        self.metrics_history = []
    
    def check_health(self):
        """Comprehensive health check"""
        
        health_report = {
            'timestamp': datetime.now(),
            'checks': {}
        }
        
        # Check 1: Index size
        size_status = self._check_size()
        health_report['checks']['size'] = size_status
        
        # Check 2: Document count consistency
        count_status = self._check_doc_count()
        health_report['checks']['count'] = count_status
        
        # Check 3: Query performance
        perf_status = self._check_query_performance()
        health_report['checks']['performance'] = perf_status
        
        # Check 4: Embedding validity
        emb_status = self._check_embeddings()
        health_report['checks']['embeddings'] = emb_status
        
        # Check 5: Corruption detection
        corr_status = self._detect_corruption()
        health_report['checks']['corruption'] = corr_status
        
        # Overall health
        all_healthy = all(check.get('healthy', False) for check in health_report['checks'].values())
        health_report['healthy'] = all_healthy
        health_report['severity'] = 'critical' if not all_healthy else 'ok'
        
        self.metrics_history.append(health_report)
        
        return health_report
    
    def _check_size(self):
        """Check index size is reasonable"""
        size_bytes = self.index.size_bytes()
        doc_count = self.index.doc_count()
        
        # Average bytes per document
        avg_per_doc = size_bytes / doc_count if doc_count > 0 else 0
        
        # Expect ~5KB per document (embeddings + metadata)
        healthy = 1000 < avg_per_doc < 50000
        
        return {
            'healthy': healthy,
            'size_mb': size_bytes / 1024 / 1024,
            'avg_per_doc': avg_per_doc,
            'doc_count': doc_count
        }
    
    def _check_doc_count(self):
        """Verify document count consistency"""
        indexed_count = self.index.doc_count()
        
        # Verify count matches actual documents
        actual_count = len(self.index.list_all())
        
        healthy = indexed_count == actual_count
        
        return {
            'healthy': healthy,
            'indexed_count': indexed_count,
            'actual_count': actual_count
        }
    
    def _check_query_performance(self):
        """Check query latency"""
        import time
        
        # Run benchmark query
        start = time.time()
        _ = self.index.search("test query", top_k=10)
        latency = (time.time() - start) * 1000  # ms
        
        # Healthy if <500ms
        healthy = latency < 500
        
        return {
            'healthy': healthy,
            'latency_ms': latency,
            'warning_threshold_ms': 500
        }
    
    def _check_embeddings(self):
        """Verify embedding validity"""
        sample_docs = self.index.sample_docs(n=100)
        
        invalid_count = 0
        for doc in sample_docs:
            embedding = doc['embedding']
            
            # Check for NaN/Inf
            if not np.isfinite(embedding).all():
                invalid_count += 1
        
        healthy = invalid_count == 0
        
        return {
            'healthy': healthy,
            'invalid_count': invalid_count,
            'checked_count': len(sample_docs)
        }
    
    def _detect_corruption(self):
        """Detect index corruption"""
        # Try to read and verify index structure
        try:
            test_read = self.index.verify_integrity()
            healthy = test_read
        except:
            healthy = False
        
        return {
            'healthy': healthy,
            'message': 'Index structure valid' if healthy else 'Corruption detected'
        }
```

## 4. Version Control

### Managing Multiple Index Versions

```python
class IndexVersionControl:
    """Manage multiple versions of index"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.versions = {}
        self.current_version = None
    
    def create_snapshot(self, label: str = None):
        """Create snapshot of current index"""
        
        timestamp = datetime.now().isoformat()
        version_id = f"v_{timestamp}"
        
        if label:
            version_id = f"{label}_{version_id}"
        
        # Copy index to versioned location
        version_path = f"{self.base_path}/versions/{version_id}"
        
        self.versions[version_id] = {
            'timestamp': timestamp,
            'label': label,
            'path': version_path,
            'status': 'active'
        }
        
        self.current_version = version_id
        
        return version_id
    
    def rollback(self, version_id: str):
        """Rollback to specific version"""
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        # Restore from backup
        self.current_version = version_id
        
        return self.versions[version_id]
    
    def list_versions(self):
        """List all available versions"""
        return [
            {
                **v,
                'current': version_id == self.current_version
            }
            for version_id, v in self.versions.items()
        ]
    
    def cleanup_old_versions(self, keep_count: int = 5):
        """Remove old versions, keep recent N"""
        
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        to_delete = sorted_versions[keep_count:]
        
        for version_id, _ in to_delete:
            del self.versions[version_id]
```

## 5. Refresh Strategies

### Keeping Index Fresh

```python
class IndexRefreshStrategy:
    """Strategy for refreshing index"""
    
    def __init__(self, index, refresh_interval_hours: int = 24):
        self.index = index
        self.refresh_interval = refresh_interval_hours * 3600  # seconds
        self.last_refresh = datetime.now()
    
    def should_refresh(self):
        """Determine if refresh needed"""
        time_since_refresh = (datetime.now() - self.last_refresh).total_seconds()
        return time_since_refresh > self.refresh_interval
    
    def refresh_by_age(self, max_age_days: int = 30):
        """Refresh documents older than threshold"""
        
        old_docs = self.index.find_by_age(max_age_days)
        
        refreshed_count = 0
        for doc in old_docs:
            # Fetch fresh content
            fresh_content = self._fetch_fresh_content(doc['id'])
            
            if fresh_content:
                # Update in index
                embedding = self._embed(fresh_content)
                self.index.update(doc['id'], embedding)
                refreshed_count += 1
        
        self.last_refresh = datetime.now()
        
        return refreshed_count
    
    def refresh_changed_sources(self):
        """Refresh documents whose sources changed"""
        
        refreshed_count = 0
        
        for doc in self.index.list_all():
            source_hash = self._get_source_hash(doc['source'])
            
            if source_hash != doc.get('source_hash'):
                # Source changed, update document
                fresh_content = self._fetch_fresh_content(doc['id'])
                
                embedding = self._embed(fresh_content)
                metadata = doc['metadata'].copy()
                metadata['source_hash'] = source_hash
                
                self.index.update(doc['id'], embedding, metadata)
                refreshed_count += 1
        
        return refreshed_count
    
    def _fetch_fresh_content(self, doc_id: str):
        """Fetch fresh content for document"""
        # Implementation depends on source
        pass
    
    def _get_source_hash(self, source: str):
        """Get hash of source to detect changes"""
        import hashlib
        return hashlib.md5(source.encode()).hexdigest()
```

## Best Practices

1. **Regular monitoring** - weekly health checks recommended
2. **Incremental updates** - batch updates to avoid disruptions
3. **Snapshots before major operations** - easy rollback
4. **Monitor fragmentation** - defragment when >30%
5. **Refresh stale content** - keep index fresh
6. **Audit trails** - track all modifications
7. **Test on replicas** - verify changes before production

## Conclusion

Index maintenance is critical for long-term RAG success. Regular monitoring, incremental updates, and refresh strategies keep indices fresh and performant. Most issues come from accumulated fragmentation and stale content; proactive maintenance prevents problems.
