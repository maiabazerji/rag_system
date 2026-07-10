# Production Deployment: Running RAG at Scale

## Introduction

Deploying RAG to production requires careful planning for reliability, scalability, and monitoring. This guide covers production deployment patterns and best practices.

## 1. Architecture Design

### High-Level Architecture

```python
from dataclasses import dataclass
from typing import List

@dataclass
class RAGArchitecture:
    """Production RAG architecture specification"""
    
    # Components
    document_storage: str  # Where source docs live
    vector_db: str        # Where embeddings stored
    cache_layer: str      # Redis, memcached, etc
    embedding_service: str  # Local or API
    llm_service: str      # OpenAI, Claude, local, etc
    
    # Redundancy
    vector_db_replicas: int = 3
    llm_replicas: int = 2
    cache_replicas: int = 2
    
    # Monitoring
    logging_backend: str = "elasticsearch"
    metrics_backend: str = "prometheus"
    tracing_backend: str = "jaeger"
    
    # Infrastructure
    primary_region: str
    backup_regions: List[str]
    load_balancer: str = "nginx"

class DeploymentPlanner:
    """Plan RAG deployment"""
    
    @staticmethod
    def recommend_architecture(scale: str):
        """Recommend architecture based on scale"""
        
        if scale == 'small':  # < 100K queries/day
            return {
                'embedding_service': 'local_cpu',
                'vector_db': 'qdrant_self_hosted',
                'llm_service': 'api_cached',
                'cache': 'redis_single',
                'replicas': 1
            }
        
        elif scale == 'medium':  # 100K - 1M queries/day
            return {
                'embedding_service': 'local_gpu',
                'vector_db': 'pinecone',
                'llm_service': 'api_with_fallback',
                'cache': 'redis_cluster',
                'replicas': 3
            }
        
        elif scale == 'large':  # > 1M queries/day
            return {
                'embedding_service': 'multi_gpu_cluster',
                'vector_db': 'milvus_distributed',
                'llm_service': 'multi_region',
                'cache': 'redis_cluster_multi_region',
                'replicas': 5
            }
```

## 2. Docker Containerization

### Building Container Images

```dockerfile
# Dockerfile for RAG service
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag
        image: rag-service:latest
        ports:
        - containerPort: 8000
        
        # Resource requests/limits
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        
        # Environment
        env:
        - name: VECTOR_DB_URL
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: vector-db-url
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: llm-api-key
```

## 3. Load Balancing

### Request Distribution

```python
from typing import List
import random
import time

class LoadBalancer:
    """Distribute requests across backends"""
    
    def __init__(self, backends: List[str]):
        self.backends = backends
        self.health_status = {b: True for b in backends}
        self.request_count = {b: 0 for b in backends}
    
    def select_backend(self, strategy: str = 'least_loaded'):
        """Select backend for request"""
        
        # Filter healthy backends
        healthy = [b for b in self.backends if self.health_status[b]]
        
        if not healthy:
            # Fallback to any backend
            healthy = self.backends
        
        if strategy == 'round_robin':
            selected = healthy[self.request_count[healthy[0]] % len(healthy)]
        
        elif strategy == 'least_loaded':
            selected = min(healthy, key=lambda b: self.request_count[b])
        
        elif strategy == 'random':
            selected = random.choice(healthy)
        
        self.request_count[selected] += 1
        
        return selected
    
    def health_check(self):
        """Check backend health"""
        
        for backend in self.backends:
            try:
                # Simple health check
                import requests
                response = requests.get(f"http://{backend}/health", timeout=2)
                self.health_status[backend] = response.status_code == 200
            except:
                self.health_status[backend] = False
```

## 4. Monitoring and Observability

### Comprehensive Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge
import time

class RAGMetrics:
    """Prometheus metrics for RAG"""
    
    def __init__(self):
        # Counters
        self.queries_total = Counter(
            'rag_queries_total',
            'Total queries processed',
            ['status']
        )
        
        self.errors_total = Counter(
            'rag_errors_total',
            'Total errors',
            ['error_type']
        )
        
        # Histograms
        self.query_latency = Histogram(
            'rag_query_latency_seconds',
            'Query latency in seconds',
            buckets=[0.1, 0.5, 1, 2, 5, 10]
        )
        
        self.retrieval_latency = Histogram(
            'rag_retrieval_latency_seconds',
            'Retrieval latency'
        )
        
        self.generation_latency = Histogram(
            'rag_generation_latency_seconds',
            'Generation latency'
        )
        
        # Gauges
        self.cache_size = Gauge(
            'rag_cache_size_bytes',
            'Cache size in bytes'
        )
        
        self.queue_depth = Gauge(
            'rag_queue_depth',
            'Request queue depth'
        )
    
    def record_query(self, duration: float, status: str):
        """Record query metrics"""
        self.queries_total.labels(status=status).inc()
        self.query_latency.observe(duration)
    
    def record_error(self, error_type: str):
        """Record error"""
        self.errors_total.labels(error_type=error_type).inc()
```

### Logging

```python
import logging
import json
from pythonjsonlogger import jsonlogger

class RAGLogger:
    """Structured logging for RAG"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
        # JSON formatter
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        logHandler.setFormatter(formatter)
        
        self.logger.addHandler(logHandler)
        self.logger.setLevel(logging.INFO)
    
    def log_query(self, query: str, results: int, latency: float):
        """Log query with metrics"""
        self.logger.info(
            "query_processed",
            extra={
                'query': query[:100],
                'results_count': results,
                'latency_ms': latency * 1000
            }
        )
    
    def log_error(self, error: str, traceback: str):
        """Log error with context"""
        self.logger.error(
            "error_occurred",
            extra={
                'error': error,
                'traceback': traceback
            }
        )
```

## 5. Failover and Recovery

### High Availability Setup

```python
class FailoverManager:
    """Manage failover and recovery"""
    
    def __init__(self, primary: str, backup: str):
        self.primary = primary
        self.backup = backup
        self.is_primary_active = True
    
    def health_check_primary(self):
        """Check primary health"""
        import requests
        
        try:
            response = requests.get(
                f"http://{self.primary}/health",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False
    
    def failover_to_backup(self):
        """Failover to backup"""
        
        if not self.health_check_primary():
            # Update routing to backup
            self.is_primary_active = False
            
            # Update load balancer
            self._update_load_balancer(self.backup)
            
            # Alert operations
            self._send_alert(
                "Primary failure: failover to backup activated"
            )
            
            return True
        
        return False
    
    def failback_to_primary(self):
        """Failback to primary after recovery"""
        
        if self.health_check_primary() and not self.is_primary_active:
            self.is_primary_active = True
            
            # Update routing
            self._update_load_balancer(self.primary)
            
            # Alert operations
            self._send_alert(
                "Primary recovered: failback activated"
            )
            
            return True
        
        return False
    
    def _update_load_balancer(self, target: str):
        """Update load balancer routing"""
        pass
    
    def _send_alert(self, message: str):
        """Send alert to operations"""
        pass
```

## 6. Deployment Checklist

```python
class DeploymentChecklist:
    """Verification checklist for production deployment"""
    
    def __init__(self):
        self.checks = {
            'infrastructure': [
                'Load balancer configured',
                'Replicas deployed (≥3)',
                'Health checks enabled',
                'Network policies configured'
            ],
            'monitoring': [
                'Prometheus configured',
                'Logging pipeline active',
                'Alerts configured',
                'Dashboards created'
            ],
            'security': [
                'API authentication enabled',
                'Rate limiting configured',
                'Data encryption enabled',
                'Secrets management setup'
            ],
            'testing': [
                'Load testing passed',
                'Failover tested',
                'Recovery tested',
                'Disaster recovery plan'
            ],
            'documentation': [
                'Runbook created',
                'Architecture documented',
                'Deployment guide written',
                'Troubleshooting guide'
            ]
        }
    
    def verify_deployment(self):
        """Verify all checks pass"""
        all_passed = True
        
        for category, checks in self.checks.items():
            print(f"\n{category.upper()}:")
            for check in checks:
                print(f"  - {check}")
        
        return all_passed
```

## Best Practices

1. **Container everything** - reproducible deployments
2. **Use orchestration** - Kubernetes for scaling
3. **Multi-region** - disaster recovery
4. **Blue-green deployment** - zero-downtime updates
5. **Comprehensive monitoring** - catch issues early
6. **Automated rollback** - quick recovery from issues
7. **Load test before production** - understand limits
8. **Graceful degradation** - fallback to simpler retrieval

## Conclusion

Production RAG requires careful architecture, robust monitoring, and failover planning. Cloud-native deployment patterns (containers, orchestration, IaC) enable reliable operations at scale. Start simple, add redundancy incrementally based on actual failure modes.
