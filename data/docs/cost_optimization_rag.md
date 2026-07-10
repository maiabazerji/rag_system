# Cost Optimization for RAG Systems

## Introduction

RAG systems can be expensive due to embedding computations, LLM queries, and infrastructure costs. This guide covers techniques to optimize costs while maintaining quality.

## 1. Embedding Cost Analysis

### Cost Breakdown

```python
class CostCalculator:
    """Calculate RAG system costs"""
    
    def __init__(self):
        # Cost per million tokens (approximate as of 2024)
        self.embedding_costs = {
            'text-embedding-3-large': 0.13,  # $/1M tokens
            'text-embedding-3-small': 0.02,
            'cohere-embed-3': 0.10,
            'local-e5-large': 0.0  # No API cost
        }
        
        self.llm_costs = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # $/1K tokens
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'claude-3-opus': {'input': 0.015, 'output': 0.075},
        }
        
        self.infrastructure_costs = {
            'vector_db_pinecone': 100,  # $/month per pod
            'vector_db_weaviate_cloud': 50,  # $/month
            'vector_db_self_hosted': 20,  # $/month (estimated)
        }
    
    def calculate_monthly_cost(self, config: dict):
        """Calculate monthly cost for configuration"""
        monthly_cost = {}
        
        # Embedding costs
        docs_to_index = config.get('docs_per_month', 0)
        avg_doc_tokens = config.get('avg_doc_tokens', 300)
        embedding_model = config.get('embedding_model', 'text-embedding-3-large')
        
        embedding_tokens = docs_to_index * avg_doc_tokens
        monthly_cost['embedding'] = (
            embedding_tokens / 1_000_000 * 
            self.embedding_costs.get(embedding_model, 0.13)
        )
        
        # Query costs
        queries_per_month = config.get('queries_per_month', 100000)
        avg_query_tokens = config.get('avg_query_tokens', 50)
        embedding_model = config.get('embedding_model', 'text-embedding-3-large')
        
        query_embedding_tokens = queries_per_month * avg_query_tokens
        monthly_cost['query_embedding'] = (
            query_embedding_tokens / 1_000_000 *
            self.embedding_costs.get(embedding_model, 0.13)
        )
        
        # LLM generation costs
        avg_context_tokens = config.get('avg_context_tokens', 1000)
        llm_model = config.get('llm_model', 'gpt-3.5-turbo')
        
        total_input_tokens = queries_per_month * (avg_query_tokens + avg_context_tokens)
        total_output_tokens = queries_per_month * config.get('avg_response_tokens', 200)
        
        llm_rates = self.llm_costs.get(llm_model, self.llm_costs['gpt-3.5-turbo'])
        monthly_cost['llm_input'] = (total_input_tokens / 1000) * llm_rates['input']
        monthly_cost['llm_output'] = (total_output_tokens / 1000) * llm_rates['output']
        
        # Infrastructure
        monthly_cost['infrastructure'] = config.get('infrastructure_cost', 100)
        
        return {
            'breakdown': monthly_cost,
            'total': sum(monthly_cost.values())
        }
    
    def compare_costs(self, configs: dict):
        """Compare costs of different approaches"""
        comparison = {}
        
        for name, config in configs.items():
            comparison[name] = self.calculate_monthly_cost(config)
        
        return comparison
```

## 2. Caching Strategies

### Query Result Caching

```python
class CachingCostOptimizer:
    """Calculate savings from caching"""
    
    def __init__(self, embedding_cost_per_1m: float = 0.13):
        self.embedding_cost = embedding_cost_per_1m
        self.cache_stats = {}
    
    def calculate_cache_savings(self, cache_hit_rate: float, monthly_queries: int):
        """Calculate cost savings from caching"""
        queries_saved = monthly_queries * cache_hit_rate
        
        # Cost per query embedding (1536 tokens / 1M * cost)
        cost_per_query_embedding = (50 / 1_000_000) * self.embedding_cost
        
        savings = queries_saved * cost_per_query_embedding
        
        return {
            'queries_saved': queries_saved,
            'monthly_savings': savings,
            'yearly_savings': savings * 12
        }
    
    def optimize_cache_strategy(self, monthly_queries: int, avg_cache_hit: float):
        """Optimize cache configuration"""
        # Benchmark different cache sizes
        cache_configs = [
            {'size_gb': 10, 'hit_rate': 0.4, 'cost': 5},
            {'size_gb': 50, 'hit_rate': 0.65, 'cost': 20},
            {'size_gb': 200, 'hit_rate': 0.8, 'cost': 50}
        ]
        
        costs = {}
        for config in cache_configs:
            savings = self.calculate_cache_savings(
                config['hit_rate'],
                monthly_queries
            )
            
            net_cost = config['cost'] - (savings['monthly_savings'] / 12)
            
            costs[config['size_gb']] = {
                'cache_cost': config['cost'],
                'monthly_savings': savings['monthly_savings'],
                'net_cost': net_cost
            }
        
        return costs
```

## 3. Model Selection for Cost

### Balancing Quality vs Cost

```python
class ModelCostComparison:
    """Compare models by cost-quality tradeoff"""
    
    def __init__(self):
        self.models = {
            'text-embedding-3-large': {
                'cost': 0.13,  # $/1M tokens
                'quality': 95,  # BEIR nDCG
                'speed': 200,  # ms
                'dims': 3072
            },
            'text-embedding-3-small': {
                'cost': 0.02,
                'quality': 85,
                'speed': 100,
                'dims': 1536
            },
            'e5-large': {
                'cost': 0.0,  # Open-source
                'quality': 90,
                'speed': 100,
                'dims': 1024
            },
            'bge-large': {
                'cost': 0.0,
                'quality': 89,
                'speed': 80,
                'dims': 1024
            },
            'e5-base': {
                'cost': 0.0,
                'quality': 85,
                'speed': 50,
                'dims': 768
            }
        }
    
    def calculate_cost_per_quality_point(self, monthly_queries: int):
        """Calculate cost per unit of quality"""
        comparison = {}
        
        for model_name, metrics in self.models.items():
            monthly_cost = (monthly_queries / 1_000_000) * metrics['cost']
            
            cost_per_quality = monthly_cost / metrics['quality']
            
            comparison[model_name] = {
                'monthly_cost': monthly_cost,
                'quality': metrics['quality'],
                'cost_per_quality_point': cost_per_quality,
                'speed': metrics['speed']
            }
        
        return comparison
    
    def recommend_model(self, budget_constraint: float, quality_requirement: float):
        """Recommend model based on constraints"""
        suitable = []
        
        for model_name, metrics in self.models.items():
            if (metrics['quality'] >= quality_requirement and 
                metrics['cost'] <= budget_constraint):
                suitable.append((model_name, metrics))
        
        if suitable:
            # Return cheapest suitable model
            return min(suitable, key=lambda x: x[1]['cost'])
        
        # Return best quality within budget
        return min(self.models.items(), 
                  key=lambda x: x[1]['cost'])
```

## 4. Inference Optimization

### Batch Processing

```python
class BatchProcessingOptimizer:
    """Optimize costs through batching"""
    
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
    
    def calculate_batch_savings(self, query_rate_per_sec: float, batch_size: int):
        """Calculate latency vs throughput tradeoff"""
        # Single query overhead ~50ms
        single_query_overhead = 0.05
        
        # Batched queries have amortized overhead
        batch_overhead_per_query = single_query_overhead / batch_size
        
        # Actual processing time
        processing_time_per_query = 0.02
        
        single_latency = single_query_overhead + processing_time_per_query
        batch_latency = batch_overhead_per_query + processing_time_per_query
        
        latency_increase = (batch_latency - single_latency) * 1000  # ms
        throughput_gain = batch_size
        
        return {
            'latency_increase_ms': latency_increase,
            'throughput_gain_factor': throughput_gain,
            'cost_reduction': (throughput_gain - 1) / throughput_gain  # % reduction
        }
```

## 5. Context Window Optimization

### Smart Context Truncation

```python
class ContextOptimizer:
    """Optimize context to reduce token usage"""
    
    def __init__(self, llm_token_cost: float = 0.0005):
        self.token_cost = llm_token_cost
    
    def calculate_optimal_context_size(self, queries_per_month: int, output_tokens: int = 200):
        """Find optimal context size"""
        costs = {}
        
        for context_tokens in [256, 512, 1024, 2048, 4096]:
            total_input_tokens = queries_per_month * context_tokens
            total_output_tokens = queries_per_month * output_tokens
            
            monthly_cost = (
                (total_input_tokens + total_output_tokens) / 1000 * 
                self.token_cost
            )
            
            costs[context_tokens] = monthly_cost
        
        return costs
    
    def select_most_relevant_context(self, documents, max_tokens: int = 1024):
        """Select context within token budget"""
        # Score documents by relevance
        ranked = sorted(
            documents,
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        selected = []
        token_count = 0
        
        for doc in ranked:
            doc_tokens = len(doc['text'].split())
            
            if token_count + doc_tokens <= max_tokens:
                selected.append(doc)
                token_count += doc_tokens
            else:
                # Truncate last document to fit
                remaining = max_tokens - token_count
                if remaining > 50:  # Only if meaningful
                    truncated_text = ' '.join(
                        doc['text'].split()[:remaining]
                    )
                    selected.append({
                        **doc,
                        'text': truncated_text,
                        'truncated': True
                    })
                break
        
        return selected
```

## 6. Infrastructure Cost Optimization

### Choosing Hosting Architecture

```python
class ArchitectureComparison:
    """Compare infrastructure cost by architecture"""
    
    def __init__(self):
        self.architectures = {
            'cloud_managed': {
                'base_cost': 100,  # Pinecone pod
                'per_query_cost': 0.00004,
                'per_embedding_cost': 0.13 / 1_000_000,
                'scaling': 'automatic',
                'maintenance': 'minimal'
            },
            'hybrid': {
                'base_cost': 50,  # Weaviate cloud
                'per_query_cost': 0,
                'per_embedding_cost': 0.13 / 1_000_000,
                'scaling': 'manual',
                'maintenance': 'moderate'
            },
            'self_hosted_cpu': {
                'base_cost': 30,  # Compute + storage
                'per_query_cost': 0,
                'per_embedding_cost': 0,  # Local
                'scaling': 'manual',
                'maintenance': 'high'
            },
            'self_hosted_gpu': {
                'base_cost': 200,  # GPU VM
                'per_query_cost': 0,
                'per_embedding_cost': 0,
                'scaling': 'manual',
                'maintenance': 'high'
            }
        }
    
    def calculate_breakeven(self, monthly_queries: int, monthly_embeddings: int):
        """Calculate cost breakeven points"""
        comparison = {}
        
        for arch_name, arch_costs in self.architectures.items():
            monthly_cost = (
                arch_costs['base_cost'] +
                monthly_queries * arch_costs['per_query_cost'] +
                monthly_embeddings * arch_costs['per_embedding_cost']
            )
            
            comparison[arch_name] = {
                'monthly_cost': monthly_cost,
                'yearly_cost': monthly_cost * 12,
                'cost_per_query': monthly_cost / monthly_queries if monthly_queries > 0 else 0
            }
        
        return comparison
```

## 7. Observability for Cost Control

### Cost Tracking and Alerting

```python
class CostMonitor:
    """Monitor costs in real-time"""
    
    def __init__(self, budget_limit: float):
        self.budget_limit = budget_limit
        self.current_spending = {}
        self.alerts = []
    
    def track_api_usage(self, api_name: str, tokens_used: int, cost_per_1m: float):
        """Track API usage"""
        if api_name not in self.current_spending:
            self.current_spending[api_name] = {'tokens': 0, 'cost': 0}
        
        cost = (tokens_used / 1_000_000) * cost_per_1m
        
        self.current_spending[api_name]['tokens'] += tokens_used
        self.current_spending[api_name]['cost'] += cost
        
        # Check if over budget
        total_cost = sum(c['cost'] for c in self.current_spending.values())
        
        if total_cost > self.budget_limit:
            self.alerts.append({
                'level': 'warning',
                'message': f'Cost {total_cost:.2f} exceeds budget {self.budget_limit:.2f}',
                'timestamp': datetime.now()
            })
    
    def get_cost_summary(self):
        """Get current cost summary"""
        summary = {}
        
        for api, usage in self.current_spending.items():
            summary[api] = {
                'tokens': usage['tokens'],
                'cost': usage['cost'],
                'projected_monthly': usage['cost'] * (30 / datetime.now().day)
            }
        
        total = sum(s['cost'] for s in summary.values())
        summary['total'] = total
        
        return summary
```

## 8. Cost Optimization Checklist

- [ ] Implement aggressive query result caching (40-50% cost reduction)
- [ ] Use open-source embeddings where quality permits (70-90% cost reduction)
- [ ] Optimize context window size (20-30% cost reduction)
- [ ] Batch process queries (10-20% cost reduction)
- [ ] Use distilled embedding models (50-70% cost reduction with <5% quality loss)
- [ ] Cache embeddings (40-80% API cost reduction)
- [ ] Monitor costs in real-time
- [ ] Setup alerts for budget overruns
- [ ] Compress embeddings (70-80% storage cost reduction)
- [ ] Use self-hosted infrastructure for high volume (50-80% infrastructure cost reduction)

## Conclusion

RAG cost optimization requires balancing quality, latency, and cost across all components. The biggest savings come from caching (40-50% cost reduction), model selection (50-80% cost reduction for open-source), and context optimization (20-30% cost reduction). Most applications can reduce costs 60-70% by combining these techniques.
