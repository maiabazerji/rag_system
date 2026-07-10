# Hallucination Reduction in RAG Systems

## Introduction

Hallucination (generating false information) is a major challenge in RAG systems. This guide covers techniques to detect and reduce hallucinations.

## 1. Hallucination Fundamentals

### Types of Hallucinations

```python
from enum import Enum

class HallucinationType(Enum):
    INTRINSIC = "model generates false info inconsistent with training"
    EXTRINSIC = "model generates true-sounding info not in context"
    CONTRADICTION = "response contradicts provided context"
    FABRICATION = "invented facts not grounded anywhere"

class HallucinationAnalyzer:
    def categorize_hallucination(self, response: str, context: str):
        """Categorize type of hallucination"""
        
        if self._contradicts_context(response, context):
            return HallucinationType.CONTRADICTION
        
        elif self._unsupported_by_context(response, context):
            return HallucinationType.EXTRINSIC
        
        else:
            return HallucinationType.INTRINSIC
    
    def _contradicts_context(self, response: str, context: str):
        """Check if response contradicts context"""
        # Use NLI model
        from transformers import pipeline
        
        nli_pipeline = pipeline("zero-shot-classification")
        
        # Extract key claim from response
        claims = self._extract_claims(response)
        
        for claim in claims:
            # Check if context contradicts
            result = nli_pipeline(context, claim, 
                                hypothesis_template="This passage is about {}",
                                multi_class=False)
            
            if result['labels'][0] == 'contradiction':
                return True
        
        return False
    
    def _unsupported_by_context(self, response: str, context: str):
        """Check if response lacks grounding in context"""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(response_words & context_words) / len(response_words)
        
        # If less than 30% word overlap, likely unsupported
        return overlap < 0.3
    
    def _extract_claims(self, text: str):
        """Extract factual claims from text"""
        sentences = text.split('.')
        return [s.strip() for s in sentences if len(s.split()) > 5]
```

## 2. Prevention Techniques

### Grounding Enforcement

```python
class GroundingEnforcer:
    """Force model to ground responses in context"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_grounded(self, query: str, context: str):
        """Generate with explicit grounding instructions"""
        
        prompt = f"""Based ONLY on the provided context, answer the question.
        
IMPORTANT RULES:
1. Only use information from the context
2. If the answer is not in the context, say "I don't know"
3. Every claim must be traceable to the context
4. Do NOT use general knowledge

CONTEXT:
{context}

QUESTION: {query}

ANSWER: """
        
        response = self.llm.generate(prompt)
        
        return response
    
    def generate_with_citations(self, query: str, context: str, doc_ids: list):
        """Generate with required citations"""
        
        prompt = f"""Answer the question using the provided context.
        After each fact, cite the source as [Doc 1], [Doc 2], etc.

DOCUMENTS:
{self._format_docs(context, doc_ids)}

QUESTION: {query}

Answer with citations:"""
        
        response = self.llm.generate(prompt)
        
        # Verify citations are present
        verified = self._verify_citations(response, doc_ids)
        
        return response, verified
    
    def _format_docs(self, context: str, doc_ids: list):
        """Format documents with IDs"""
        formatted = []
        for i, doc_id in enumerate(doc_ids, 1):
            formatted.append(f"[Doc {i}] {context}")  # Simplified
        
        return "\n".join(formatted)
    
    def _verify_citations(self, response: str, doc_ids: list):
        """Check if all citations are valid"""
        import re
        
        citations = re.findall(r'\[Doc (\d+)\]', response)
        
        valid_ids = set(range(1, len(doc_ids) + 1))
        
        for cite in citations:
            if int(cite) not in valid_ids:
                return False
        
        return True
```

### Confidence Scoring

```python
class ConfidenceScorer:
    """Score confidence in generated responses"""
    
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
    
    def score_confidence(self, query: str, response: str, context: str):
        """Score confidence in response"""
        
        # Multiple signals
        grounding_score = self._grounding_score(response, context)
        consistency_score = self._consistency_score(response)
        nli_score = self._nli_score(response, context)
        
        # Weighted combination
        confidence = (
            0.4 * grounding_score +
            0.3 * consistency_score +
            0.3 * nli_score
        )
        
        return {
            'overall_confidence': confidence,
            'grounding': grounding_score,
            'consistency': consistency_score,
            'nli': nli_score,
            'confidence_level': 'high' if confidence > 0.75 else
                               'medium' if confidence > 0.5 else
                               'low'
        }
    
    def _grounding_score(self, response: str, context: str):
        """How well is response grounded in context"""
        # Similarity between response and context
        response_emb = self.embedding_model.encode(response)
        context_emb = self.embedding_model.encode(context[:500])
        
        similarity = np.dot(response_emb, context_emb)
        
        # Normalize to 0-1
        return (similarity + 1) / 2
    
    def _consistency_score(self, response: str):
        """Internal consistency of response"""
        sentences = response.split('.')
        
        if len(sentences) < 2:
            return 1.0
        
        # Check if later sentences contradict earlier ones
        embeddings = self.embedding_model.encode(sentences)
        
        contradictions = 0
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j])
                
                # Very low similarity might indicate contradiction
                if similarity < -0.5:
                    contradictions += 1
        
        return max(0, 1 - (contradictions / len(sentences)))
    
    def _nli_score(self, response: str, context: str):
        """Use NLI to assess entailment"""
        from transformers import pipeline
        
        nli = pipeline("zero-shot-classification")
        
        # Check if response is entailed by context
        result = nli(context, response, 
                     hypothesis_template="This passage implies: {}",
                     multi_class=False)
        
        # Convert to confidence score
        if result['labels'][0] == 'entailment':
            return result['scores'][0]
        else:
            return 1 - result['scores'][0]
```

## 3. Post-Generation Filtering

### Fact Verification

```python
class FactVerifier:
    """Verify factual claims in responses"""
    
    def __init__(self, knowledge_base=None, search_api=None):
        self.knowledge_base = knowledge_base
        self.search_api = search_api
    
    def verify_claims(self, response: str, context: str):
        """Verify factual claims in response"""
        
        claims = self._extract_factual_claims(response)
        
        verifications = []
        
        for claim in claims:
            verified = self._verify_single_claim(claim, context)
            
            verifications.append({
                'claim': claim,
                'verified': verified['status'],
                'evidence': verified.get('evidence', ''),
                'confidence': verified.get('confidence', 0)
            })
        
        # Check overall hallucination rate
        verified_count = sum(1 for v in verifications if v['verified'])
        hallucination_rate = 1 - (verified_count / len(verifications)) if verifications else 0
        
        return {
            'verifications': verifications,
            'hallucination_rate': hallucination_rate,
            'recommendation': 'use_carefully' if hallucination_rate > 0.2 else 'use'
        }
    
    def _extract_factual_claims(self, response: str):
        """Extract verifiable claims"""
        import re
        
        # Simple heuristic: claims containing numbers, dates, proper nouns
        sentences = response.split('.')
        
        claims = []
        for sent in sentences:
            if any(c.isupper() for c in sent) or \
               re.search(r'\d{4}', sent) or \
               re.search(r'\b\d+%', sent):
                claims.append(sent.strip())
        
        return claims
    
    def _verify_single_claim(self, claim: str, context: str):
        """Verify single claim"""
        
        # Check against context
        if any(word in context for word in claim.split()[:5]):
            return {'status': True, 'evidence': context[:100], 'confidence': 0.8}
        
        # Try knowledge base
        if self.knowledge_base:
            result = self.knowledge_base.search(claim)
            if result and result['score'] > 0.7:
                return {'status': True, 'evidence': result['text'], 'confidence': result['score']}
        
        # Try search API
        if self.search_api:
            result = self.search_api.search(claim)
            if result:
                return {'status': True, 'evidence': result['snippet'], 'confidence': 0.7}
        
        # Could not verify
        return {'status': False, 'evidence': '', 'confidence': 0.5}
```

## 4. Retrieval-Based Mitigation

### Increasing Context Quality

```python
class HallucinationMitigatingRetriever:
    """Retrieval tuned to reduce hallucination"""
    
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
    
    def retrieve_for_grounding(self, query: str, top_k: int = 10):
        """Retrieve specifically for preventing hallucination"""
        
        # Get more documents than needed
        retrieved = self.base_retriever.retrieve(query, top_k=top_k*2)
        
        # Filter for document diversity
        diverse_docs = self._select_diverse(retrieved, top_k)
        
        # Prioritize documents with explicit information over generic
        prioritized = self._prioritize_factual(diverse_docs)
        
        return prioritized
    
    def _select_diverse(self, docs: list, top_k: int):
        """Select diverse documents to reduce overlap"""
        from sklearn.cluster import KMeans
        
        embeddings = []
        for doc in docs:
            # Embed document
            emb = self.embedding_model.encode(doc['text'][:200])
            embeddings.append(emb)
        
        # Cluster to find diverse docs
        n_clusters = min(top_k, len(docs))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Select one from each cluster
        selected = []
        for cluster_id in range(n_clusters):
            cluster_docs = [doc for doc, label in zip(docs, labels) if label == cluster_id]
            if cluster_docs:
                # Pick best from cluster
                selected.append(max(cluster_docs, key=lambda x: x.get('score', 0)))
        
        return selected
    
    def _prioritize_factual(self, docs: list):
        """Prioritize factual/specific information"""
        
        scored_docs = []
        
        for doc in docs:
            # Score by specificity (contains numbers, dates, named entities)
            specificity_score = self._compute_specificity(doc['text'])
            
            doc['specificity'] = specificity_score
            scored_docs.append(doc)
        
        # Sort by specificity * relevance
        scored_docs.sort(
            key=lambda d: d['specificity'] * d.get('score', 1),
            reverse=True
        )
        
        return scored_docs
    
    def _compute_specificity(self, text: str):
        """Compute how specific/factual text is"""
        import re
        
        specificity = 0
        
        # Numbers
        numbers = len(re.findall(r'\d+', text))
        specificity += min(numbers, 5) / 5
        
        # Dates
        dates = len(re.findall(r'\d{4}', text))
        specificity += min(dates, 3) / 3
        
        # Proper nouns (capitalized words)
        capitalized = len([w for w in text.split() if w[0].isupper()])
        specificity += min(capitalized, 10) / 10
        
        return min(specificity / 3, 1.0)
```

## 5. Hallucination Monitoring

### Production Monitoring

```python
class HallucinationMonitor:
    """Monitor hallucination in production"""
    
    def __init__(self, verification_threshold: float = 0.5):
        self.threshold = verification_threshold
        self.hallucination_log = []
    
    def monitor_response(self, query: str, response: str, context: str):
        """Monitor single response for hallucinations"""
        
        # Analyze hallucinations
        analysis = self._analyze_hallucinations(response, context)
        
        # Log if high hallucination
        if analysis['hallucination_score'] > self.threshold:
            self.hallucination_log.append({
                'query': query,
                'response': response,
                'hallucination_score': analysis['hallucination_score'],
                'timestamp': datetime.now()
            })
        
        return analysis
    
    def _analyze_hallucinations(self, response: str, context: str):
        """Analyze hallucination level"""
        
        # Use multiple signals
        grounding = self._check_grounding(response, context)
        consistency = self._check_consistency(response)
        contradiction = self._check_contradiction(response, context)
        
        # Combine signals
        hallucination_score = (
            (1 - grounding) * 0.5 +
            (1 - consistency) * 0.3 +
            contradiction * 0.2
        )
        
        return {
            'hallucination_score': hallucination_score,
            'grounding': grounding,
            'consistency': consistency,
            'contradiction': contradiction
        }
    
    def get_hallucination_stats(self):
        """Get hallucination statistics"""
        if not self.hallucination_log:
            return {}
        
        scores = [h['hallucination_score'] for h in self.hallucination_log]
        
        return {
            'mean_hallucination_score': np.mean(scores),
            'hallucination_rate': len(self.hallucination_log),
            'max_hallucination': np.max(scores)
        }
```

## Best Practices

1. **Ground in context** - enforce that model uses provided information
2. **Require citations** - make claims traceable
3. **Verify facts** - check against knowledge base
4. **Monitor confidence** - flag low-confidence responses
5. **Diverse retrieval** - reduce information redundancy
6. **Use graded prompts** - guide model to admission of uncertainty

## Conclusion

Hallucination reduction requires multi-layered approach: grounding enforcement, confidence scoring, post-generation filtering, and retrieval tuning. No single technique eliminates hallucination; combining multiple approaches reduces it by 50-70%. Always measure and monitor hallucination in production.
