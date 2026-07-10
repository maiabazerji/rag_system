# Citation and Attribution in RAG Systems

## Introduction

Proper citation and attribution build trust in RAG systems. This guide covers techniques for tracking sources and generating citations.

## 1. Citation Tracking

### Document Metadata Management

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class CitationMetadata:
    doc_id: str
    source: str
    url: str = None
    page_number: int = None
    timestamp: str = None
    author: str = None
    confidence_score: float = 1.0

class CitationTracker:
    """Track citations throughout RAG pipeline"""
    
    def __init__(self):
        self.document_registry = {}
        self.citation_chain = []
    
    def register_document(self, doc_id: str, metadata: CitationMetadata):
        """Register document with metadata"""
        self.document_registry[doc_id] = metadata
    
    def track_retrieval(self, query: str, retrieved_docs: List[str]):
        """Track which documents were retrieved"""
        self.citation_chain.append({
            'stage': 'retrieval',
            'query': query,
            'documents': retrieved_docs,
            'timestamp': datetime.now()
        })
    
    def track_reranking(self, reranked_docs: List[tuple]):
        """Track reranking results"""
        doc_ids, scores = zip(*reranked_docs)
        
        self.citation_chain.append({
            'stage': 'reranking',
            'documents': doc_ids,
            'scores': scores,
            'timestamp': datetime.now()
        })
    
    def get_citation_chain(self, doc_id: str):
        """Get full citation chain for a document"""
        chain = []
        
        for event in self.citation_chain:
            if doc_id in event.get('documents', []):
                chain.append(event)
        
        return chain
```

## 2. Citation Generation

### Inline Citations

```python
class InlineCitationGenerator:
    """Generate citations inline with text"""
    
    def __init__(self, document_registry: Dict):
        self.registry = document_registry
    
    def generate_citations(self, text: str, source_segments: Dict[str, str]):
        """
        Add inline citations to text
        source_segments: {doc_id: relevant_text_segment}
        """
        cited_text = text
        citation_map = {}
        
        # Process each source
        for doc_id, segment in source_segments.items():
            metadata = self.registry.get(doc_id, {})
            
            # Find where segment appears in text
            if segment in text:
                # Add citation marker
                citation_marker = self._create_citation_marker(doc_id, metadata)
                
                cited_text = cited_text.replace(
                    segment,
                    f"{segment}{citation_marker}"
                )
                
                citation_map[doc_id] = metadata
        
        return {
            'cited_text': cited_text,
            'citations': citation_map
        }
    
    def _create_citation_marker(self, doc_id: str, metadata: dict):
        """Create citation marker"""
        if metadata.get('url'):
            return f" [link to {metadata['source']}]"
        else:
            return f" [Source: {metadata.get('source', doc_id)}]"
    
    def generate_footnotes(self, citations: Dict):
        """Generate footnotes/bibliography"""
        footnotes = []
        
        for i, (doc_id, metadata) in enumerate(citations.items(), 1):
            footnote = self._format_footnote(i, metadata)
            footnotes.append(footnote)
        
        return "\n".join(footnotes)
    
    def _format_footnote(self, number: int, metadata: dict):
        """Format single footnote"""
        parts = [f"{number}."]
        
        if metadata.get('author'):
            parts.append(f"{metadata['author']}.")
        
        parts.append(f"Source: {metadata.get('source', 'Unknown')}")
        
        if metadata.get('url'):
            parts.append(f"URL: {metadata['url']}")
        
        if metadata.get('page_number'):
            parts.append(f"Page {metadata['page_number']}")
        
        return " ".join(parts)
```

### Structured Citations

```python
class StructuredCitationGenerator:
    """Generate structured citations in various formats"""
    
    def __init__(self, document_registry: Dict):
        self.registry = document_registry
    
    def generate_json_citations(self, response: str, sources: List[str]):
        """Generate citations in JSON format"""
        return {
            'response': response,
            'citations': [
                self._format_citation_json(doc_id)
                for doc_id in sources
            ]
        }
    
    def generate_bibtex_citations(self, sources: List[str]):
        """Generate BibTeX formatted citations"""
        bibtex = []
        
        for i, doc_id in enumerate(sources, 1):
            metadata = self.registry.get(doc_id, {})
            
            entry = f"""@misc{{ref{i},
  title={{{metadata.get('source', 'Unknown')}}},
  author={{{metadata.get('author', 'Unknown')}}},
  year={{{metadata.get('timestamp', 'Unknown')}}},
  url={{{metadata.get('url', '')}}}
}}"""
            
            bibtex.append(entry)
        
        return "\n".join(bibtex)
    
    def generate_apa_citations(self, sources: List[str]):
        """Generate APA formatted citations"""
        citations = []
        
        for doc_id in sources:
            metadata = self.registry.get(doc_id, {})
            
            citation = f"{metadata.get('author', 'Unknown')} ({metadata.get('timestamp', 'n.d.')}). {metadata.get('source', 'Unknown')}."
            
            if metadata.get('url'):
                citation += f" Retrieved from {metadata['url']}"
            
            citations.append(citation)
        
        return "\n".join(citations)
    
    def _format_citation_json(self, doc_id: str):
        """Format citation as JSON"""
        metadata = self.registry.get(doc_id, {})
        
        return {
            'id': doc_id,
            'source': metadata.get('source'),
            'url': metadata.get('url'),
            'author': metadata.get('author'),
            'page': metadata.get('page_number'),
            'confidence': metadata.get('confidence_score', 1.0)
        }
```

## 3. Attribution Display

### Web UI Citation Rendering

```python
class CitationRenderer:
    """Render citations in web UI"""
    
    @staticmethod
    def render_html_citations(response: str, citations: Dict):
        """Render citations as HTML"""
        html = "<div class='rag-response'>"
        
        # Response with citation links
        html += f"<p class='response-text'>{response}</p>"
        
        # Citation sidebar
        html += "<aside class='citations'>"
        html += "<h3>Sources</h3>"
        html += "<ul>"
        
        for i, (doc_id, metadata) in enumerate(citations.items(), 1):
            html += f"""<li>
                <a href="{metadata.get('url', '#')}" target="_blank">
                    [{i}] {metadata.get('source')}
                </a>
                <small>{metadata.get('author', 'Unknown')}</small>
            </li>"""
        
        html += "</ul>"
        html += "</aside>"
        html += "</div>"
        
        return html
    
    @staticmethod
    def generate_citation_card(metadata: dict):
        """Generate HTML card for single citation"""
        return f"""
        <div class="citation-card">
            <h4>{metadata.get('source')}</h4>
            <p class="author">{metadata.get('author', 'Unknown')}</p>
            <p class="url">
                <a href="{metadata.get('url', '#')}" target="_blank">
                    View Source
                </a>
            </p>
            <p class="confidence">
                Confidence: {metadata.get('confidence_score', 1.0):.0%}
            </p>
        </div>
        """
```

## 4. Citation Verification

### Source Verification

```python
class SourceVerifier:
    """Verify citation sources"""
    
    def __init__(self):
        self.verified_cache = {}
    
    def verify_url(self, url: str):
        """Verify URL is accessible"""
        import requests
        
        if url in self.verified_cache:
            return self.verified_cache[url]
        
        try:
            response = requests.head(url, timeout=5)
            verified = response.status_code < 400
        except:
            verified = False
        
        self.verified_cache[url] = verified
        return verified
    
    def verify_attribution(self, citation: dict):
        """Verify citation attribution"""
        verification = {
            'url_valid': False,
            'content_accessible': False,
            'confidence': 0
        }
        
        # Check URL
        if citation.get('url'):
            verification['url_valid'] = self.verify_url(citation['url'])
        
        # Check if source appears in registry
        verification['confidence'] = citation.get('confidence_score', 0)
        
        return verification
```

## 5. Citation Quality Metrics

### Evaluating Citation Quality

```python
class CitationQualityEvaluator:
    """Evaluate quality of citations"""
    
    def evaluate_citation_coverage(self, response: str, citations: list):
        """Check if all claims are cited"""
        # Extract claims from response
        sentences = response.split('.')
        
        cited_claims = 0
        total_claims = 0
        
        for sent in sentences:
            if len(sent.split()) > 5:  # Meaningful sentence
                total_claims += 1
                
                # Check if sentence is supported by citations
                if any(self._is_claim_supported(sent, cit) for cit in citations):
                    cited_claims += 1
        
        return {
            'coverage': cited_claims / total_claims if total_claims > 0 else 0,
            'cited_claims': cited_claims,
            'total_claims': total_claims
        }
    
    def evaluate_citation_relevance(self, claim: str, citation: dict):
        """Evaluate if citation supports claim"""
        from sentence_transformers import util, SentenceTransformer
        
        model = SentenceTransformer('intfloat/e5-large')
        
        claim_emb = model.encode(claim)
        source_emb = model.encode(citation.get('source', ''))
        
        similarity = util.pytorch_cos_sim(claim_emb, source_emb)
        
        return float(similarity[0])
    
    def _is_claim_supported(self, claim: str, citation: dict):
        """Simple check if claim is supported"""
        relevance = self.evaluate_citation_relevance(claim, citation)
        return relevance > 0.5
```

## 6. Citation Best Practices

```python
class CitationBestPractices:
    """Guidelines for citation"""
    
    @staticmethod
    def check_citation_requirements(response: str, citations: dict):
        """Check if citation meets best practices"""
        checks = {
            'has_citations': len(citations) > 0,
            'all_sources_included': True,
            'proper_format': True,
            'accessible_sources': True
        }
        
        # Check if all URLs are accessible
        for metadata in citations.values():
            if metadata.get('url'):
                try:
                    import requests
                    r = requests.head(metadata['url'], timeout=5)
                    if r.status_code >= 400:
                        checks['accessible_sources'] = False
                except:
                    checks['accessible_sources'] = False
        
        return checks
    
    @staticmethod
    def recommendations_for_citations(checks: dict):
        """Get recommendations for improving citations"""
        recommendations = []
        
        if not checks['has_citations']:
            recommendations.append("Add citations to support claims")
        
        if not checks['accessible_sources']:
            recommendations.append("Ensure all cited sources are still accessible")
        
        if not checks['all_sources_included']:
            recommendations.append("Include all sources used in generation")
        
        return recommendations
```

## Best Practices

1. **Track all sources** - maintain complete audit trail
2. **Use structured formats** - JSON, BibTeX, APA
3. **Verify URLs** - check links are still valid
4. **Cite inline** - place citations where they support claims
5. **Include metadata** - author, date, confidence
6. **Test accessibility** - verify sources are accessible
7. **Multiple formats** - support different citation styles

## Conclusion

Proper citation and attribution are essential for trust in RAG systems. Implement tracking from retrieval through generation, support multiple citation formats, and verify sources. Transparent attribution differentiates RAG systems from unreliable AI.
