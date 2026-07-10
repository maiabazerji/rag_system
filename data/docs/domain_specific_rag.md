# Domain-Specific RAG Systems: Specialized Applications

## Introduction

Generic RAG systems work reasonably well across domains, but domain-specific adaptations significantly improve accuracy and utility. This guide covers techniques for building RAG systems specialized for particular fields.

## 1. Technical Documentation RAG

### Code-Aware Indexing

```python
import ast
from typing import List, Dict

class CodeDocumentationIndexer:
    """Index code and documentation together"""
    
    def __init__(self, embedding_model):
        self.model = embedding_model
    
    def parse_code_file(self, file_path):
        """Extract functions, classes, and docstrings"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []  # Not valid Python
        
        elements = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                element = {
                    'type': 'function',
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'signature': self._get_function_signature(node),
                    'line': node.lineno,
                    'file': file_path
                }
                elements.append(element)
            
            elif isinstance(node, ast.ClassDef):
                element = {
                    'type': 'class',
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'methods': [
                        m.name for m in node.body
                        if isinstance(m, ast.FunctionDef)
                    ],
                    'line': node.lineno,
                    'file': file_path
                }
                elements.append(element)
        
        return elements
    
    def _get_function_signature(self, node):
        """Extract function signature"""
        args = node.args
        arg_names = [arg.arg for arg in args.args]
        return f"{node.name}({', '.join(arg_names)})"
    
    def index_code_documentation(self, code_files):
        """Index code and documentation"""
        index = []
        
        for file_path in code_files:
            elements = self.parse_code_file(file_path)
            
            for elem in elements:
                # Create searchable text
                search_text = self._element_to_text(elem)
                
                # Embed
                embedding = self.model.encode(search_text)
                
                index.append({
                    'element': elem,
                    'text': search_text,
                    'embedding': embedding,
                    'file': file_path
                })
        
        return index
    
    def _element_to_text(self, elem):
        """Convert code element to searchable text"""
        parts = []
        
        parts.append(f"{elem['type'].upper()}: {elem['name']}")
        
        if 'signature' in elem:
            parts.append(f"Signature: {elem['signature']}")
        
        if elem.get('docstring'):
            parts.append(f"Documentation: {elem['docstring']}")
        
        if 'methods' in elem and elem['methods']:
            parts.append(f"Methods: {', '.join(elem['methods'])}")
        
        return "\n".join(parts)
```

### Technical Query Understanding

```python
class TechnicalQueryProcessor:
    def __init__(self, llm):
        self.llm = llm
    
    def extract_technical_intent(self, query):
        """Understand technical intent"""
        prompt = f"""Analyze this technical query:
        {query}
        
        Identify:
        1. Primary intent (usage, error, documentation, example)
        2. Technology/library mentioned
        3. Specific error or problem (if any)
        4. Related concepts
        
        Format as JSON."""
        
        return self.llm.generate(prompt, output_format='json')
    
    def resolve_abbreviations(self, query):
        """Resolve technical abbreviations"""
        tech_abbreviations = {
            'API': 'Application Programming Interface',
            'REST': 'Representational State Transfer',
            'HTTP': 'Hypertext Transfer Protocol',
            'JSON': 'JavaScript Object Notation',
            'CRUD': 'Create Read Update Delete',
            'ORM': 'Object-Relational Mapping',
            'MVC': 'Model View Controller',
        }
        
        expanded = query
        for abbr, full in tech_abbreviations.items():
            if abbr in query:
                expanded = expanded.replace(abbr, f"{abbr} ({full})")
        
        return expanded
```

## 2. Medical/Healthcare RAG

### Medical Document Processing

```python
class MedicalDocumentProcessor:
    def __init__(self, embedding_model, medical_ner_model):
        self.model = embedding_model
        self.ner = medical_ner_model
    
    def extract_medical_entities(self, text):
        """Extract medical entities (diagnoses, drugs, etc.)"""
        entities = {
            'diagnoses': [],
            'medications': [],
            'procedures': [],
            'lab_values': [],
            'vital_signs': []
        }
        
        # Use medical NER model
        ner_results = self.ner.extract(text)
        
        for entity, label in ner_results:
            if label == 'DIAGNOSIS':
                entities['diagnoses'].append(entity)
            elif label == 'MEDICATION':
                entities['medications'].append(entity)
            elif label == 'PROCEDURE':
                entities['procedures'].append(entity)
            elif label == 'LAB_VALUE':
                entities['lab_values'].append(entity)
            elif label == 'VITAL_SIGNS':
                entities['vital_signs'].append(entity)
        
        return entities
    
    def chunk_medical_document(self, document):
        """Chunk medical documents respecting structure"""
        chunks = []
        
        sections = document.split('\n\n')
        current_chunk = []
        
        for section in sections:
            current_chunk.append(section)
            
            # Create chunk when medical entities change significantly
            if self._should_chunk(current_chunk):
                chunk_text = '\n\n'.join(current_chunk)
                
                entities = self.extract_medical_entities(chunk_text)
                
                chunks.append({
                    'text': chunk_text,
                    'entities': entities,
                    'embedding': self.model.encode(chunk_text)
                })
                
                current_chunk = []
        
        return chunks
    
    def _should_chunk(self, sections):
        """Determine if should create chunk"""
        return len('\n\n'.join(sections).split()) > 300
```

### Medical Query Safety

```python
class MedicalQuerySafetyChecker:
    """Ensure safe handling of medical queries"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def check_query_safety(self, query):
        """Check if query is safe for system to answer"""
        prompt = f"""Evaluate if it's safe for an AI system to answer this medical query:
        {query}
        
        Consider:
        - Is it requesting diagnosis of symptoms? (risky)
        - Is it requesting treatment advice? (risky)
        - Is it requesting general information? (safe)
        
        Provide: safe/unsafe and explanation."""
        
        assessment = self.llm.generate(prompt)
        return assessment
    
    def add_safety_disclaimer(self, response, query_type='medical'):
        """Add appropriate disclaimer to response"""
        disclaimers = {
            'medical': 'DISCLAIMER: This is not medical advice. Consult a healthcare professional.',
            'legal': 'DISCLAIMER: This is not legal advice. Consult a lawyer.',
            'financial': 'DISCLAIMER: This is not financial advice. Consult a financial advisor.'
        }
        
        disclaimer = disclaimers.get(query_type, '')
        
        return f"{response}\n\n{disclaimer}"
```

## 3. Legal Document RAG

### Legal Citation Handling

```python
import re

class LegalCitationExtractor:
    def __init__(self):
        # Patterns for common legal citations
        self.citation_patterns = {
            'statute': r'(\d+\s+U\.S\.C\.?\s+§?\s+\d+)',  # US Code
            'regulation': r'(\d+\s+CFR\s+§?\s+\d+)',  # Code of Federal Regulations
            'case': r'(\w+\s+v\.?\s+\w+,?\s+\d+\s+\w+\.?\s+\d+)',  # Case citations
        }
    
    def extract_citations(self, text):
        """Extract legal citations"""
        citations = {}
        
        for citation_type, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, text)
            citations[citation_type] = list(set(matches))
        
        return citations
    
    def resolve_citation(self, citation):
        """Resolve citation to full reference"""
        # In production, use legal database API
        citation_db = {
            '42 U.S.C. § 1983': 'Civil Rights Act - Liability of States',
            '28 U.S.C. § 1331': 'Federal Question Jurisdiction'
        }
        
        return citation_db.get(citation, f"Citation: {citation}")
```

### Legal Document Structure

```python
class LegalDocumentStructurer:
    def parse_legal_document(self, document):
        """Parse legal document structure"""
        structure = {
            'sections': [],
            'provisions': [],
            'definitions': []
        }
        
        lines = document.split('\n')
        current_section = None
        
        for line in lines:
            # Detect section headers
            if re.match(r'^(SECTION|§)\s+\d+', line):
                current_section = line
                structure['sections'].append({'title': line, 'content': []})
            
            # Detect definitions
            elif 'means' in line or 'shall be construed' in line:
                structure['definitions'].append(line)
            
            # Add content to current section
            elif current_section and structure['sections']:
                structure['sections'][-1]['content'].append(line)
        
        return structure
```

## 4. Scientific Literature RAG

### Citation Context Extraction

```python
class ScientificCitationContextor:
    def __init__(self, embedding_model):
        self.model = embedding_model
    
    def extract_citation_context(self, paper, citation, window_size=3):
        """Extract context around citations"""
        paragraphs = paper.split('\n\n')
        
        citation_contexts = []
        
        for i, para in enumerate(paragraphs):
            if citation in para:
                # Get surrounding paragraphs
                start = max(0, i - window_size)
                end = min(len(paragraphs), i + window_size + 1)
                
                context = '\n\n'.join(paragraphs[start:end])
                
                citation_contexts.append({
                    'context': context,
                    'paragraph_index': i,
                    'embedding': self.model.encode(context)
                })
        
        return citation_contexts
    
    def find_related_citations(self, paper, citation):
        """Find citations related to specified citation"""
        contexts = self.extract_citation_context(paper, citation)
        
        related_citations = []
        
        for context in contexts:
            # Extract other citations in context
            citations_in_context = self._extract_citations(context['context'])
            related_citations.extend(
                [c for c in citations_in_context if c != citation]
            )
        
        return list(set(related_citations))
```

### Research Topic Clustering

```python
from sklearn.cluster import KMeans

class ResearchTopicClusterer:
    def __init__(self, embedding_model):
        self.model = embedding_model
    
    def cluster_papers_by_topic(self, papers, num_topics=10):
        """Cluster papers by research topic"""
        # Extract abstracts
        abstracts = [p.get('abstract', p.get('content', '')) for p in papers]
        
        # Embed abstracts
        embeddings = self.model.encode(abstracts)
        
        # Cluster
        clusterer = KMeans(n_clusters=num_topics)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Organize by cluster
        topics = {}
        for i, label in enumerate(cluster_labels):
            if label not in topics:
                topics[label] = []
            topics[label].append(papers[i])
        
        return topics
```

## 5. Financial RAG

### Financial Concept Normalization

```python
class FinancialConceptNormalizer:
    def __init__(self):
        self.concept_mappings = {
            'P/E': 'Price-to-Earnings Ratio',
            'ROI': 'Return on Investment',
            'EBITDA': 'Earnings Before Interest, Taxes, Depreciation, Amortization',
            'APY': 'Annual Percentage Yield',
            'APR': 'Annual Percentage Rate',
        }
    
    def normalize_query(self, query):
        """Normalize financial terminology"""
        normalized = query
        
        for abbr, full in self.concept_mappings.items():
            if abbr in query:
                normalized = normalized.replace(abbr, f"{abbr} ({full})")
        
        return normalized
    
    def parse_financial_statement(self, statement):
        """Parse financial statements"""
        parsed = {
            'items': [],
            'totals': {},
            'ratios': []
        }
        
        # Extract line items
        for line in statement.split('\n'):
            if any(char.isdigit() for char in line):
                parts = line.split()
                if len(parts) >= 2:
                    # Assume last part is amount
                    parsed['items'].append({
                        'description': ' '.join(parts[:-1]),
                        'value': float(parts[-1])
                    })
        
        return parsed
```

## 6. Domain Adaptation Techniques

### Transfer Learning for Domain-Specific Embeddings

```python
class DomainAdaptationTrainer:
    def __init__(self, base_model, domain_name):
        self.base_model = base_model
        self.domain_name = domain_name
    
    def fine_tune_for_domain(self, domain_documents, domain_query_pairs):
        """Fine-tune embeddings for specific domain"""
        from sentence_transformers import losses, InputExample
        
        # Prepare training data
        train_examples = []
        
        for query, document in domain_query_pairs:
            train_examples.append(
                InputExample(
                    texts=[query, document],
                    label=1.0  # Positive pair
                )
            )
        
        # Fine-tune with domain data
        train_loss = losses.ContrastiveLoss(self.base_model)
        
        # Train
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=32
        )
        
        self.base_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100
        )
        
        return self.base_model
    
    def evaluate_domain_adaptation(self, test_pairs):
        """Evaluate fine-tuned model"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        queries = [pair[0] for pair in test_pairs]
        documents = [pair[1] for pair in test_pairs]
        
        query_embeddings = self.base_model.encode(queries)
        doc_embeddings = self.base_model.encode(documents)
        
        # Compute similarity
        similarities = cosine_similarity(query_embeddings, doc_embeddings)
        
        # Compute MRR
        mrr = 0
        for i, sim_row in enumerate(similarities):
            if sim_row[i] == np.max(sim_row):
                mrr += 1 / (np.argsort(-sim_row)[0] + 1)
        
        return mrr / len(test_pairs)
```

## Best Practices by Domain

| Domain | Key Challenge | Solution |
|--------|---------------|----------|
| Technical | Code understanding | AST parsing, signature extraction |
| Medical | Safety, terminology | Entity extraction, disclaimers |
| Legal | Citations, complex structure | Citation resolution, section parsing |
| Scientific | Context, citations | Citation context windows |
| Financial | Terminology, figures | Concept normalization, number parsing |

## Conclusion

Domain-specific RAG systems require understanding domain-specific structure, terminology, and safety requirements. Start with generic embeddings, then add domain-specific parsing, entity extraction, and fine-tuning for significant improvements. Always validate domain-specific safety requirements.
