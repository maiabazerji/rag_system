# Knowledge Graphs for RAG: Structured Knowledge Integration

## Introduction

Knowledge graphs provide structured representations of information that enhance RAG systems. This guide covers integration of knowledge graphs with retrieval and generation.

## 1. Knowledge Graph Basics

### Graph Construction

```python
from typing import List, Tuple, Dict
import networkx as nx

class KnowledgeGraphBuilder:
    """Build knowledge graphs from unstructured text"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def extract_entities_and_relations(self, text: str):
        """Extract entities and relations from text"""
        from transformers import pipeline
        
        # Extract entities
        ner_pipeline = pipeline("ner", model="dbmdz/bert-base-cased-finetuned-conll03-english")
        entities = ner_pipeline(text)
        
        # Extract relations (simplified)
        from transformers import pipeline
        
        relations = self._extract_relations_simple(text, entities)
        
        return entities, relations
    
    def add_triple(self, subject: str, relation: str, obj: str, confidence: float = 1.0):
        """Add RDF triple to graph"""
        if not self.graph.has_node(subject):
            self.graph.add_node(subject, type='entity')
        
        if not self.graph.has_node(obj):
            self.graph.add_node(obj, type='entity')
        
        self.graph.add_edge(subject, obj, relation=relation, confidence=confidence)
    
    def build_from_text(self, text: str):
        """Build knowledge graph from text"""
        entities, relations = self.extract_entities_and_relations(text)
        
        # Convert to triples
        triples = self._convert_to_triples(entities, relations)
        
        # Add to graph
        for subject, relation, obj, confidence in triples:
            self.add_triple(subject, relation, obj, confidence)
        
        return self.graph
    
    def _extract_relations_simple(self, text: str, entities: list):
        """Simplified relation extraction"""
        # In production, use relation extraction model
        relations = []
        
        # Simple patterns
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities[i+1:], i+1):
                relations.append({
                    'subject': ent1['word'],
                    'object': ent2['word'],
                    'relation': 'related_to'
                })
        
        return relations
    
    def _convert_to_triples(self, entities: list, relations: list):
        """Convert to RDF triples"""
        triples = []
        
        for rel in relations:
            triples.append((
                rel['subject'],
                rel['relation'],
                rel['object'],
                0.8  # confidence
            ))
        
        return triples
```

## 2. Knowledge Graph Retrieval

### Graph-Based Retrieval

```python
class KnowledgeGraphRetriever:
    """Retrieve information using knowledge graph"""
    
    def __init__(self, knowledge_graph: nx.DiGraph):
        self.graph = knowledge_graph
    
    def retrieve_by_entity(self, entity: str, hop_distance: int = 2):
        """Retrieve information about entity and related entities"""
        
        if entity not in self.graph:
            return []
        
        # Get subgraph around entity
        nodes = self._get_neighbors(entity, hop_distance)
        
        subgraph = self.graph.subgraph(nodes)
        
        # Convert to readable format
        facts = []
        for u, v, data in subgraph.edges(data=True):
            facts.append({
                'subject': u,
                'relation': data.get('relation', 'unknown'),
                'object': v,
                'confidence': data.get('confidence', 1.0)
            })
        
        return facts
    
    def retrieve_by_relation(self, relation: str):
        """Retrieve all facts with specific relation"""
        facts = []
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('relation') == relation:
                facts.append({
                    'subject': u,
                    'relation': relation,
                    'object': v,
                    'confidence': data.get('confidence', 1.0)
                })
        
        return facts
    
    def find_path(self, start: str, end: str):
        """Find path between two entities"""
        try:
            path = nx.shortest_path(self.graph, start, end)
            return path
        except:
            return None
    
    def _get_neighbors(self, entity: str, distance: int):
        """Get all nodes within hop distance"""
        neighbors = {entity}
        
        for _ in range(distance):
            new_neighbors = set()
            for node in neighbors:
                # Predecessors and successors
                new_neighbors.update(self.graph.predecessors(node))
                new_neighbors.update(self.graph.successors(node))
            neighbors.update(new_neighbors)
        
        return neighbors
```

## 3. Knowledge Graph Integration with RAG

### Enhanced Context from Knowledge Graphs

```python
class KnowledgeGraphAugmentedRAG:
    """RAG enhanced with knowledge graph"""
    
    def __init__(self, retriever, knowledge_graph, embedding_model, llm):
        self.retriever = retriever
        self.kg = KnowledgeGraphRetriever(knowledge_graph)
        self.embedding_model = embedding_model
        self.llm = llm
    
    def retrieve_augmented(self, query: str):
        """Retrieve with knowledge graph augmentation"""
        
        # Standard retrieval
        docs = self.retriever.retrieve(query)
        
        # Extract entities from query
        entities = self._extract_query_entities(query)
        
        # Retrieve from knowledge graph
        kg_facts = []
        for entity in entities:
            facts = self.kg.retrieve_by_entity(entity)
            kg_facts.extend(facts)
        
        # Combine and rank
        combined = self._combine_retrieval(docs, kg_facts, query)
        
        return combined
    
    def _extract_query_entities(self, query: str):
        """Extract entities from query"""
        from transformers import pipeline
        
        ner = pipeline("ner")
        entities = ner(query)
        
        return [e['word'] for e in entities if e['entity'] in ['B-PER', 'B-LOC', 'B-ORG']]
    
    def _combine_retrieval(self, docs: list, kg_facts: list, query: str):
        """Combine and rank document and kg results"""
        
        combined = []
        
        # Add documents
        for doc in docs:
            combined.append({
                'type': 'document',
                'content': doc['text'],
                'score': doc.get('score', 0),
                'source': 'document_retrieval'
            })
        
        # Add knowledge graph facts
        query_emb = self.embedding_model.encode(query)
        
        for fact in kg_facts:
            fact_text = f"{fact['subject']} {fact['relation']} {fact['object']}"
            fact_emb = self.embedding_model.encode(fact_text)
            
            similarity = np.dot(query_emb, fact_emb)
            
            combined.append({
                'type': 'fact',
                'content': fact_text,
                'score': similarity * fact.get('confidence', 1.0),
                'source': 'knowledge_graph',
                'fact': fact
            })
        
        # Sort by relevance score
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        return combined[:20]  # Top 20
    
    def generate_with_kg(self, query: str):
        """Generate using both documents and knowledge graph"""
        
        # Retrieve augmented
        combined = self.retrieve_augmented(query)
        
        # Format context
        doc_context = "\n".join([
            c['content'] for c in combined
            if c['type'] == 'document'
        ][:3])  # Top 3 documents
        
        kg_context = "\n".join([
            c['content'] for c in combined
            if c['type'] == 'fact'
        ][:5])  # Top 5 facts
        
        # Build prompt
        prompt = f"""Use the following information to answer the question:

DOCUMENTS:
{doc_context}

KNOWLEDGE FACTS:
{kg_context}

QUESTION: {query}

ANSWER:"""
        
        response = self.llm.generate(prompt)
        
        return response
```

## 4. Knowledge Graph Reasoning

### Path-Based Reasoning

```python
class KGReasoner:
    """Perform reasoning over knowledge graphs"""
    
    def __init__(self, knowledge_graph):
        self.graph = knowledge_graph
    
    def find_relations(self, entity1: str, entity2: str):
        """Find relations between two entities"""
        
        try:
            path = nx.shortest_path(self.graph, entity1, entity2)
            
            # Get relations along path
            relations = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                rel_data = self.graph[u][v]
                relations.append({
                    'from': u,
                    'relation': rel_data.get('relation'),
                    'to': v
                })
            
            return relations
        except:
            return []
    
    def transitive_closure(self, relation: str):
        """Compute transitive closure for relation"""
        # E.g., if A->parent->B and B->parent->C, then A->grandparent->C
        
        transitive = []
        
        # Find all pairs with relation
        for u, v, data in self.graph.edges(data=True):
            if data.get('relation') == relation:
                # Find paths continuing with same relation
                for w in self.graph.successors(v):
                    if self.graph[v][w].get('relation') == relation:
                        transitive.append((u, f'({relation})*2', w))
        
        return transitive
    
    def answer_question(self, question: str):
        """Answer question using knowledge graph"""
        
        # Parse question
        parsed = self._parse_question(question)
        
        if parsed.get('type') == 'relation':
            # "What is X relation to Y?"
            relations = self.find_relations(parsed['entity1'], parsed['entity2'])
            return relations
        
        elif parsed.get('type') == 'property':
            # "What is property of X?"
            return self._get_properties(parsed['entity'])
        
        return None
    
    def _parse_question(self, question: str):
        """Parse natural language question"""
        # Simplified parsing
        
        if ' is ' in question and ' to ' in question:
            parts = question.split(' is ')
            entity1 = parts[0].strip()
            
            rest = parts[1]
            entity2 = rest.split(' to ')[-1].strip()
            
            return {'type': 'relation', 'entity1': entity1, 'entity2': entity2}
        
        return {'type': 'unknown'}
    
    def _get_properties(self, entity: str):
        """Get all properties of entity"""
        props = []
        
        # Outgoing relations
        for v, data in self.graph[entity].items():
            props.append({
                'relation': data.get('relation'),
                'value': v
            })
        
        return props
```

## 5. Knowledge Graph Maintenance

### Updating and Cleaning

```python
class KnowledgeGraphMaintainer:
    """Maintain knowledge graph quality"""
    
    def __init__(self, graph):
        self.graph = graph
    
    def remove_low_confidence(self, threshold: float = 0.5):
        """Remove edges below confidence threshold"""
        edges_to_remove = []
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('confidence', 1.0) < threshold:
                edges_to_remove.append((u, v))
        
        self.graph.remove_edges_from(edges_to_remove)
    
    def merge_entities(self, entity1: str, entity2: str):
        """Merge duplicate entities"""
        if entity1 not in self.graph or entity2 not in self.graph:
            return
        
        # Move all edges from entity2 to entity1
        for pred in list(self.graph.predecessors(entity2)):
            for u, v, data in self.graph.in_edges(entity2, data=True):
                self.graph.add_edge(u, entity1, **data)
        
        for succ in list(self.graph.successors(entity2)):
            for u, v, data in self.graph.out_edges(entity2, data=True):
                self.graph.add_edge(entity1, v, **data)
        
        # Remove entity2
        self.graph.remove_node(entity2)
    
    def detect_contradictions(self):
        """Detect contradictory facts"""
        contradictions = []
        
        # Simple heuristic: opposite relations
        opposite_pairs = [
            ('is_parent_of', 'is_child_of'),
            ('before', 'after'),
            ('greater_than', 'less_than')
        ]
        
        for rel1, rel2 in opposite_pairs:
            for u, v, data1 in self.graph.edges(data=True):
                if data1.get('relation') == rel1:
                    if self.graph.has_edge(v, u):
                        data2 = self.graph[v][u]
                        if data2.get('relation') == rel2:
                            contradictions.append({
                                'type': 'contradiction',
                                'triple1': (u, rel1, v),
                                'triple2': (v, rel2, u)
                            })
        
        return contradictions
```

## Best Practices

1. **Use typed entities** - maintain entity types
2. **Track confidence** - store confidence scores
3. **Regular maintenance** - remove contradictions
4. **Combine with text** - knowledge graphs + documents
5. **Version control** - track graph evolution
6. **Query optimization** - index frequent queries

## Conclusion

Knowledge graphs provide structured alternative to text for retrieval. Most effective when combined with document retrieval - knowledge graphs for structure and relationships, documents for detailed information. Integration requires careful entity extraction and deduplication.
