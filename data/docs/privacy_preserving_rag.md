# Privacy-Preserving RAG: Building Secure Systems

## Introduction

Privacy is critical for RAG systems handling sensitive data. This guide covers techniques for maintaining privacy while preserving retrieval quality.

## 1. Data Anonymization

### Text Anonymization

```python
import re
from typing import List, Dict

class TextAnonymizer:
    def __init__(self):
        self.entity_mapping = {}
        self.counter = {}
    
    def anonymize_text(self, text: str, entity_types=None):
        """Anonymize sensitive entities in text"""
        if entity_types is None:
            entity_types = ['person', 'email', 'phone', 'ssn']
        
        anonymized = text
        
        for entity_type in entity_types:
            anonymized = self._anonymize_entity_type(
                anonymized,
                entity_type
            )
        
        return anonymized
    
    def _anonymize_entity_type(self, text: str, entity_type: str):
        """Anonymize specific entity type"""
        
        if entity_type == 'email':
            pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            replacement = self._get_placeholder('email')
        
        elif entity_type == 'phone':
            pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            replacement = self._get_placeholder('phone')
        
        elif entity_type == 'ssn':
            pattern = r'\b\d{3}-\d{2}-\d{4}\b'
            replacement = self._get_placeholder('ssn')
        
        elif entity_type == 'person':
            # More complex - use NER
            return self._anonymize_names(text)
        
        else:
            return text
        
        return re.sub(pattern, replacement, text)
    
    def _anonymize_names(self, text: str):
        """Anonymize person names using NER"""
        from transformers import pipeline
        
        nlp = pipeline("ner", model="dbmdz/bert-base-cased-finetuned-conll03-english")
        
        entities = nlp(text)
        
        anonymized = text
        for entity in reversed(entities):  # Reverse to maintain indices
            if entity['entity'] in ['B-PER', 'I-PER']:
                start = entity['start']
                end = entity['end']
                replacement = self._get_placeholder('person')
                anonymized = anonymized[:start] + replacement + anonymized[end:]
        
        return anonymized
    
    def _get_placeholder(self, entity_type: str):
        """Get placeholder for entity type"""
        if entity_type not in self.counter:
            self.counter[entity_type] = 0
        
        self.counter[entity_type] += 1
        
        return f"[{entity_type.upper()}_{self.counter[entity_type]}]"
    
    def deanonymize(self, text: str, mapping: Dict):
        """Reverse anonymization using mapping"""
        deanon = text
        
        for placeholder, original in mapping.items():
            deanon = deanon.replace(placeholder, original)
        
        return deanon
```

## 2. Federated Learning for RAG

### Distributed Training Without Data Sharing

```python
import torch
from torch.nn import Module

class FederatedEmbeddingTrainer:
    """Train embeddings across multiple parties without sharing data"""
    
    def __init__(self, num_parties: int, embedding_dim: int = 1536):
        self.num_parties = num_parties
        self.embedding_dim = embedding_dim
        self.global_model = self._init_model()
        self.party_models = [
            self._init_model() for _ in range(num_parties)
        ]
    
    def federated_training_round(self, party_local_data_list: List, epochs: int = 1):
        """Execute one round of federated training"""
        # Each party trains locally
        local_updates = []
        
        for party_id, local_data in enumerate(party_local_data_list):
            # Train locally without sharing raw data
            update = self._train_party_locally(
                self.party_models[party_id],
                local_data,
                epochs
            )
            local_updates.append(update)
        
        # Aggregate updates (e.g., federated averaging)
        self._aggregate_updates(local_updates)
        
        # Distribute updated model
        for party_id in range(self.num_parties):
            self.party_models[party_id].load_state_dict(
                self.global_model.state_dict()
            )
    
    def _train_party_locally(self, model, local_data, epochs):
        """Train model on party's local data"""
        optimizer = torch.optim.Adam(model.parameters())
        
        for epoch in range(epochs):
            for batch in local_data:
                query, document = batch
                
                # Forward pass
                query_emb = model.encode(query)
                doc_emb = model.encode(document)
                
                # Contrastive loss
                loss = self._contrastive_loss(query_emb, doc_emb)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return model.state_dict()
    
    def _aggregate_updates(self, local_updates):
        """Aggregate using Federated Averaging"""
        # Simple average of parameters
        avg_state = {}
        
        for key in local_updates[0].keys():
            avg_state[key] = torch.stack([
                update[key] for update in local_updates
            ]).mean(dim=0)
        
        self.global_model.load_state_dict(avg_state)
    
    def _init_model(self):
        """Initialize embedding model"""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('intfloat/e5-large')
```

## 3. Differential Privacy

### Differentially Private Embeddings

```python
import numpy as np

class DifferentiallyPrivateEmbedder:
    """Add differential privacy to embeddings"""
    
    def __init__(self, embedding_model, epsilon: float = 1.0, delta: float = 1e-5):
        self.model = embedding_model
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta       # Failure probability
    
    def embed_with_dp(self, texts: List[str]):
        """Embed texts with differential privacy"""
        # Get base embeddings
        embeddings = self.model.encode(texts)
        
        # Add Laplace noise for differential privacy
        sensitivity = self._compute_sensitivity(embeddings)
        
        scale = sensitivity / self.epsilon
        
        # Add noise
        noise = np.random.laplace(0, scale, embeddings.shape)
        
        private_embeddings = embeddings + noise
        
        return private_embeddings
    
    def _compute_sensitivity(self, embeddings):
        """Compute local sensitivity"""
        # For embeddings, sensitivity depends on model
        # Simplified: use L2 norm bound
        return np.max(np.linalg.norm(embeddings, axis=1))
    
    def search_with_dp(self, query_text: str, documents: List[str], top_k: int = 10):
        """Search with differential privacy"""
        # Query embedding with DP
        query_emb = self.embed_with_dp([query_text])[0]
        
        # Document embeddings with DP
        doc_embs = self.embed_with_dp(documents)
        
        # Search
        similarities = np.dot(doc_embs, query_emb)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                'document': documents[i],
                'score': similarities[i]
            }
            for i in top_indices
        ]
```

## 4. Secure Enclave Computation

### Trusted Execution Environment (TEE)

```python
class SecureEnclaveRetriever:
    """Retrieve within secure enclave"""
    
    def __init__(self, enclave_endpoint: str):
        self.enclave_endpoint = enclave_endpoint
    
    def retrieve_in_enclave(self, encrypted_query: bytes, index_commitment: str):
        """Retrieve within TEE"""
        # Query is only decrypted inside secure enclave
        request = {
            'encrypted_query': encrypted_query,
            'index_commitment': index_commitment,
            'operation': 'retrieve'
        }
        
        # Send to TEE (e.g., Intel SGX, AMD SEV)
        response = self._send_to_enclave(request)
        
        # Get encrypted results
        encrypted_results = response['encrypted_results']
        
        return encrypted_results
    
    def decrypt_results(self, encrypted_results: bytes, decryption_key: bytes):
        """Decrypt results from enclave"""
        from cryptography.fernet import Fernet
        
        cipher = Fernet(decryption_key)
        decrypted = cipher.decrypt(encrypted_results)
        
        return decrypted
    
    def _send_to_enclave(self, request: dict):
        """Send request to secure enclave"""
        import requests
        
        response = requests.post(
            f"{self.enclave_endpoint}/retrieve",
            json=request
        )
        
        return response.json()
```

## 5. Homomorphic Encryption

### Computing on Encrypted Data

```python
class HomomorphicRAG:
    """RAG with homomorphic encryption"""
    
    def __init__(self):
        # Initialize HE scheme
        # In practice, use libraries like TenSEAL
        self.he_scheme = self._init_homomorphic_encryption()
    
    def _init_homomorphic_encryption(self):
        """Initialize homomorphic encryption"""
        # Example with TenSEAL
        try:
            import tenseal as ts
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_lengths=[40, 21, 21, 21, 21, 21, 40]
            )
            context.generate_galois_keys()
            return context
        except:
            return None
    
    def encrypt_query(self, query_embedding: np.ndarray):
        """Encrypt query embedding"""
        if self.he_scheme is None:
            return None
        
        encrypted = self.he_scheme.encrypt(query_embedding)
        return encrypted
    
    def compute_similarity_encrypted(self, encrypted_query, encrypted_doc):
        """Compute similarity on encrypted data"""
        # Dot product on encrypted data
        similarity = encrypted_query * encrypted_doc
        
        return similarity
    
    def decrypt_results(self, encrypted_similarity, private_key):
        """Decrypt similarity results"""
        similarity = encrypted_similarity.decrypt(private_key)
        return similarity
```

## 6. Privacy-Preserving Evaluation

### Federated Evaluation

```python
class FederatedEvaluator:
    """Evaluate RAG without centralizing data"""
    
    def __init__(self, num_parties: int):
        self.num_parties = num_parties
        self.party_metrics = {}
    
    def federated_evaluation(self, query_sets_per_party: List[List[str]]):
        """Evaluate across parties"""
        # Each party evaluates locally
        metrics_per_party = []
        
        for party_id, queries in enumerate(query_sets_per_party):
            metrics = self._evaluate_party_locally(
                party_id,
                queries
            )
            metrics_per_party.append(metrics)
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(metrics_per_party)
        
        return aggregated
    
    def _evaluate_party_locally(self, party_id: int, queries: List[str]):
        """Evaluate on party's local data"""
        # Don't share queries or results
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'ndcg': 0.0,
            'query_count': len(queries)
        }
        
        # Evaluate locally (implementation omitted)
        
        return metrics
    
    def _aggregate_metrics(self, metrics_per_party: List[Dict]):
        """Aggregate without revealing party-specific metrics"""
        aggregated = {}
        
        # Use secure aggregation protocol (e.g., SecAgg)
        for metric_name in metrics_per_party[0].keys():
            values = [m[metric_name] for m in metrics_per_party]
            
            # Aggregate (e.g., weighted average)
            aggregated[metric_name] = np.mean(values)
        
        return aggregated
```

## 7. Data Governance

### Audit Logging

```python
import json
from datetime import datetime

class PrivacyAuditLog:
    """Log all data access for privacy compliance"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
    
    def log_access(self, user_id: str, query: str, documents_accessed: List[str], timestamp=None):
        """Log data access"""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'user_id': user_id,
            'action': 'retrieve',
            'query_hash': self._hash_query(query),
            'num_docs_accessed': len(documents_accessed),
            'doc_ids': [self._hash_doc(d) for d in documents_accessed]
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_deletion(self, document_id: str, user_id: str, reason: str):
        """Log data deletion for right-to-be-forgotten"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': 'deletion',
            'document_id': self._hash_doc(document_id),
            'requested_by': user_id,
            'reason': reason
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _hash_query(self, query: str):
        """Hash query for privacy"""
        import hashlib
        return hashlib.sha256(query.encode()).hexdigest()
    
    def _hash_doc(self, doc: str):
        """Hash document for privacy"""
        import hashlib
        return hashlib.sha256(doc.encode()).hexdigest()
    
    def get_user_audit_trail(self, user_id: str):
        """Retrieve audit trail for user"""
        trail = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get('user_id') == user_id:
                    trail.append(entry)
        
        return trail
```

### Right to Deletion

```python
class RightToBeForgettenManager:
    """Handle data deletion requests"""
    
    def __init__(self, db, vector_db, audit_log):
        self.db = db
        self.vector_db = vector_db
        self.audit_log = audit_log
    
    def delete_user_data(self, user_id: str):
        """Delete all data associated with user"""
        # Find all documents associated with user
        user_docs = self.db.find_documents_by_user(user_id)
        
        for doc_id in user_docs:
            # Delete from database
            self.db.delete_document(doc_id)
            
            # Delete from vector database
            self.vector_db.delete_document(doc_id)
            
            # Log deletion
            self.audit_log.log_deletion(
                doc_id,
                user_id,
                "Right to be forgotten request"
            )
    
    def anonymize_user_data(self, user_id: str):
        """Anonymize instead of delete (if deletion not possible)"""
        # Find documents
        user_docs = self.db.find_documents_by_user(user_id)
        
        anonymizer = TextAnonymizer()
        
        for doc_id in user_docs:
            # Get document
            doc = self.db.get_document(doc_id)
            
            # Anonymize
            anon_text = anonymizer.anonymize_text(doc['text'])
            
            # Update
            self.db.update_document(doc_id, {'text': anon_text})
            
            # Update vector database
            new_embedding = self._encode(anon_text)
            self.vector_db.update_embedding(doc_id, new_embedding)
```

## Best Practices Checklist

- [ ] Anonymize sensitive data before storage
- [ ] Use encryption for data in transit and at rest
- [ ] Implement access controls and logging
- [ ] Use TLS 1.3 for all communications
- [ ] Regularly audit access patterns
- [ ] Implement right-to-deletion capabilities
- [ ] Use VPNs for sensitive deployments
- [ ] Encrypt embedding indices
- [ ] Monitor for privacy violations
- [ ] Comply with GDPR/CCPA requirements

## Conclusion

Privacy-preserving RAG requires careful consideration at every layer. Combine multiple techniques (anonymization, encryption, differential privacy) for defense in depth. Always balance privacy with utility and ensure compliance with regulations.
