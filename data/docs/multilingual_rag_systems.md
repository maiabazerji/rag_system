# Multilingual RAG Systems: Cross-Language Retrieval and Generation

## Introduction

Building RAG systems that work across languages presents unique challenges and opportunities. This guide covers techniques for indexing, retrieving, and generating across multiple languages.

## 1. Multilingual Embeddings

### Models Designed for Multilingual Tasks

**mBERT (Multilingual BERT)**
```
- Trained on 104 languages
- Shared embedding space across all languages
- 768 dimensions
- BEIR Performance: 59.2 nDCG@10 (multilingual)
- Best for: Balanced multilingual support
```

**XLM-R (Cross-Lingual RoBERTa)**
```
- 100+ languages in single model
- 768/1024 dimensions
- Superior cross-lingual transfer
- Better performance than mBERT
- Best for: Production multilingual systems
```

**BERT-M (Multilingual DistilBERT)**
```
- 50+ languages
- Lighter weight than mBERT
- Suitable for edge/mobile deployment
- Dimensions: 512
```

**BGE-M3 (Baichuan Embeddings Multilingual)**
```
- 100+ languages
- Supports sparse/dense/cross-lingual
- State-of-the-art on mBEIR benchmark
- Recommended for new projects
```

### Implementation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class MultilingualEmbedder:
    def __init__(self, model_name='intfloat/multilingual-e5-large'):
        self.model = SentenceTransformer(model_name)
        self.supported_languages = self._load_supported_languages()
    
    def embed_multilingual(self, texts, languages=None):
        """Embed texts in multiple languages"""
        embeddings = []
        
        for text, lang in zip(texts, languages or [None]*len(texts)):
            # Some models benefit from language-specific prefixes
            if lang:
                prefixed_text = f"[{lang}] {text}"
            else:
                prefixed_text = text
            
            emb = self.model.encode(prefixed_text)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def cross_lingual_search(self, query, query_lang, documents, doc_langs):
        """Search documents in different language than query"""
        query_emb = self.model.encode(query)
        doc_embs = self.embed_multilingual(documents, doc_langs)
        
        similarities = np.dot(doc_embs, query_emb)
        
        return np.argsort(similarities)[::-1]
    
    def _load_supported_languages(self):
        """Load list of supported languages"""
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic',
            'ru': 'Russian',
            'pt': 'Portuguese',
            'hi': 'Hindi',
            # ... more languages
        }
```

## 2. Language Detection and Routing

### Language Identification

```python
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # For reproducibility

class LanguageDetector:
    def __init__(self):
        self.confidence_threshold = 0.8
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            lang = detect(text)
            return lang
        except:
            return 'unknown'
    
    def detect_languages_in_text(self, text):
        """Detect multiple languages in same text"""
        from langdetect import detect_langs
        
        try:
            probs = detect_langs(text)
            return [
                {'language': p.lang, 'probability': p.prob}
                for p in probs
                if p.prob > 0.1
            ]
        except:
            return []
    
    def get_language_code(self, language_name):
        """Convert language name to code"""
        language_map = {
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'chinese': 'zh',
            'japanese': 'ja',
            # ... more mappings
        }
        return language_map.get(language_name.lower(), 'en')
```

### Language-Based Routing

```python
class MultilingualRouter:
    def __init__(self, detector, language_specific_retriever_configs):
        self.detector = detector
        self.configs = language_specific_retriever_configs
    
    def route_query(self, query):
        """Route query based on detected language"""
        lang = self.detector.detect_language(query)
        
        config = self.configs.get(lang, self.configs['en'])
        
        return {
            'language': lang,
            'retriever': config['retriever'],
            'index': config['index'],
            'generation_config': config['generation']
        }
```

## 3. Cross-Lingual Document Retrieval

### Zero-Shot Cross-Lingual Transfer

```python
class CrossLingualRetriever:
    def __init__(self, embedding_model, document_corpus):
        self.model = embedding_model
        self.corpus = document_corpus  # Multilingual documents
        self.embeddings = {}
        self._index_documents()
    
    def _index_documents(self):
        """Index documents in their original languages"""
        for doc_id, doc in self.corpus.items():
            language = doc.get('language', 'en')
            
            # Embed in shared multilingual space
            emb = self.model.embed_multilingual(
                [doc['text']],
                languages=[language]
            )
            
            self.embeddings[doc_id] = emb[0]
    
    def search_cross_lingual(self, query, query_lang, top_k=10):
        """Search documents in different languages"""
        query_emb = self.model.embed_multilingual(
            [query],
            languages=[query_lang]
        )[0]
        
        scores = []
        for doc_id, doc_emb in self.embeddings.items():
            similarity = np.dot(query_emb, doc_emb)
            scores.append((doc_id, similarity))
        
        results = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            {
                'doc_id': doc_id,
                'document': self.corpus[doc_id],
                'score': score
            }
            for doc_id, score in results
        ]
    
    def get_language_specific_results(self, query, query_lang, target_lang=None):
        """Retrieve documents in specific target language"""
        all_results = self.search_cross_lingual(query, query_lang)
        
        if target_lang:
            filtered = [
                r for r in all_results
                if r['document'].get('language') == target_lang
            ]
            return filtered
        
        return all_results
```

## 4. Machine Translation for RAG

### Translation Quality Assessment

```python
from transformers import MarianMTModel, MarianTokenizer

class TranslationEngine:
    def __init__(self):
        self.models = {}
        self._load_translation_models()
    
    def _load_translation_models(self):
        """Load translation models for common language pairs"""
        language_pairs = [
            ('en', 'es'),  # English to Spanish
            ('es', 'en'),  # Spanish to English
            ('en', 'fr'),  # English to French
            ('en', 'de'),  # English to German
            ('en', 'zh'),  # English to Chinese
        ]
        
        for src, tgt in language_pairs:
            model_name = f"Helsinki-NLP/Opus-MT-{src}-{tgt}"
            try:
                self.models[(src, tgt)] = {
                    'model': MarianMTModel.from_pretrained(model_name),
                    'tokenizer': MarianTokenizer.from_pretrained(model_name)
                }
            except:
                pass  # Model not available
    
    def translate(self, text, source_lang, target_lang, quality_threshold=0.7):
        """Translate text between languages"""
        pair = (source_lang, target_lang)
        
        if pair not in self.models:
            # Try reverse direction then back-translate
            if (target_lang, source_lang) in self.models:
                return self._back_translate(text, source_lang, target_lang)
            else:
                return None
        
        model_data = self.models[pair]
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        # Translate
        translated = model.generate(**inputs)
        
        # Decode
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Optional: assess quality
        quality = self._assess_translation_quality(
            text,
            result,
            source_lang,
            target_lang
        )
        
        if quality < quality_threshold:
            return None  # Low quality translation
        
        return result
    
    def _back_translate(self, text, source_lang, target_lang):
        """Translate via intermediate language"""
        # Find path through available models
        intermediate = self._find_translation_path(source_lang, target_lang)
        
        current = text
        for src, tgt in intermediate:
            current = self.translate(current, src, tgt)
            if current is None:
                return None
        
        return current
    
    def _assess_translation_quality(self, original, translated, src, tgt):
        """Assess translation quality (0-1)"""
        # Use back-translation score or embeddings
        # Simplified: just check if translation is non-empty
        return 0.9 if translated else 0.0
    
    def _find_translation_path(self, src, tgt):
        """Find path between source and target languages"""
        # Simple BFS to find translation path
        from collections import deque
        
        queue = deque([(src, [(src, None)])])
        visited = {src}
        
        while queue:
            current, path = queue.popleft()
            
            if current == tgt:
                return [(path[i][0], path[i+1][0]) for i in range(len(path)-1)]
            
            for (s, t) in self.models.keys():
                if s == current and t not in visited:
                    visited.add(t)
                    new_path = path + [(t, None)]
                    queue.append((t, new_path))
        
        return None
```

### Translation-Enhanced RAG

```python
class TranslationAugmentedRAG:
    def __init__(self, retriever, translator, llm):
        self.retriever = retriever
        self.translator = translator
        self.llm = llm
    
    def retrieve_and_translate(self, query, query_lang, target_langs=None):
        """Retrieve and optionally translate documents"""
        # Retrieve in query language
        results = self.retriever.search_cross_lingual(
            query,
            query_lang
        )
        
        # Optionally translate to other languages
        if target_langs:
            for target_lang in target_langs:
                for result in results:
                    translated = self.translator.translate(
                        result['document']['text'],
                        result['document']['language'],
                        target_lang
                    )
                    if translated:
                        result[f'text_{target_lang}'] = translated
        
        return results
```

## 5. Multilingual Generation

### Language-Aware LLM Prompting

```python
class MultilingualGenerator:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_multilingual(self, prompt, context, target_language):
        """Generate response in target language"""
        system_prompt = f"""You are a helpful AI assistant that responds in {target_language}.
        Answer based only on the provided context.
        Always respond in {target_language}, regardless of the language of the context or question."""
        
        full_prompt = f"""Context:
{context}

Query: {prompt}

Please respond in {target_language}:"""
        
        response = self.llm.generate(full_prompt, system_prompt=system_prompt)
        return response
    
    def maintain_language_consistency(self, query_lang, context_langs, target_lang):
        """Determine best language for generation"""
        # Prefer target language, but consider context language
        if target_lang == query_lang:
            return target_lang
        
        # If context is in query language, maintain that
        if query_lang in context_langs:
            return query_lang
        
        # Use target language as fallback
        return target_lang
```

## 6. Multilingual Evaluation

```python
class MultilingualEvaluator:
    def __init__(self, translator):
        self.translator = translator
    
    def evaluate_cross_lingual_retrieval(self, test_cases):
        """Evaluate retrieval across languages"""
        results = {
            'same_language': [],
            'cross_language': []
        }
        
        for test in test_cases:
            query = test['query']
            query_lang = test['language']
            relevant_docs = test['relevant_docs']
            
            retrieved = self.retriever.search_cross_lingual(
                query,
                query_lang
            )
            
            retrieved_ids = [r['doc_id'] for r in retrieved[:10]]
            
            # Evaluate
            if test.get('target_language') == query_lang:
                metric_type = 'same_language'
            else:
                metric_type = 'cross_language'
            
            recall = len(
                set(retrieved_ids) & set(relevant_docs)
            ) / len(relevant_docs)
            
            results[metric_type].append(recall)
        
        return {
            'same_language_recall': np.mean(results['same_language']),
            'cross_language_recall': np.mean(results['cross_language'])
        }
    
    def evaluate_translation_quality(self, test_pairs):
        """Evaluate translation accuracy"""
        from nltk.translate.bleu_score import sentence_bleu
        
        scores = []
        
        for original, reference_translation, src_lang, tgt_lang in test_pairs:
            translated = self.translator.translate(
                original,
                src_lang,
                tgt_lang
            )
            
            if translated:
                score = sentence_bleu(
                    [reference_translation.split()],
                    translated.split()
                )
                scores.append(score)
        
        return np.mean(scores) if scores else 0
```

## 7. Best Practices

1. **Use dedicated multilingual embeddings** - they outperform language-specific models
2. **Detect language automatically** - don't assume query language
3. **Avoid unnecessary translation** - better to search in shared embedding space
4. **Test cross-lingual retrieval** - benchmark same-language vs cross-language performance
5. **Cache translation results** - translation is expensive
6. **Monitor language distribution** - ensure balanced performance across languages

## Conclusion

Multilingual RAG requires careful consideration of embedding spaces, translation quality, and language-specific generation. Modern multilingual embeddings enable effective zero-shot cross-lingual retrieval. For best results, combine multilingual embeddings with optional translation and language-aware generation.
