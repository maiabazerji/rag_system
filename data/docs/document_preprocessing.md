# Document Preprocessing: Preparing Data for RAG

## Introduction

Quality of document preprocessing directly impacts RAG performance. This guide covers techniques for extracting, cleaning, and preparing documents.

## 1. Text Extraction

### Handling Different Document Types

```python
from typing import Dict, List, Tuple
import PyPDF2
from docx import Document
import json

class DocumentExtractor:
    """Extract text from various document formats"""
    
    def extract_text(self, file_path: str):
        """Extract text based on file type"""
        
        if file_path.endswith('.pdf'):
            return self._extract_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._extract_docx(file_path)
        elif file_path.endswith('.txt'):
            return self._extract_txt(file_path)
        elif file_path.endswith('.json'):
            return self._extract_json(file_path)
        elif file_path.endswith('.html'):
            return self._extract_html(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text.append(page.extract_text())
        
        return '\n'.join(text)
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = Document(file_path)
        
        paragraphs = [para.text for para in doc.paragraphs]
        
        # Also extract from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                paragraphs.append(' '.join(row_text))
        
        return '\n'.join(paragraphs)
    
    def _extract_txt(self, file_path: str) -> str:
        """Extract text from TXT"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _extract_json(self, file_path: str) -> str:
        """Extract text from JSON"""
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Convert to text (depends on JSON structure)
        if isinstance(data, dict):
            return ' '.join([f"{k}: {v}" for k, v in data.items()])
        elif isinstance(data, list):
            return ' '.join([str(item) for item in data])
        else:
            return str(data)
    
    def _extract_html(self, file_path: str) -> str:
        """Extract text from HTML"""
        from bs4 import BeautifulSoup
        
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        return soup.get_text()
```

## 2. Text Cleaning

### Normalization and Sanitization

```python
import re
import unicodedata

class TextCleaner:
    """Clean and normalize text"""
    
    def clean_text(self, text: str):
        """Apply all cleaning steps"""
        
        # Remove extra whitespace
        text = self._normalize_whitespace(text)
        
        # Remove special characters (preserve some)
        text = self._remove_special_chars(text)
        
        # Fix encoding issues
        text = self._fix_encoding(text)
        
        # Normalize punctuation
        text = self._normalize_punctuation(text)
        
        # Remove duplicate content
        text = self._remove_duplicates(text)
        
        return text
    
    def _normalize_whitespace(self, text: str):
        """Remove extra spaces, newlines, tabs"""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove tabs
        text = text.replace('\t', ' ')
        
        return text.strip()
    
    def _remove_special_chars(self, text: str, keep_chars=''):
        """Remove unwanted special characters"""
        # Keep alphanumeric, punctuation, and spaces
        keep_pattern = f'[^a-zA-Z0-9\\s.!?,;:\'-{keep_chars}]'
        text = re.sub(keep_pattern, '', text)
        
        return text
    
    def _fix_encoding(self, text: str):
        """Fix common encoding issues"""
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove control characters
        text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Cc')
        
        return text
    
    def _normalize_punctuation(self, text: str):
        """Normalize punctuation"""
        # Fix common issues
        text = re.sub(r'(?<!\w)\.(?!\w)', '.', text)  # Fix spacing around dots
        text = re.sub(r' +([.,!?])', r'\1', text)     # Remove space before punctuation
        
        return text
    
    def _remove_duplicates(self, text: str):
        """Remove duplicate sentences"""
        sentences = text.split('.')
        unique_sentences = []
        seen = set()
        
        for sent in sentences:
            sent = sent.strip()
            if sent and sent not in seen:
                unique_sentences.append(sent)
                seen.add(sent)
        
        return '. '.join(unique_sentences)
```

## 3. Language Detection

### Identifying and Filtering Languages

```python
from langdetect import detect, LangDetectException

class LanguageProcessor:
    """Detect and handle different languages"""
    
    def detect_language(self, text: str):
        """Detect language of text"""
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return 'unknown'
    
    def filter_by_language(self, text: str, allowed_languages: list):
        """Filter text by language"""
        lang = self.detect_language(text)
        
        if lang in allowed_languages:
            return text
        else:
            return None
    
    def detect_mixed_language(self, text: str):
        """Detect if text contains multiple languages"""
        from langdetect import detect_langs
        
        try:
            probabilities = detect_langs(text)
            
            languages = []
            for prob in probabilities:
                if prob.prob > 0.1:
                    languages.append((prob.lang, prob.prob))
            
            return languages
        except:
            return []
```

## 4. Entity Extraction

### Preserving Important Entities

```python
from transformers import pipeline

class EntityExtractor:
    """Extract and preserve named entities"""
    
    def __init__(self):
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-base-cased-finetuned-conll03-english")
    
    def extract_entities(self, text: str):
        """Extract named entities"""
        entities = self.ner_pipeline(text)
        
        organized = {}
        for entity in entities:
            entity_type = entity['entity'].split('-')[-1]
            
            if entity_type not in organized:
                organized[entity_type] = []
            
            organized[entity_type].append(entity['word'])
        
        return organized
    
    def preserve_entities_in_preprocessing(self, text: str):
        """Ensure entities survive cleaning"""
        entities = self.extract_entities(text)
        
        # Mark entities to preserve
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                text = text.replace(entity, f"[{entity_type}:{entity}]")
        
        return text
    
    def restore_entities(self, text: str):
        """Restore entities after processing"""
        # Restore marked entities
        text = re.sub(r'\[(\w+):(\w+)\]', r'\2', text)
        
        return text
```

## 5. Structure Preservation

### Maintaining Document Structure

```python
class StructurePreserver:
    """Maintain document structure during processing"""
    
    def parse_structure(self, text: str):
        """Parse document structure"""
        structure = {
            'headings': [],
            'sections': [],
            'paragraphs': [],
            'lists': [],
            'tables': []
        }
        
        # Detect headings (lines with specific formatting)
        for line in text.split('\n'):
            if re.match(r'^#{1,6}\s', line):  # Markdown heading
                structure['headings'].append(line)
            elif re.match(r'^[A-Z][A-Z\s]+$', line):  # ALL CAPS heading
                structure['headings'].append(line)
        
        return structure
    
    def maintain_hierarchy(self, text: str):
        """Preserve heading hierarchy"""
        lines = text.split('\n')
        processed = []
        current_section = None
        
        for line in lines:
            # Detect heading level
            heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
            
            if heading_match:
                level = len(heading_match.group(1))
                content = heading_match.group(2)
                
                # Create section marker
                section_id = f"section_{level}_{len(processed)}"
                processed.append(f"<{section_id}> {content}")
                current_section = section_id
            else:
                if current_section:
                    # Associate paragraph with section
                    processed.append(f"[{current_section}] {line}")
                else:
                    processed.append(line)
        
        return '\n'.join(processed)
```

## 6. Quality Assessment

### Evaluating Extracted Text Quality

```python
class TextQualityAssessor:
    """Assess quality of extracted/processed text"""
    
    def assess_quality(self, text: str):
        """Comprehensive quality assessment"""
        
        quality_scores = {
            'completeness': self._assess_completeness(text),
            'coherence': self._assess_coherence(text),
            'readability': self._assess_readability(text),
            'noise_level': self._assess_noise(text)
        }
        
        overall_score = np.mean(list(quality_scores.values()))
        
        return {
            'scores': quality_scores,
            'overall_quality': overall_score,
            'acceptable': overall_score > 0.6
        }
    
    def _assess_completeness(self, text: str):
        """Check if text is complete"""
        # Heuristic: has paragraphs, sentences, variety of content
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Good: 10-30 words per sentence
        completeness = 1 - abs(20 - avg_sentence_length) / 20
        return max(0, min(1, completeness))
    
    def _assess_coherence(self, text: str):
        """Check logical flow"""
        # Simple: check for common connective words
        connectives = ['therefore', 'however', 'additionally', 'furthermore', 'because']
        
        connective_count = sum(1 for conn in connectives if conn in text.lower())
        
        coherence = min(1, connective_count / 3)
        return coherence
    
    def _assess_readability(self, text: str):
        """Check readability metrics"""
        # Flesch Kincaid grade level (simplified)
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0
        
        avg_word_length = np.mean([len(w) for w in words])
        
        # Optimal: 4-6 character words
        readability = 1 - abs(5 - avg_word_length) / 5
        return max(0, min(1, readability))
    
    def _assess_noise(self, text: str):
        """Check for noise/garbage"""
        # Count unusual characters
        unusual_chars = len(re.findall(r'[^a-zA-Z0-9\s.!?,;:\'-\n]', text))
        noise_ratio = unusual_chars / len(text) if text else 0
        
        # Acceptable: <5% unusual characters
        noise_score = 1 - min(1, noise_ratio / 0.05)
        return max(0, noise_score)
```

## 7. Best Practices

1. **Preserve metadata** - keep source, date, author information
2. **Maintain structure** - preserve headings and sections
3. **Clean iteratively** - apply transformations gradually
4. **Preserve entities** - protect important proper nouns
5. **Language detection** - handle multilingual content
6. **Quality gates** - reject low-quality documents
7. **Audit trails** - track processing steps

## Conclusion

Document preprocessing is often 70% of RAG quality. Careful extraction, intelligent cleaning, and structure preservation lay the foundation for good retrieval and generation. Invest time in preprocessing; it pays dividends in downstream quality.
