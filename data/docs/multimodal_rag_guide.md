# Multimodal RAG: Handling Images, Videos, and More

## Introduction

Multimodal RAG extends retrieval to images, videos, tables, and other content types. This guide covers architectures, models, and implementation strategies for multimodal retrieval-augmented generation.

## 1. Multimodal Embedding Models

### Vision Encoders

**CLIP (Contrastive Language-Image Pre-training)**
```
- Embeds images and text in shared space
- Zero-shot image classification capabilities
- Fast inference (50-100ms per image)
- Effective for matching images to text queries
- Model: openai/clip-vit-large-patch14
```

**DINOv2**
```
- Vision transformer without labels
- Rich visual representations
- Better for fine-grained visual concepts
- More computationally expensive than CLIP
- Model: facebookresearch/dinov2
```

### Implementation

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class MultimodalEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    
    def embed_image(self, image_path):
        """Embed image to vector"""
        image = Image.open(image_path)
        
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(
                **inputs
            )
        
        # Normalize
        image_features = image_features / image_features.norm(
            p=2,
            dim=-1,
            keepdim=True
        )
        
        return image_features.cpu().numpy()
    
    def embed_text(self, text):
        """Embed text to vector"""
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # Normalize
        text_features = text_features / text_features.norm(
            p=2,
            dim=-1,
            keepdim=True
        )
        
        return text_features.cpu().numpy()
    
    def search(self, query_text, image_embeddings):
        """Search images by text query"""
        query_emb = self.embed_text(query_text)
        
        # Compute similarity
        similarities = [
            np.dot(query_emb[0], img_emb)
            for img_emb in image_embeddings
        ]
        
        return np.argsort(similarities)[::-1]
```

## 2. Table and Structured Data Retrieval

### Table Understanding

```python
class TableRetriever:
    def __init__(self, embedding_model):
        self.model = embedding_model
    
    def index_tables(self, tables):
        """Index tables for retrieval"""
        table_embeddings = []
        
        for table in tables:
            # Convert table to text
            table_text = self._table_to_text(table)
            
            # Embed
            embedding = self.model.encode(table_text)
            table_embeddings.append({
                'embedding': embedding,
                'table': table,
                'text': table_text
            })
        
        return table_embeddings
    
    def _table_to_text(self, table):
        """Convert table to searchable text"""
        text_parts = []
        
        # Add column names
        if 'columns' in table:
            text_parts.append("Columns: " + ", ".join(table['columns']))
        
        # Add first few rows
        if 'data' in table:
            for row in table['data'][:3]:
                text_parts.append(str(row))
        
        return " ".join(text_parts)
    
    def retrieve_table_cells(self, query, table):
        """Find relevant cells in table"""
        # Embed each cell
        cell_embeddings = {}
        
        for i, row in enumerate(table['data']):
            for j, cell in enumerate(row):
                cell_text = str(cell)
                cell_emb = self.model.encode(cell_text)
                cell_embeddings[(i, j)] = cell_emb
        
        # Find most similar cells
        query_emb = self.model.encode(query)
        
        similarities = [
            (pos, np.dot(query_emb, emb))
            for pos, emb in cell_embeddings.items()
        ]
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
```

### Table Context Preservation

```python
class TableContextualizer:
    """Maintain table context during retrieval"""
    
    def extract_table_context(self, table, relevant_cells):
        """Extract surrounding context for retrieved cells"""
        context = {
            'columns': table['columns'],
            'relevant_rows': []
        }
        
        row_indices = set(cell[0] for cell in relevant_cells)
        
        for row_idx in row_indices:
            row_with_context = {
                'row_index': row_idx,
                'data': table['data'][row_idx],
                'nearby_rows': self._get_nearby_rows(
                    table, row_idx, window=1
                )
            }
            context['relevant_rows'].append(row_with_context)
        
        return context
    
    def _get_nearby_rows(self, table, row_idx, window=1):
        """Get rows near the specified row"""
        nearby = []
        data = table['data']
        
        for i in range(
            max(0, row_idx - window),
            min(len(data), row_idx + window + 1)
        ):
            if i != row_idx:
                nearby.append(data[i])
        
        return nearby
```

## 3. Document Understanding (Layout Analysis)

### OCR and Layout Understanding

```python
class DocumentUnderstandingEngine:
    def __init__(self, ocr_engine=None, layout_model=None):
        self.ocr = ocr_engine or self._init_ocr()
        self.layout_model = layout_model
    
    def process_document(self, image_path):
        """Extract text, layout, and visual features"""
        # OCR to get text
        ocr_result = self.ocr.extract_text(image_path)
        
        # Layout detection (tables, headers, etc.)
        layout = self.layout_model.detect_layout(image_path)
        
        # Create structured document
        structured_doc = self._structure_document(
            ocr_result,
            layout
        )
        
        return structured_doc
    
    def _structure_document(self, ocr_result, layout):
        """Create structured representation"""
        doc = {
            'full_text': ocr_result['text'],
            'sections': [],
            'tables': [],
            'images': []
        }
        
        for element in layout['elements']:
            if element['type'] == 'heading':
                doc['sections'].append({
                    'heading': element['text'],
                    'content': []
                })
            elif element['type'] == 'table':
                doc['tables'].append(element)
            elif element['type'] == 'image':
                doc['images'].append(element)
            elif element['type'] == 'text' and doc['sections']:
                doc['sections'][-1]['content'].append(element['text'])
        
        return doc
```

## 4. Video Retrieval

### Keyframe Extraction

```python
import cv2
import numpy as np

class VideoKeyframeExtractor:
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.keyframes = []
    
    def extract_keyframes(self, video_path, method='uniform', fps=1):
        """Extract representative frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if method == 'uniform':
            frame_indices = np.linspace(
                0,
                total_frames - 1,
                int(total_frames * fps / 30)
            ).astype(int)
        elif method == 'variance':
            frame_indices = self._select_by_variance(cap)
        
        keyframes = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                keyframes.append({
                    'frame': frame,
                    'timestamp': idx / fps,
                    'frame_idx': idx
                })
        
        cap.release()
        return keyframes
    
    def _select_by_variance(self, video_capture, sample_rate=30):
        """Select frames with high variance (scene changes)"""
        variances = []
        prev_frame = None
        frame_indices = []
        frame_idx = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                if prev_frame is not None:
                    # Compute frame difference
                    diff = cv2.absdiff(prev_frame, frame)
                    variance = np.var(diff)
                    
                    variances.append((frame_idx, variance))
                
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame_idx += 1
        
        # Select frames with highest variance
        variances.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([v[0] for v in variances[:10]])
        
        return selected_indices
```

### Video Search

```python
class VideoSearcher:
    def __init__(self, embedding_model):
        self.model = embedding_model
    
    def index_video(self, video_path, metadata=None):
        """Index video for search"""
        extractor = VideoKeyframeExtractor(self.model)
        keyframes = extractor.extract_keyframes(video_path)
        
        video_index = {
            'video_path': video_path,
            'metadata': metadata,
            'keyframes': []
        }
        
        for kf in keyframes:
            # Embed keyframe image
            frame_emb = self.model.encode_image(kf['frame'])
            
            # Also embed visual description
            desc = self._describe_frame(kf['frame'])
            text_emb = self.model.encode_text(desc)
            
            video_index['keyframes'].append({
                'timestamp': kf['timestamp'],
                'frame_idx': kf['frame_idx'],
                'image_embedding': frame_emb,
                'text_embedding': text_emb,
                'description': desc
            })
        
        return video_index
    
    def search_videos(self, query, video_indices):
        """Search across multiple videos"""
        query_emb_text = self.model.encode_text(query)
        
        results = []
        
        for video_idx in video_indices:
            for kf in video_idx['keyframes']:
                similarity = np.dot(
                    query_emb_text,
                    kf['text_embedding']
                )
                
                results.append({
                    'video': video_idx['video_path'],
                    'timestamp': kf['timestamp'],
                    'description': kf['description'],
                    'score': similarity
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
```

## 5. Multimodal Fusion

### Cross-Modal Retrieval

```python
class CrossModalRetriever:
    """Retrieve across different modalities"""
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    def retrieve_multimodal(self, query, corpus):
        """Retrieve images, text, tables matching query"""
        # Could be text or image query
        if isinstance(query, str):
            query_emb = self.embedder.embed_text(query)
        else:
            query_emb = self.embedder.embed_image(query)
        
        results = []
        
        # Search text documents
        for doc in corpus.get('text_docs', []):
            doc_emb = self.embedder.embed_text(doc['text'])
            score = np.dot(query_emb, doc_emb)
            results.append({
                'type': 'text',
                'content': doc,
                'score': score
            })
        
        # Search images
        for img in corpus.get('images', []):
            img_emb = self.embedder.embed_image(img['path'])
            score = np.dot(query_emb, img_emb)
            results.append({
                'type': 'image',
                'content': img,
                'score': score
            })
        
        # Search tables
        for table in corpus.get('tables', []):
            table_text = self._table_to_text(table)
            table_emb = self.embedder.embed_text(table_text)
            score = np.dot(query_emb, table_emb)
            results.append({
                'type': 'table',
                'content': table,
                'score': score
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _table_to_text(self, table):
        """Convert table to text for embedding"""
        text_parts = [", ".join(str(x) for x in row) for row in table['data']]
        return " | ".join(text_parts)
```

## 6. Multimodal Generation

### Context Integration

```python
class MultimodalRAGGenerator:
    """Generate using multimodal context"""
    
    def __init__(self, llm, vision_model):
        self.llm = llm
        self.vision_model = vision_model
    
    def generate_with_multimodal_context(
        self,
        query,
        retrieved_items
    ):
        """Generate answer using mixed modality context"""
        
        # Organize retrieved items by type
        text_docs = [
            r for r in retrieved_items if r['type'] == 'text'
        ]
        images = [r for r in retrieved_items if r['type'] == 'image']
        tables = [r for r in retrieved_items if r['type'] == 'table']
        
        # Build prompt
        prompt_parts = []
        prompt_parts.append(f"Question: {query}\n")
        
        # Add text context
        if text_docs:
            prompt_parts.append("Relevant text:")
            for doc in text_docs[:3]:
                prompt_parts.append(f"- {doc['content']['text'][:200]}")
        
        # Add table context
        if tables:
            prompt_parts.append("\nRelevant data:")
            for table in tables[:2]:
                table_text = self._format_table(table['content'])
                prompt_parts.append(table_text)
        
        # Note about images (actual multimodal LLMs would process directly)
        if images:
            prompt_parts.append(
                f"\n[{len(images)} relevant image(s) shown separately]"
            )
        
        final_prompt = "\n".join(prompt_parts)
        
        # Generate response
        response = self.llm.generate(final_prompt)
        
        return response
    
    def _format_table(self, table):
        """Format table for LLM input"""
        lines = []
        if 'columns' in table:
            lines.append(" | ".join(table['columns']))
            lines.append("-" * 40)
        
        for row in table['data'][:5]:
            lines.append(" | ".join(str(x) for x in row))
        
        return "\n".join(lines)
```

## 7. Production Considerations

### Multimodal Caching

```python
class MultimodalCache:
    """Cache multimodal embeddings"""
    
    def __init__(self, backend='redis'):
        self.cache = self._init_cache(backend)
    
    def cache_embedding(self, item_id, modality, embedding):
        """Cache embedding with modality type"""
        key = f"{modality}:{item_id}"
        self.cache.set(key, embedding, expire=86400)
    
    def get_embedding(self, item_id, modality):
        """Retrieve cached embedding"""
        key = f"{modality}:{item_id}"
        return self.cache.get(key)
```

## Best Practices

1. **Choose right embeddings** - different modalities need different encoders
2. **Preserve layout/context** - maintain structure during retrieval
3. **Cache aggressively** - multimodal embeddings are expensive to compute
4. **Combine modalities wisely** - not all queries benefit from all modalities
5. **Handle missing modalities** - gracefully fall back when content unavailable

## Conclusion

Multimodal RAG extends retrieval to rich content beyond text. Success requires appropriate embeddings for each modality, careful context preservation, and thoughtful fusion strategies. Start with text + images, then add other modalities as needed.
