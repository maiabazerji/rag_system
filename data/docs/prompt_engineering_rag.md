# Prompt Engineering for RAG: Maximizing Generation Quality

## Introduction

Prompt engineering significantly impacts RAG quality. This guide covers techniques for designing prompts that effectively use retrieved context, reduce hallucination, and improve generation accuracy.

## 1. RAG Prompt Architecture

### Basic RAG Prompt Structure

```python
class RAGPromptTemplate:
    """Standard RAG prompt template"""
    
    SYSTEM_PROMPT = """You are a helpful AI assistant. 
    Answer questions based on the provided context.
    If the context doesn't contain relevant information, say so.
    Never make up information."""
    
    CONTEXT_PROMPT = """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    @classmethod
    def format_prompt(cls, context: str, question: str) -> str:
        return cls.CONTEXT_PROMPT.format(
            context=context,
            question=question
        )
```

### System vs User Prompts

**System Prompt (Instructions):**
```
- Defines assistant role and behavior
- Sets ground rules for response
- Specifies format expectations
- Typically fixed for entire session
```

**User Prompt (Specific Request):**
```
- Contains actual question/task
- Includes retrieved context
- Variable per interaction
```

## 2. Context Presentation Techniques

### 1. Direct Context Insertion

```python
def format_direct_context(retrieved_docs, question):
    """Simple concatenation of context"""
    context_text = "\n\n".join(
        [f"[Source {i+1}] {doc['text']}" 
         for i, doc in enumerate(retrieved_docs)]
    )
    
    prompt = f"""Answer the following question using the provided context.

Context:
{context_text}

Question: {question}

Answer:"""
    
    return prompt
```

**Advantages:**
- Simple, straightforward
- Full context available
- Good for short contexts

**Disadvantages:**
- Works poorly with many documents
- Context loss with long passages
- May overwhelm models

### 2. Hierarchical Context Presentation

```python
def format_hierarchical_context(retrieved_docs, question):
    """Present context with structure"""
    sections = []
    
    # Primary documents (high relevance)
    primary = [d for d in retrieved_docs if d['score'] > 0.8]
    if primary:
        sections.append("High Relevance Sources:")
        for doc in primary[:3]:
            sections.append(f"- {doc['text'][:200]}...")
    
    # Secondary documents
    secondary = [d for d in retrieved_docs if 0.6 < d['score'] <= 0.8]
    if secondary:
        sections.append("\nAdditional Context:")
        for doc in secondary[:2]:
            sections.append(f"- {doc['text'][:150]}...")
    
    context_text = "\n".join(sections)
    
    prompt = f"""Answer based on this information:

{context_text}

Question: {question}

Answer:"""
    
    return prompt
```

### 3. Question-Focused Context

```python
def format_question_focused_context(retrieved_docs, question):
    """Highlight query-relevant parts"""
    from sentence_transformers import util
    
    # Find most query-relevant sentences
    relevant_sentences = []
    
    for doc in retrieved_docs[:3]:
        sentences = doc['text'].split('.')
        
        # Score sentences by relevance
        query_embedding = get_embedding(question)
        
        for sentence in sentences:
            sent_embedding = get_embedding(sentence)
            score = util.pytorch_cos_sim(query_embedding, sent_embedding)
            
            if score > 0.6:
                relevant_sentences.append((sentence, score))
    
    # Sort by relevance and take top
    top_sentences = sorted(relevant_sentences, key=lambda x: x[1], reverse=True)[:5]
    
    context_text = ". ".join([s[0] for s in top_sentences])
    
    prompt = f"""Using these key points:

{context_text}

Answer: {question}

Response:"""
    
    return prompt
```

## 3. Few-Shot Prompting with RAG

```python
class FewShotRAGPrompt:
    """Few-shot examples improve generation quality"""
    
    EXAMPLES = [
        {
            "question": "What is RAG?",
            "context": "RAG combines retrieval with generation. It retrieves relevant documents first, then uses them to generate answers.",
            "answer": "RAG (Retrieval-Augmented Generation) is a technique that retrieves relevant documents from a knowledge base before generating responses. This helps ensure answers are grounded in actual information."
        },
        {
            "question": "How does embedding work?",
            "context": "Embeddings convert text to vectors capturing semantic meaning. Modern embeddings use transformer models trained on large datasets.",
            "answer": "Embeddings transform text into numerical vectors that capture semantic meaning. Advanced embedding models like BERT use transformer architectures trained on large text corpora to create meaningful representations."
        }
    ]
    
    @classmethod
    def format_few_shot(cls, context, question):
        """Create few-shot prompt"""
        examples_text = ""
        
        for i, ex in enumerate(cls.EXAMPLES, 1):
            examples_text += f"""Example {i}:
Context: {ex['context']}
Question: {ex['question']}
Answer: {ex['answer']}

"""
        
        prompt = f"""{examples_text}Now answer this:

Context: {context}
Question: {question}
Answer:"""
        
        return prompt
```

## 4. Chain-of-Thought Prompting

```python
def format_cot_prompt(context, question):
    """Chain-of-thought promotes reasoning"""
    
    prompt = f"""Answer the following question step by step.

Context:
{context}

Question: {question}

Let's think through this step by step:
1. First, identify the key information in the context relevant to this question.
2. Then, consider what we know from the context.
3. Finally, construct the answer based on this reasoning.

Answer:"""
    
    return prompt
```

## 5. Instruction-Based Prompting

```python
class InstructionBasedPrompt:
    """Explicit instructions improve compliance"""
    
    @staticmethod
    def format_with_instructions(context, question, instructions=None):
        default_instructions = [
            "Base your answer only on the provided context",
            "If information is not in the context, say 'Information not available'",
            "Quote relevant passages when applicable",
            "Provide clear, concise answers",
            "Use the same language as the question"
        ]
        
        instructions = instructions or default_instructions
        
        instructions_text = "\n".join(
            [f"{i+1}. {inst}" for i, inst in enumerate(instructions)]
        )
        
        prompt = f"""Answer the question following these instructions:

{instructions_text}

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
```

## 6. Context Management

### Handling Multiple Documents

```python
class ContextManager:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens
    
    def select_relevant_documents(self, docs, question, encoder):
        """Select most relevant docs that fit in context window"""
        question_embedding = encoder.encode(question)
        
        # Score and sort by relevance
        scored_docs = []
        for doc in docs:
            doc_embedding = encoder.encode(doc['text'][:500])
            score = np.dot(question_embedding, doc_embedding)
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Select docs up to token limit
        selected = []
        token_count = 0
        
        for doc, score in scored_docs:
            doc_tokens = len(doc['text'].split())
            
            if token_count + doc_tokens < self.max_tokens:
                selected.append(doc)
                token_count += doc_tokens
            else:
                break
        
        return selected
    
    def truncate_context(self, docs, max_tokens):
        """Truncate context to fit token limit"""
        context_parts = []
        token_count = 0
        
        for doc in docs:
            words = doc['text'].split()
            
            for i, word in enumerate(words):
                if token_count < max_tokens:
                    context_parts.append(word)
                    token_count += 1
                else:
                    break
            
            if token_count >= max_tokens:
                break
        
        return " ".join(context_parts) + "..."
```

### Lost in the Middle Problem

Research shows models perform worse when relevant information appears in the middle of long contexts.

```python
def reorder_context_for_importance(docs):
    """Place most important docs at start and end"""
    if len(docs) <= 2:
        return docs
    
    # Reorder: most relevant first and last
    docs_by_importance = sorted(
        enumerate(docs),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    reordered = []
    
    # Add top documents at start
    for i in range(len(docs_by_importance) // 2):
        reordered.append(docs_by_importance[i][1])
    
    # Add remaining at end (reversed to put highest at very end)
    for i in range(len(docs_by_importance) // 2, len(docs_by_importance)):
        reordered.insert(-1 if reordered else 0, docs_by_importance[i][1])
    
    return reordered
```

## 7. Reducing Hallucination

### Technique 1: Explicit Grounding Instructions

```python
def format_grounded_prompt(context, question):
    """Explicit instructions to stay grounded in context"""
    
    prompt = f"""Answer ONLY using information from the provided context.

IMPORTANT RULES:
- Only answer if the answer is explicitly in the context
- If you cannot find the answer in the context, respond: "I cannot find this information in the provided context"
- Do NOT make up or infer information
- Do NOT use your training data if it contradicts the context
- Quote the context when relevant

Context:
{context}

Question: {question}

Answer:"""
    
    return prompt
```

### Technique 2: Attribution Prompt

```python
def format_attribution_prompt(retrieved_docs, question):
    """Prompt with source attribution requirement"""
    
    context_with_sources = ""
    for i, doc in enumerate(retrieved_docs):
        context_with_sources += f"[Source {i+1}] {doc['text']}\n"
    
    prompt = f"""Answer the question and cite your sources.

{context_with_sources}

Question: {question}

Answer (include source citations like [Source 1]):"""
    
    return prompt
```

### Technique 3: Uncertainty Expression

```python
def format_uncertainty_aware_prompt(context, question):
    """Encourage model to express uncertainty"""
    
    prompt = f"""Answer the question. If you're not certain, express your confidence level.

Context:
{context}

Question: {question}

Answer: (feel free to express uncertainty like "I'm about 70% confident that..." or "The context doesn't fully address...")"""
    
    return prompt
```

## 8. Format and Structure Prompts

```python
class StructuredPrompt:
    """Request structured outputs"""
    
    @staticmethod
    def format_qa_structured(context, question):
        """Request Q&A format output"""
        return f"""Answer the question in this format:

Context:
{context}

Question: {question}

Q: {question}
A: [Your answer here]
Confidence: [High/Medium/Low]
Sources: [Cite relevant parts of context]"""
    
    @staticmethod
    def format_json_structured(context, question):
        """Request JSON output"""
        return f"""Answer in JSON format.

Context:
{context}

Question: {question}

{{
  "answer": "your answer here",
  "confidence": 0-1,
  "sources": ["source 1", "source 2"],
  "explanation": "why you answered this way"
}}"""
    
    @staticmethod
    def format_markdown_structured(context, question):
        """Request Markdown format"""
        return f"""Answer in Markdown format.

Context:
{context}

Question: {question}

## Answer
[Your answer]

### Key Points
- Point 1
- Point 2

### Sources
- [Source 1]
- [Source 2]"""
```

## 9. Temperature and Generation Parameters

```python
class RAGGenerationConfig:
    """Configuration for RAG generation"""
    
    # Configuration profiles
    PROFILES = {
        'factual': {
            'temperature': 0.3,  # Lower = more consistent
            'top_p': 0.9,
            'max_tokens': 500,
            'description': 'For fact-based QA, minimize hallucination'
        },
        'creative': {
            'temperature': 0.7,
            'top_p': 0.95,
            'max_tokens': 1000,
            'description': 'For creative tasks, allow diversity'
        },
        'balanced': {
            'temperature': 0.5,
            'top_p': 0.92,
            'max_tokens': 750,
            'description': 'Balanced between consistency and quality'
        }
    }
    
    @classmethod
    def get_config(cls, task_type='factual'):
        """Get generation config for task type"""
        return cls.PROFILES.get(task_type, cls.PROFILES['balanced'])
```

## 10. Evaluation and Iteration

```python
class PromptEvaluator:
    """Evaluate prompt effectiveness"""
    
    @staticmethod
    def evaluate_prompt(prompt_template, test_cases, llm):
        """Evaluate prompt on test cases"""
        results = {
            'accuracy': 0,
            'hallucination_rate': 0,
            'grounding_score': 0,
            'fluency': 0
        }
        
        for test_case in test_cases:
            context = test_case['context']
            question = test_case['question']
            expected = test_case['expected_answer']
            
            prompt = prompt_template(context, question)
            response = llm.generate(prompt)
            
            # Score response
            scores = {
                'exact_match': 1 if response == expected else 0,
                'hallucination': 1 if PromptEvaluator._has_hallucination(
                    response, context
                ) else 0,
                'grounding': PromptEvaluator._grounding_score(
                    response, context
                )
            }
        
        return results
    
    @staticmethod
    def _has_hallucination(response, context):
        """Check if response contains information not in context"""
        # Simplified check - in practice use NLI model
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        novel_words = response_words - context_words
        hallucination_threshold = 0.3
        
        return len(novel_words) / len(response_words) > hallucination_threshold
    
    @staticmethod
    def _grounding_score(response, context):
        """Score how well response is grounded in context"""
        # Compute overlap between response and context
        response_phrases = set(response.lower().split())
        context_phrases = set(context.lower().split())
        
        overlap = response_phrases & context_phrases
        
        return len(overlap) / len(response_phrases) if response_phrases else 0
```

## Best Practices Summary

1. **Be explicit about using context** - tell model to base answer only on provided context
2. **Use hierarchical context** - order by importance, avoid "lost in middle"
3. **Encourage attribution** - ask for source citations
4. **Provide examples** - few-shot prompts improve quality
5. **Use lower temperature** - 0.3-0.5 for factual tasks reduces hallucination
6. **Test different formats** - JSON, markdown, structured output each have merits
7. **Monitor grounding** - measure how well responses stay grounded in context

## Conclusion

Prompt engineering is as critical as retrieval in RAG systems. Well-designed prompts significantly reduce hallucination, improve accuracy, and increase user trust. Start with explicit grounding instructions and iterate based on your specific use case and model.
