# Observability: Weights & Biases for LLM Apps

Weights & Biases (W&B) provides experiment tracking originally for ML
training, but the DeepLearning.AI "Evaluating and Debugging Generative AI"
course adapts it to LLM applications.

## Core Primitives Used in LLM Work

- **Runs**: one unit of work — an evaluation pass, a prompt variant sweep, a
  benchmark. Each run has a config (model, prompt version, top-k, etc.) and
  a set of logged metrics.
- **Tables**: row-oriented logs with arbitrary columns. For LLM apps, a row
  is typically a single (question, retrieved_context, answer, judge_scores)
  tuple. Tables make run-to-run comparison trivial in the W&B UI.
- **Artifacts**: versioned binary or text blobs. Useful for committing the
  exact prompt template and golden dataset snapshot used by a run, so the
  run is fully reproducible.
- **Sweeps**: hyperparameter searches. For RAG, the natural axes are
  retrieval top-k, rerank top-k, chunk size, chunk overlap, and prompt
  version.

## Relation to Langfuse

Langfuse focuses on request-level tracing in production (spans, costs,
latencies). W&B focuses on run-level evaluation and experiment history.
They are complementary: Langfuse answers "what happened in this user's
request?"; W&B answers "did this change improve the aggregate metrics?".
