# RAG Evaluation Metrics

Four metrics from the RAGAS framework are the working baseline for most
production RAG systems:

- **Faithfulness**: of the claims in the generated answer, the fraction that
  are entailed by the retrieved context. Faithfulness is the canonical
  hallucination signal.
- **Answer Relevance**: how well the generated answer actually addresses the
  user's question, independent of correctness. Computed by generating
  candidate questions from the answer and measuring their similarity to the
  original.
- **Context Precision**: whether relevant context chunks are ranked near the
  top of the retrieved list. A precision-at-k style signal for the retriever.
- **Context Recall**: whether the information needed to answer the question
  is actually present in the retrieved context, evaluated against a ground
  truth answer.

## LLM-as-Judge

LLM-as-judge pipelines use a strong model (e.g., Claude Opus) to score
answers against a rubric. Calibration matters: run the judge against a small
human-labeled set to verify agreement before trusting aggregate scores.

## Regression Testing

A prompt or retrieval change is a regression when an aggregate metric drops
by more than a configured threshold (commonly 0.05) against the previous
committed run on the golden dataset.
