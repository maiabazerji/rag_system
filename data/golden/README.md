# Golden evaluation datasets

Each dataset is a JSON Lines file with one question per line.
`app.eval.metrics.load_dataset` loads it by name, without the `.jsonl` suffix:

```bash
python scripts/measure_retrieval.py --dataset golden_v3            # retrieval only, free
python scripts/run_eval.py --dataset golden_v3 --strategy classic  # retrieval + LLM judge
```

## Datasets

| File | Questions | Language | Corpus | Status |
|---|---:|---|---|---|
| `golden_v1.jsonl` | 34 | English | `data/docs` | Frozen baseline: one expected source per question (two questions have two) |
| `golden_v2.jsonl` | 34 | English | `data/docs` | Frozen: v1's questions with every document that covers each answer; includes 2 refusal cases (`note`) |
| `golden_fr_v1.jsonl` | 16 | French | `data/docs` (English) | Frozen: cross-lingual retrieval |
| `golden_v3.jsonl` | 54 | English | `data/docs` | Categorised set with evidence, see below |
| `golden_fr_business_v1.jsonl` | 40 | French | `data/demo_fr_business/docs` | Categorised set over the synthetic French business corpus. Ingest that corpus into its own collection first (see `data/demo_fr_business/README.md`) |

**`golden_v1`, `golden_v2` and `golden_fr_v1` are kept unchanged** so that scores stay
comparable with earlier runs. The new fields were not backfilled into them. New
categories and metadata go into new files, never into existing ones.

## Fields

Required by the harness:

| Field | Type | Meaning |
|---|---|---|
| `question` | string | The question sent to the pipeline |
| `ideal_answer` | string | Reference answer for the judge's answer-correctness score |
| `expected_sources` | list[string] | Document filenames that should be retrieved. Matching is by lowercased basename (`app/eval/retrieval.py`). An empty list means retrieval is not scored |

Optional, added in `golden_v3` and `golden_fr_business_v1` (the harness ignores unknown
fields):

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable identifier (`v3-001`, `frbiz-001`, ...) |
| `relevant_doc_ids` | list[string] | Same identifiers as `expected_sources` (bare filenames), kept equal to it |
| `question_type` | enum | One of the eight categories below |
| `difficulty` | `easy` \| `medium` \| `hard` | Author's estimate |
| `expected_behavior` | `answer` \| `refuse` | `refuse` when the corpus does not contain the answer |
| `notes` | string | Each supporting document with a verbatim snippet of 25 words or fewer (`doc.md: "..."`), plus any reviewer note |
| `evidence` | list[{doc, quote}] | The same snippets in machine-checkable form |

`golden_v2` also uses a free-text `note` field on its refusal questions.

## Categories

| `question_type` | Definition | Expected behaviour |
|---|---|---|
| `single_hop` | The answer is in one passage of one document | answer |
| `multi_hop` | The answer requires chaining facts from 2 or more documents | answer |
| `comparison` | Contrast two items, from one document or from several | answer |
| `aggregation` | Collect, count or sum values spread over a table or several documents | answer |
| `ambiguous` | The question is underspecified, or the corpus holds conflicting values. The ideal answer names the options or the conflict | answer (and clarify) |
| `unanswerable` | The answer is not in the corpus. `relevant_doc_ids` is `[]` | refuse |
| `citation_sensitive` | The answer depends on one exact figure or term in one document | answer |
| `adversarial` | The question has a false or misleading premise. The ideal answer corrects it and cites the document | answer (correct the premise) |

### Counts

| Category | golden_v3 | golden_fr_business_v1 |
|---|---:|---:|
| single_hop | 7 | 5 |
| multi_hop | 7 | 5 |
| comparison | 7 | 5 |
| aggregation | 6 | 5 |
| ambiguous | 6 | 5 |
| unanswerable | 7 | 5 |
| citation_sensitive | 7 | 5 |
| adversarial | 7 | 5 |
| **Total** | **54** | **40** |
| Difficulty (easy / medium / hard) | 14 / 31 / 9 | 13 / 23 / 4 |

Unanswerable questions have no expected sources, so `score_retrieval` skips them. Use
the answer-level metrics (refusal, judge) for them. When you report retrieval
metrics by category, filter on `question_type`.

## How the answers were verified

- **golden_v3.** Every `ideal_answer` was written from the text of the cited documents
  in `data/docs`. Every `evidence.quote` was checked by script to be a verbatim substring
  of the named document, after collapsing whitespace, because the source files wrap
  lines. Some `ambiguous` questions deliberately rely on real inconsistencies in the
  corpus. For example, BGE-Large is given 768 dimensions in
  `dense_embeddings_vector_search.md` but 1024 in `embedding_models_guide.md`, and
  text-embedding-3-large is priced at $0.02 in one document and $0.13 per million
  tokens in another.
- **Unanswerable questions.** For each one, the key terms were searched
  case-insensitively in `data/docs` **and** in the top-level `README.md`, `EvalRAG.md`
  and `LEARN.md`, because `scripts/ingest.py` ingests those files by default. The
  search terms are recorded in `notes`.
- **golden_fr_business_v1.** Every amount was recomputed in Python from the documents:
  invoice lines, discounts, VAT, penalties and register sums. Evidence quotes were
  checked verbatim in the same way.
- **Validation checks.** A validation pass confirms that every line parses, that all
  required fields are present, that each category has enough questions, that every
  `relevant_doc_ids` entry exists as a file, that `expected_sources` equals
  `relevant_doc_ids`, and that every evidence quote occurs in its document.
