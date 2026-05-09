# Reranking and Context Compression

Bi-encoders — used at retrieval time — embed queries and documents
independently, which is fast but limits precision. A cross-encoder takes the
query and a candidate document together and outputs a single relevance score.
Because it can attend jointly over the pair, a cross-encoder is significantly
more accurate than the bi-encoder, at the cost of being too slow for the full
index. The standard pattern is to retrieve top-k with a bi-encoder (k = 50 to
200) and then rerank with a cross-encoder down to k' = 5 to 10.

## Context Compression

Even after reranking, the selected chunks may contain irrelevant sentences
that waste the context window and distract the generator. Context compression
reduces each chunk to the spans most relevant to the query. Common techniques
include extractive sentence selection, LLM-based summarization conditioned on
the query, and LongLLMLingua-style token pruning.

## Practical Defaults

A reasonable starting configuration is bi-encoder top-k of 50,
cross-encoder rerank top-k of 8, and then compression to roughly 2,000
tokens of grounded context before generation.
