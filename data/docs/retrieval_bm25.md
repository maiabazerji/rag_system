# BM25 and Hybrid Retrieval

BM25 (Best Match 25) is a bag-of-words ranking function from the probabilistic
retrieval family. It scores a document by the sum of its query-term weights,
where each weight is an IDF factor multiplied by a saturating term-frequency
factor that accounts for document length.

The canonical BM25 parameters are `k1` (term frequency saturation, typically
1.2–2.0) and `b` (length normalization, typically 0.75). Larger `k1` lets
term frequency continue to increase the score; larger `b` penalizes longer
documents more.

## Dense vs Sparse

Dense retrievers embed text into a continuous vector space and retrieve by
cosine or dot-product similarity. They capture semantic similarity but can
miss exact keyword matches, especially for rare identifiers, product codes,
or proper nouns. Sparse retrievers like BM25 are strong on exact lexical
overlap but weak on paraphrases.

## Hybrid Retrieval

Hybrid retrieval fuses dense and sparse scores. A common approach is
Reciprocal Rank Fusion (RRF), which sums `1 / (k + rank_i)` across rankers.
RRF is parameter-light and robust because it does not require score
calibration between heterogeneous retrievers.
