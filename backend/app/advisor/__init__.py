"""Strategy advisor: recommend classic, graph or agentic RAG for a described project.

- ``profile``: turn a free-text project description into a ``ProjectProfile``
  (Claude extraction with a keyword-heuristic fallback; explicit overrides win).
- ``scoring``: deterministic, explainable scoring of the three strategies.
- ``validate``: run a customer's own questions through the recommended
  strategies and report a measured scorecard.
"""
