"""GDPR support: PII detection and redaction, erasure, export and retention.

Submodules are imported directly (``from app.privacy.pii import redact``)
rather than re-exported here. :mod:`app.logging_config` depends on
:mod:`app.privacy.pii`, and the erasure and retention modules reach into the
vector store, so importing them from this package's ``__init__`` would drag the
whole RAG stack into logging setup.

See ``docs/gdpr/README.md`` for what the system stores and how to operate it.
"""
