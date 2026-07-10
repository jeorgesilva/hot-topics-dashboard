"""Shared multilingual sentence-embedding model (Fase 4).

Extracted from src/scoring/framing.py so src/scoring/contradiction.py can
reuse the same loaded model for NLI candidate-pair selection without a
circular import between the two scoring modules.
"""
from __future__ import annotations

_model = None

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def get_model():
    """Lazy-load and return the shared SentenceTransformer instance."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model
