"""Shared NLI (natural language inference) pipeline (Fase 4 / Fase 5).

Extracted from src/scoring/contradiction.py, mirroring the src/nlp/embeddings.py
precedent, so src/scoring/contradiction.py and src/scoring/claim_verifier.py
can reuse the same loaded model instead of holding two copies of a
multilingual DeBERTa model in memory.
"""
from __future__ import annotations

_pipeline = None

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def get_pipeline():
    """Lazy-load and return the shared NLI text-classification pipeline."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model=NLI_MODEL,
            top_k=None,
            tokenizer_kwargs={"truncation": True, "max_length": 256},
        )
    return _pipeline
