"""Tests for src/orchestrator.py — unified pipeline entry point."""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator import run_pipeline


def _make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "target_topics": 2,
        "articles_per_topic": 5,
        "broad_search": False,
        "no_newsapi": True,
        "db_path": "data/test_orch.db",
        "log_level": "WARNING",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _patched_pipeline(extra_fetchone_side_effects=None):
    """Return a context manager stacking all necessary patches."""
    result_conn = MagicMock()
    side_effects = extra_fetchone_side_effects or [(3,), (2,)]
    result_conn.execute.return_value.fetchone.side_effect = side_effects

    patches = [
        patch("src.orchestrator.sqlite3.connect", return_value=result_conn),
        patch("src.orchestrator.run_nlp.load_nlp_models"),
        patch("src.orchestrator.run_all.run_pipeline"),
        patch("src.orchestrator.run_nlp.run_nlp_pipeline"),
        patch("src.orchestrator.init_db", return_value=MagicMock()),
        patch("src.orchestrator.compute_scores.score_all_topics"),
    ]
    return patches


def test_run_pipeline_returns_expected_keys():
    """run_pipeline returns a dict with all required keys and correct structure."""
    patches = _patched_pipeline()
    mocks = [p.start() for p in patches]
    try:
        result = run_pipeline(_make_args())
    finally:
        for p in patches:
            p.stop()

    assert set(result.keys()) == {
        "topics_collected",
        "topics_scored",
        "elapsed_total_s",
        "elapsed_per_step",
    }
    assert set(result["elapsed_per_step"].keys()) == {"collect", "nlp", "scores"}
    assert result["topics_collected"] == 3
    assert result["topics_scored"] == 2
    assert result["elapsed_total_s"] >= 0.0
    for step_time in result["elapsed_per_step"].values():
        assert step_time >= 0.0


def test_models_loaded_once():
    """load_nlp_models is called exactly once per run_pipeline invocation."""
    patches = _patched_pipeline(extra_fetchone_side_effects=[(1,), (1,), (1,), (1,)])
    mocks = [p.start() for p in patches]
    mock_load = mocks[1]  # second patch is load_nlp_models

    try:
        run_pipeline(_make_args())
        run_pipeline(_make_args())
    finally:
        for p in patches:
            p.stop()

    # Orchestrator calls load_nlp_models once per run_pipeline call.
    # The actual caching (second call is a no-op) lives inside load_nlp_models itself.
    assert mock_load.call_count == 2


def test_standalone_scripts_unaffected():
    """Importing orchestrator does not break the individual module interfaces."""
    from src.scoring import compute_scores, run_nlp  # noqa: F401
    from src.scrapers import run_all  # noqa: F401

    assert callable(run_all.run_pipeline)
    assert callable(run_nlp.run_nlp_pipeline)
    assert callable(run_nlp.load_nlp_models)
    assert callable(compute_scores.score_all_topics)
    assert callable(compute_scores.main)
    assert callable(run_nlp.main)
    assert callable(run_all.main)
