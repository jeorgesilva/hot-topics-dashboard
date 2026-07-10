"""Tests for src/scoring/weights.py — the single source of truth for score weights."""
from __future__ import annotations

import pytest

from src.scoring.weights import (
    ARTICLE_RISK_WEIGHTS,
    COMPOSITE_RISK_WEIGHTS,
    WEIGHT_GROUPS,
    validate_weight_groups,
)


@pytest.mark.parametrize("name,group", WEIGHT_GROUPS.items())
def test_weight_group_sums_to_one(name, group):
    assert sum(group.values()) == pytest.approx(1.0, abs=1e-9), (
        f"{name} weights must sum to 1.0"
    )


def test_article_risk_weights_has_expected_keys():
    assert set(ARTICLE_RISK_WEIGHTS) == {
        "source_distrust",
        "sentiment_extremity",
        "sensationalism",
        "attribution_vagueness",
    }


def test_composite_risk_weights_has_expected_keys():
    assert set(COMPOSITE_RISK_WEIGHTS) == {
        "avg_article_risk",
        "framing_inconsistency",
        "fact_inconsistency",
    }


def test_validate_weight_groups_raises_on_bad_sum(monkeypatch):
    monkeypatch.setitem(WEIGHT_GROUPS, "BROKEN", {"a": 0.5, "b": 0.4})
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_weight_groups()
