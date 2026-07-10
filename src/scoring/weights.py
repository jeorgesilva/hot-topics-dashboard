"""Single source of truth for scoring formula weights.

`article_scorer.py` and `compute_scores.py` import their weights from this
module instead of hardcoding them. `scripts/render_docs.py` also reads these
values to (re)generate the "Scoring formulas" sections of README.md and
.claude/CLAUDE.md — do not hand-edit weight numbers in either doc.
"""
from __future__ import annotations

# article_risk = sum(ARTICLE_RISK_WEIGHTS[k] * signal[k]), per-article score
# stored in raw_items.article_risk_score. See src/scoring/article_scorer.py.
ARTICLE_RISK_WEIGHTS: dict[str, float] = {
    "source_distrust": 0.15,
    "sentiment_extremity": 0.30,
    "sensationalism": 0.30,
    "attribution_vagueness": 0.25,
}

# composite_risk = sum(COMPOSITE_RISK_WEIGHTS[k] * signal[k]), per-topic score
# stored in topic_scores.composite_risk. See src/scoring/compute_scores.py.
#
# fact_inconsistency (Fase 4): kept at 0.35 despite the metric itself being
# redefined from Jaccard entity-overlap to NLI contradiction proportion (see
# src/scoring/contradiction.py). The two metrics have different statistical
# profiles — Jaccard was near-1.0 on almost any topic with a mixed source
# tier (any two tiers cite somewhat different entities), so it acted as
# roughly-constant noise inflating composite_risk; the NLI proportion is 0.0
# on most topics and only nonzero when a genuine contradiction is detected,
# so it should behave as a rarer but much higher-precision signal. Whether
# 0.35 is still the right weight for that new distribution needs empirical
# validation against labelled topics (see notebooks/, per project testing
# convention — not decided here without real data).
COMPOSITE_RISK_WEIGHTS: dict[str, float] = {
    "avg_article_risk": 0.55,
    "framing_inconsistency": 0.10,
    "fact_inconsistency": 0.35,
}

WEIGHT_GROUPS: dict[str, dict[str, float]] = {
    "ARTICLE_RISK_WEIGHTS": ARTICLE_RISK_WEIGHTS,
    "COMPOSITE_RISK_WEIGHTS": COMPOSITE_RISK_WEIGHTS,
}


def validate_weight_groups() -> None:
    """Raise ValueError if any weight group does not sum to 1.0.

    Called at import time so a bad edit to this file fails fast, and also
    exercised directly by tests/test_scoring/test_weights.py for a clear
    CI failure message.
    """
    for name, group in WEIGHT_GROUPS.items():
        total = sum(group.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{name} must sum to 1.0, got {total}")


validate_weight_groups()
