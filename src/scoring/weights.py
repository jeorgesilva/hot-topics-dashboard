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

# linguistic_only_risk = sum(COMPOSITE_RISK_WEIGHTS[k] * signal[k]), per-topic
# score stored in topic_scores.linguistic_only_risk (Fase 7: the always-on
# signal, evidence-independent). See src/scoring/compute_scores.py.
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

# --- Fase 7: two-tier score --------------------------------------------------
#
# topic_scores now exposes two independent risk numbers instead of blending
# everything into one:
#   linguistic_only_risk   — the formula above (COMPOSITE_RISK_WEIGHTS),
#                             always computable once run_nlp has scored a topic.
#   evidence_grounded_risk — fraction of RAG-verified claims (Fase 5,
#                             claim_verifications) that were "refuted" among
#                             all claims with a definite verdict (supported ∪
#                             refuted — excludes not_enough_evidence). NULL
#                             when the topic has zero claims with a verdict.
#
# EVIDENCE_COVERAGE_THRESHOLD is the explicit, documented combination rule for
# topic_scores.composite_risk — the single number used for dashboard sorting
# and the misinformation-flag banner (see compute_scores.py::compute_overall_risk):
#   composite_risk = evidence_grounded_risk  if evidence_coverage > threshold
#                   = linguistic_only_risk    otherwise
# Below the threshold, too few claims have a definite verdict for the
# evidence-grounded number to be more trustworthy than the linguistic signal.
EVIDENCE_COVERAGE_THRESHOLD: float = 0.30

# overall_confidence bands (topic_scores.overall_confidence: 'high'/'medium'/'low').
# Derived in compute_scores.py::compute_overall_confidence as the average of:
#   - evidence_coverage itself (more claims with a definite verdict → more grounded)
#   - the topic's mean source_reliability.resolve_reliability() confidence
#     (Fase 6) across its article domains
CONFIDENCE_BAND_HIGH: float = 0.65
CONFIDENCE_BAND_MEDIUM: float = 0.35


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
