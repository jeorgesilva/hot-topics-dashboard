"""Tests for src/scoring/attribution.py — no external dependencies."""
from __future__ import annotations

import pytest

from src.scoring.attribution import score_attribution_vagueness


# ---------------------------------------------------------------------------
# Boundary / trivial inputs
# ---------------------------------------------------------------------------

class TestBoundaryInputs:
    def test_empty_string_returns_zero(self):
        assert score_attribution_vagueness("") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert score_attribution_vagueness("   \t\n  ") == 0.0

    def test_returns_float(self):
        assert isinstance(score_attribution_vagueness("some text"), float)

    def test_score_bounded_zero_to_one(self):
        extreme = (
            "Experts say reportedly it is claimed according to sources "
            "many believe rumors suggest studies show a source familiar with"
        )
        score = score_attribution_vagueness(extreme)
        assert 0.0 <= score <= 1.0

    def test_factual_named_source_returns_zero(self):
        # Named, verifiable attribution — no vague patterns
        score = score_attribution_vagueness(
            "The WHO Director-General Tedros Adhanom Ghebreyesus said "
            "the vaccine rollout is on track."
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# Anonymous plural agents
# ---------------------------------------------------------------------------

class TestAnonymousAgents:
    def test_experts_say_detected(self):
        assert score_attribution_vagueness("Experts say the vaccine is dangerous.") > 0.0

    def test_officials_warn_detected(self):
        assert score_attribution_vagueness("Officials warn of a new threat.") > 0.0

    def test_sources_claim_detected(self):
        assert score_attribution_vagueness("Sources claim the president is resigning.") > 0.0

    def test_researchers_suggest_detected(self):
        assert score_attribution_vagueness("Researchers suggest the treatment works.") > 0.0

    def test_insiders_report_detected(self):
        assert score_attribution_vagueness("Insiders report the deal is imminent.") > 0.0


# ---------------------------------------------------------------------------
# Adverbial hedges
# ---------------------------------------------------------------------------

class TestAdverbialHedges:
    def test_reportedly_detected(self):
        assert score_attribution_vagueness("The CEO reportedly resigned last night.") > 0.0

    def test_allegedly_detected(self):
        assert score_attribution_vagueness("He allegedly stole millions from investors.") > 0.0

    def test_purportedly_detected(self):
        assert score_attribution_vagueness("The document is purportedly authentic.") > 0.0

    def test_supposedly_detected(self):
        assert score_attribution_vagueness("The footage was supposedly recorded in secret.") > 0.0

    def test_apparently_detected(self):
        assert score_attribution_vagueness("Apparently the government knew all along.") > 0.0


# ---------------------------------------------------------------------------
# Agentless passives
# ---------------------------------------------------------------------------

class TestAgentlessPassives:
    def test_it_is_claimed_detected(self):
        assert score_attribution_vagueness("It is claimed that the election was rigged.") > 0.0

    def test_it_was_reported_detected(self):
        assert score_attribution_vagueness("It was reported that the minister fled.") > 0.0

    def test_it_has_been_alleged_detected(self):
        assert score_attribution_vagueness("It has been alleged that funds were misused.") > 0.0

    def test_it_is_believed_detected(self):
        assert score_attribution_vagueness("It is believed that millions are at risk.") > 0.0


# ---------------------------------------------------------------------------
# "According to" with unnamed sources
# ---------------------------------------------------------------------------

class TestAccordingToUnnamed:
    def test_according_to_sources_detected(self):
        assert score_attribution_vagueness("According to sources, talks broke down.") > 0.0

    def test_according_to_officials_detected(self):
        assert score_attribution_vagueness("According to officials, the plan is ready.") > 0.0

    def test_according_to_anonymous_detected(self):
        assert score_attribution_vagueness("According to anonymous insiders, layoffs are coming.") > 0.0

    def test_named_source_not_flagged(self):
        # "According to Reuters" is a named source — should NOT be caught
        score = score_attribution_vagueness("According to Reuters, the talks concluded.")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Collective / speculative framing
# ---------------------------------------------------------------------------

class TestCollectiveSpeculative:
    def test_some_say_detected(self):
        assert score_attribution_vagueness("Some say the government is hiding the truth.") > 0.0

    def test_many_believe_detected(self):
        assert score_attribution_vagueness("Many believe the pandemic was engineered.") > 0.0

    def test_critics_argue_detected(self):
        assert score_attribution_vagueness("Critics argue the policy is unconstitutional.") > 0.0

    def test_skeptics_claim_detected(self):
        assert score_attribution_vagueness("Skeptics claim the data was manipulated.") > 0.0


# ---------------------------------------------------------------------------
# Rumour / speculation patterns
# ---------------------------------------------------------------------------

class TestRumourSpeculation:
    def test_rumors_suggest_detected(self):
        assert score_attribution_vagueness("Rumors suggest the CEO will be fired.") > 0.0

    def test_speculation_indicates_detected(self):
        assert score_attribution_vagueness("Speculation indicates a hidden agenda.") > 0.0


# ---------------------------------------------------------------------------
# Unattributed evidence
# ---------------------------------------------------------------------------

class TestUnattributedEvidence:
    def test_studies_show_detected(self):
        assert score_attribution_vagueness("Studies show the drug causes cancer.") > 0.0

    def test_research_suggests_detected(self):
        assert score_attribution_vagueness("Research suggests the vaccine is ineffective.") > 0.0

    def test_data_confirms_detected(self):
        assert score_attribution_vagueness("Data confirms the conspiracy theory.") > 0.0


# ---------------------------------------------------------------------------
# Anonymity constructions
# ---------------------------------------------------------------------------

class TestAnonymityConstructions:
    def test_source_familiar_with_detected(self):
        assert score_attribution_vagueness(
            "A source familiar with the matter confirmed the deal."
        ) > 0.0

    def test_speaking_on_condition_of_anonymity_detected(self):
        assert score_attribution_vagueness(
            "Someone speaking on condition of anonymity revealed the plan."
        ) > 0.0


# ---------------------------------------------------------------------------
# Saturation and multi-hit scoring
# ---------------------------------------------------------------------------

class TestSaturation:
    def test_three_hits_saturates_at_one(self):
        # Three distinct patterns → score == 1.0
        text = (
            "Experts say the vaccine is dangerous. "
            "It is claimed that millions are at risk. "
            "According to sources, the government is hiding the truth."
        )
        assert score_attribution_vagueness(text) == 1.0

    def test_more_than_three_hits_capped_at_one(self):
        text = (
            "Experts say it is claimed that reportedly according to sources "
            "many believe rumors suggest studies show."
        )
        assert score_attribution_vagueness(text) == 1.0

    def test_two_hits_below_one(self):
        text = (
            "Officials warn of a new threat. "
            "Allegedly the evidence was destroyed."
        )
        score = score_attribution_vagueness(text)
        assert 0.0 < score < 1.0

    def test_one_hit_scores_one_third(self):
        score = score_attribution_vagueness("Experts say this is dangerous.")
        assert abs(score - round(1 / 3, 4)) < 1e-4

    def test_case_insensitive(self):
        lower = score_attribution_vagueness("experts say this is dangerous.")
        upper = score_attribution_vagueness("EXPERTS SAY THIS IS DANGEROUS.")
        assert lower == upper
