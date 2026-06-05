"""Tests for src/scoring/attribution.py — no external dependencies."""
from __future__ import annotations

import pytest

from src.scoring.attribution import _named_source_discount, score_attribution_vagueness


# ---------------------------------------------------------------------------
# Minimal spaCy-like mock objects for NER-enhanced tests
# ---------------------------------------------------------------------------

class _Ent:
    def __init__(self, label: str):
        self.label_ = label

class _Token:
    def __init__(self, lemma: str, pos: str = "VERB"):
        self.lemma_ = lemma
        self.pos_ = pos

class _Sent:
    def __init__(self, tokens: list[_Token], ents: list[_Ent]):
        self._tokens = tokens
        self.ents = ents

    def __iter__(self):
        return iter(self._tokens)

class _Doc:
    def __init__(self, sents: list[_Sent]):
        self._sents = sents

    @property
    def sents(self):
        return iter(self._sents)

    def has_annotation(self, attr: str) -> bool:
        return True  # mock always provides sentence boundaries via _sents

class _NLP:
    """Fake spaCy Language: always returns the same pre-built doc."""
    def __init__(self, doc: _Doc):
        self._doc = doc

    def __call__(self, text: str) -> _Doc:
        return self._doc


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


# ---------------------------------------------------------------------------
# _named_source_discount unit tests
# ---------------------------------------------------------------------------

class TestNamedSourceDiscount:
    def test_per_entity_with_attribution_verb_gives_discount(self):
        sent = _Sent([_Token("say")], [_Ent("PER")])
        assert _named_source_discount(_Doc([sent])) == 0.20

    def test_org_entity_also_gives_discount(self):
        sent = _Sent([_Token("report")], [_Ent("ORG")])
        assert _named_source_discount(_Doc([sent])) == 0.20

    def test_two_anchored_sentences_sum_discounts(self):
        sent = _Sent([_Token("say")], [_Ent("PER")])
        assert _named_source_discount(_Doc([sent, sent])) == 0.40

    def test_discount_capped_at_max(self):
        sent = _Sent([_Token("say")], [_Ent("PER")])
        # three anchored sentences → would be 0.60 without cap
        assert _named_source_discount(_Doc([sent, sent, sent])) == 0.40

    def test_no_named_entity_no_discount(self):
        sent = _Sent([_Token("say")], [])
        assert _named_source_discount(_Doc([sent])) == 0.0

    def test_named_entity_but_no_attribution_verb_no_discount(self):
        # PER entity present but the verb is not an attribution verb
        sent = _Sent([_Token("run", pos="VERB")], [_Ent("PER")])
        assert _named_source_discount(_Doc([sent])) == 0.0

    def test_non_verb_token_ignored(self):
        # "say" tagged as NOUN should not match
        sent = _Sent([_Token("say", pos="NOUN")], [_Ent("PER")])
        assert _named_source_discount(_Doc([sent])) == 0.0

    def test_empty_doc_returns_zero(self):
        assert _named_source_discount(_Doc([])) == 0.0

    def test_loc_entity_gives_no_discount(self):
        # Only PER and ORG are source-anchor signals, not LOC
        sent = _Sent([_Token("say")], [_Ent("LOC")])
        assert _named_source_discount(_Doc([sent])) == 0.0


# ---------------------------------------------------------------------------
# score_attribution_vagueness — NER-enhanced path
# ---------------------------------------------------------------------------

class TestNEREnhancedPath:
    def test_named_source_reduces_score(self):
        text = "Experten sagen, die Lage sei kritisch."
        pure = score_attribution_vagueness(text)
        assert pure > 0.0

        nlp = _NLP(_Doc([_Sent([_Token("sagen")], [_Ent("ORG")])]))
        with_ner = score_attribution_vagueness(text, nlp=nlp)
        assert with_ner < pure

    def test_no_named_source_score_unchanged(self):
        text = "Officials warn of an imminent threat."
        pure = score_attribution_vagueness(text)

        nlp = _NLP(_Doc([_Sent([_Token("warn")], [])]))
        with_ner = score_attribution_vagueness(text, nlp=nlp)
        assert with_ner == pure

    def test_zero_raw_score_skips_nlp(self):
        # Text with no vague patterns: nlp should not be called
        text = "Der WHO-Sprecher bestätigte die Ergebnisse."
        assert score_attribution_vagueness(text) == 0.0

        class _FailNLP:
            def __call__(self, text):
                raise AssertionError("nlp() must not be called when raw_score is 0")

        assert score_attribution_vagueness(text, nlp=_FailNLP()) == 0.0

    def test_score_never_goes_below_zero(self):
        text = "Experts say this is dangerous."
        sent = _Sent([_Token("say")], [_Ent("PER")])
        nlp = _NLP(_Doc([sent, sent, sent]))  # max discount applied
        score = score_attribution_vagueness(text, nlp=nlp)
        assert score >= 0.0

    def test_backward_compat_no_nlp_arg(self):
        text = "Experts say this is dangerous."
        assert score_attribution_vagueness(text) == score_attribution_vagueness(text, nlp=None)
