"""Claim verification via NLI (Fase 5).

Compares an extracted claim (src/nlp/claim_extractor.py) against retrieved
evidence (src/scoring/evidence_retriever.py) using the same multilingual NLI
model as src/scoring/contradiction.py (Fase 4, via the shared src/nlp/nli.py
pipeline), returning one of:

  - "supported"            — evidence entails the claim
  - "refuted"               — evidence contradicts the claim
  - "not_enough_evidence"   — no evidence was found, or the NLI model was
                               not confident enough either way

This module intentionally does NOT feed into composite_risk (that's Fase 7).
It only builds and persists claim_verifications rows so results can be
reviewed before any scoring integration decision is made.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Literal, TypedDict

from src.nlp.nli import get_pipeline
from src.scoring.evidence_retriever import Evidence, retrieve_evidence

logger = logging.getLogger(__name__)

Verdict = Literal["supported", "refuted", "not_enough_evidence"]

_ENTAILMENT_THRESHOLD = 0.60
_CONTRADICTION_THRESHOLD = 0.60


class ClaimVerification(TypedDict):
    claim_text: str
    verdict: Verdict
    evidence_text: str | None
    evidence_url: str | None
    evidence_source_type: str | None
    confidence: float


def verify_claim(claim_text: str, evidence: Evidence | None) -> ClaimVerification:
    """Compare a claim against retrieved evidence via NLI.

    Args:
        claim_text: The claim sentence to verify.
        evidence: Evidence returned by evidence_retriever.retrieve_evidence,
            or None if nothing was found.

    Returns:
        ClaimVerification with verdict "not_enough_evidence" (confidence 0.0)
        when evidence is None, otherwise the NLI-derived verdict.
    """
    if evidence is None:
        return ClaimVerification(
            claim_text=claim_text,
            verdict="not_enough_evidence",
            evidence_text=None,
            evidence_url=None,
            evidence_source_type=None,
            confidence=0.0,
        )

    pipe = get_pipeline()
    raw = pipe([{"text": claim_text, "text_pair": evidence.text}])[0]
    scores = {entry["label"].lower(): entry["score"] for entry in raw}

    contradiction_score = scores.get("contradiction", 0.0)
    entailment_score = scores.get("entailment", 0.0)

    if contradiction_score >= _CONTRADICTION_THRESHOLD:
        verdict: Verdict = "refuted"
        confidence = contradiction_score
    elif entailment_score >= _ENTAILMENT_THRESHOLD:
        verdict = "supported"
        confidence = entailment_score
    else:
        verdict = "not_enough_evidence"
        confidence = max(scores.values()) if scores else 0.0

    return ClaimVerification(
        claim_text=claim_text,
        verdict=verdict,
        evidence_text=evidence.text,
        evidence_url=evidence.url,
        evidence_source_type=evidence.source_type,
        confidence=round(confidence, 4),
    )


def save_verification(
    conn: sqlite3.Connection,
    topic_id: int | None,
    item_id: str | None,
    verification: ClaimVerification,
) -> None:
    """Persist a ClaimVerification row to claim_verifications.

    Args:
        conn: Active database connection with the claim_verifications table
            present (created by src.utils.db.init_db).
        topic_id: Topic the claim's article belongs to, if known.
        item_id: Article the claim was extracted from, if known.
        verification: Result from verify_claim.
    """
    conn.execute(
        """
        INSERT INTO claim_verifications
            (topic_id, item_id, claim_text, verdict,
             evidence_url, evidence_snippet, confidence, checked_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            topic_id,
            item_id,
            verification["claim_text"],
            verification["verdict"],
            verification["evidence_url"],
            verification["evidence_text"],
            verification["confidence"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def verify_article_claims(
    item_id: str,
    cleaned_text: str,
    conn: sqlite3.Connection,
    topic_id: int | None = None,
    persist: bool = True,
) -> list[ClaimVerification]:
    """End-to-end pipeline for one article: extract claims, retrieve evidence,
    verify each claim, and optionally persist the results.

    Args:
        item_id: The article's raw_items.id, used to exclude it from the
            corpus-fallback evidence tier and for persistence.
        cleaned_text: The article's cleaned_text.
        conn: Active database connection.
        topic_id: Topic the article belongs to. Enables the corpus-fallback
            evidence tier and is stored alongside each persisted row.
        persist: If True, write each verification to claim_verifications.

    Returns:
        List of ClaimVerification results, one per extracted claim.
    """
    from src.nlp.claim_extractor import extract_claims

    claims = extract_claims(cleaned_text)
    results: list[ClaimVerification] = []

    for claim in claims:
        evidence = retrieve_evidence(
            claim["text"],
            entities=claim["entities"],
            conn=conn,
            topic_id=topic_id,
            exclude_item_id=item_id,
        )
        verification = verify_claim(claim["text"], evidence)
        results.append(verification)
        if persist:
            save_verification(conn, topic_id, item_id, verification)

    return results
