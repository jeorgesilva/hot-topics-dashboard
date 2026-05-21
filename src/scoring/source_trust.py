"""Source trust DB loader and per-topic coverage metrics (Week 2, Step 4).

Loads the MBFC-derived trust scores from config/source_trust.csv and
computes four coverage-quality signals for each topic cluster:

  avg_trust         — mean trust score across all articles (0–100)
  trust_variance    — std dev of trust scores (high → polarised coverage)
  coverage_breadth  — count of unique credible domains (trust ≥ HIGH_TRUST_THRESHOLD)
  coverage_ratio    — fraction of articles from credible sources

These are the Person-B signals for the composite risk formula in compute_scores.py.
Person A fills avg_sentiment_extremity, sensationalism_avg, and framing_inconsistency
into the same topic_scores rows.
"""

from __future__ import annotations

import csv
import logging
import math
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CSV_PATH = _PROJECT_ROOT / "config" / "source_trust.csv"

NEUTRAL_SCORE: float = 50.0       # used as explicit default in compute_coverage_metrics
HIGH_TRUST_THRESHOLD: float = 60.0

# Scores assigned to domains absent from the trust DB.
# A missing domain is more likely to be a shell/spam site than a neutral outlet,
# so we use a penalty below 50 instead of the old neutral default.
_UNKNOWN_SCORE: float = 45.0          # unknown domain, non-breaking topic
_UNKNOWN_BREAKING_SCORE: float = 35.0 # unknown domain in a breaking/viral story


def _load_trust_db(csv_path: Path = _CSV_PATH) -> dict[str, float]:
    """Load domain → trust_score mapping from CSV.

    Strips leading 'www.' so both 'bbc.com' and 'www.bbc.com' resolve correctly.
    Skips rows with missing or non-numeric trust_score values.
    """
    db: dict[str, float] = {}
    if not csv_path.exists():
        logger.warning("source_trust.csv not found at %s — all domains will use neutral score", csv_path)
        return db
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw_domain = row.get("domain", "").strip().lower()
            raw_score = row.get("trust_score", "").strip()
            if not raw_domain or not raw_score:
                continue
            try:
                score = float(raw_score)
            except ValueError:
                continue
            domain = raw_domain.removeprefix("www.")
            db[domain] = score
    logger.info("Loaded trust scores for %d domains from %s", len(db), csv_path)
    return db


# Module-level cache — loaded once per process.
_TRUST_DB: dict[str, float] = _load_trust_db()


def get_trust_score(
    domain: str,
    neutral: float | None = None,
    topic_is_breaking: bool = False,
) -> float:
    """Return the trust score (0–100) for a domain.

    Strips 'www.' prefix before lookup. For unknown domains the returned score
    depends on context: a domain not in the trust DB is penalised below 50
    because novel or obscure domains are more likely to be shell/spam sites
    than genuinely neutral outlets. The penalty is steeper for breaking stories
    where coordinated misinformation spikes.

    Args:
        domain: Bare domain name, e.g. 'bbc.com' or 'www.reuters.com'.
        neutral: Explicit override score for unknown domains. When provided,
            bypasses the contextual defaults (for backwards-compatible call
            sites and compute_coverage_metrics which manages its own neutral).
        topic_is_breaking: If True, unknown domains receive _UNKNOWN_BREAKING_SCORE
            (35) instead of _UNKNOWN_SCORE (45).

    Returns:
        Trust score in [0, 100].
    """
    key = domain.lower().strip().removeprefix("www.")
    if key in _TRUST_DB:
        return _TRUST_DB[key]
    if neutral is not None:
        return neutral
    return _UNKNOWN_BREAKING_SCORE if topic_is_breaking else _UNKNOWN_SCORE


def _domain_from_url(url: str) -> str:
    """Extract and normalise the domain from a URL string."""
    candidate = url.strip()
    if not candidate:
        return ""

    host: str | None = None
    try:
        parsed = urlparse(candidate if "://" in candidate or candidate.startswith("//") else f"//{candidate}")
        host = parsed.hostname
    except Exception:
        host = None

    if not host:
        host = candidate.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        host = host.rsplit("@", 1)[-1].split(":", 1)[0]

    return host.lower().removeprefix("www.")


def compute_coverage_metrics(
    topic_id: int,
    conn: sqlite3.Connection,
    *,
    neutral: float = NEUTRAL_SCORE,
    high_trust_threshold: float = HIGH_TRUST_THRESHOLD,
) -> dict[str, float | int]:
    """Compute coverage-quality metrics for a single topic.

    Queries topic_sources → raw_items, derives trust scores from source
    domains, and returns the four coverage signals.

    Args:
        topic_id: ID of the topic in the topics table.
        conn: Active database connection with row_factory=sqlite3.Row.
        neutral: Trust score assigned to unknown domains.
        high_trust_threshold: Minimum score to count as a credible source.

    Returns:
        Dict with keys: avg_trust, trust_variance, coverage_breadth,
        coverage_ratio. All default to neutral/0 when the topic has no articles.
    """
    rows = conn.execute(
        """
        SELECT ri.url, ri.source
        FROM topic_sources ts
        JOIN raw_items ri ON ri.id = ts.item_id
        WHERE ts.topic_id = ?
        """,
        (topic_id,),
    ).fetchall()

    if not rows:
        return {
            "avg_trust": neutral,
            "trust_variance": 0.0,
            "coverage_breadth": 0,
            "coverage_ratio": 0.0,
        }

    scores: list[float] = []
    credible_domains: set[str] = set()

    for row in rows:
        domain = _domain_from_url(row["url"]) or row["source"]
        score = get_trust_score(domain, neutral=neutral)
        scores.append(score)
        if score >= high_trust_threshold:
            credible_domains.add(domain)

    n = len(scores)
    avg = sum(scores) / n
    variance = math.sqrt(sum((s - avg) ** 2 for s in scores) / n) if n > 1 else 0.0
    credible_count = sum(1 for s in scores if s >= high_trust_threshold)

    return {
        "avg_trust": round(avg, 4),
        "trust_variance": round(variance, 4),
        "coverage_breadth": len(credible_domains),
        "coverage_ratio": round(credible_count / n, 4),
    }


def score_coverage(conn: sqlite3.Connection) -> int:
    """Compute and persist coverage metrics for every topic.

    Upserts avg_trust, trust_variance, coverage_breadth, and coverage_ratio
    into topic_scores. Leaves Person-A columns (avg_sentiment_extremity,
    sensationalism_avg, framing_inconsistency) untouched.

    Args:
        conn: Active database connection.

    Returns:
        Number of topics processed.
    """
    topic_ids = [r["id"] for r in conn.execute("SELECT id FROM topics").fetchall()]

    for topic_id in topic_ids:
        metrics = compute_coverage_metrics(topic_id, conn)
        conn.execute(
            """
            INSERT INTO topic_scores
                (topic_id, avg_trust, trust_variance, coverage_breadth, coverage_ratio)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(topic_id) DO UPDATE SET
                avg_trust        = excluded.avg_trust,
                trust_variance   = excluded.trust_variance,
                coverage_breadth = excluded.coverage_breadth,
                coverage_ratio   = excluded.coverage_ratio
            """,
            (
                topic_id,
                metrics["avg_trust"],
                metrics["trust_variance"],
                metrics["coverage_breadth"],
                metrics["coverage_ratio"],
            ),
        )

    conn.commit()
    logger.info("Coverage metrics scored for %d topics.", len(topic_ids))
    return len(topic_ids)
