"""Dynamic trust resolver for domains absent from source_trust.csv.

For unknown domains, derives a trust score from TLD-based heuristics instead
of falling back to a flat default. Results are cached in SQLite so each domain
is resolved at most once per pipeline run (and reused across runs until the
cache is cleared).

Resolution precedence (handled by callers, not here):
  1. Static CSV  — source_trust.py's _TRUST_DB (highest priority)
  2. SQLite cache — checked here first on every call
  3. TLD heuristic — computed fresh on cache miss, then persisted

TLD scores are calibrated so:
  - .gov / .edu / .mil → HIGH (78–82) — institutional, regulated registration
  - DACH / Western-EU ccTLDs → slightly above neutral (52) — national registries
  - Generic .com / .net / .org → around neutral (44–52)
  - Known spam TLDs (.xyz, .top, .click, ...) → LOW (25–35)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Absolute trust score assigned to unknown domains by TLD.
# Calibrated relative to _UNKNOWN_SCORE = 45 in source_trust.py.
_TLD_SCORES: dict[str, float] = {
    # Institutional
    "gov": 82.0,
    "mil": 80.0,
    "edu": 78.0,
    # German-speaking / DACH
    "de": 52.0,
    "at": 52.0,
    "ch": 52.0,
    # Western Europe
    "fr": 52.0,
    "nl": 52.0,
    "be": 52.0,
    "dk": 52.0,
    "se": 52.0,
    "no": 52.0,
    "fi": 52.0,
    "it": 52.0,
    "es": 52.0,
    "pt": 52.0,
    "pl": 52.0,
    "cz": 52.0,
    "uk": 50.0,
    "ie": 50.0,
    # General / commercial
    "org": 52.0,
    "com": 46.0,
    "net": 44.0,
    "io":  44.0,
    "co":  44.0,
    "biz": 40.0,
    "info": 40.0,
    # Low-quality / spam-associated
    "xyz":    32.0,
    "top":    30.0,
    "click":  28.0,
    "online": 35.0,
    "site":   35.0,
    "win":    28.0,
    "loan":   25.0,
    "gq":     22.0,
    "cf":     22.0,
}

_DEFAULT_TLD_SCORE: float = 45.0


def _tld_score(domain: str) -> float:
    """Return a trust estimate based solely on the domain's TLD."""
    tld = domain.rsplit(".", 1)[-1].lower() if "." in domain else ""
    return _TLD_SCORES.get(tld, _DEFAULT_TLD_SCORE)


def init_cache(conn: sqlite3.Connection) -> None:
    """Create the domain_trust_cache table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS domain_trust_cache (
            domain      TEXT PRIMARY KEY,
            trust_score REAL NOT NULL,
            method      TEXT NOT NULL,
            cached_at   TEXT NOT NULL
        )
    """)
    conn.commit()


def resolve_trust(domain: str, conn: sqlite3.Connection) -> float:
    """Return a dynamically resolved trust score for an unknown domain.

    Checks the SQLite cache first. On a miss, computes a TLD heuristic score,
    stores it in the cache, and returns it. This function should only be called
    for domains already confirmed to be absent from source_trust.csv.

    Args:
        domain: Normalised domain (no 'www.' prefix, lowercase).
        conn: Active SQLite connection with domain_trust_cache table present.

    Returns:
        Trust score in [0.0, 100.0].
    """
    row = conn.execute(
        "SELECT trust_score FROM domain_trust_cache WHERE domain = ?",
        (domain,),
    ).fetchone()
    if row is not None:
        return float(row[0])

    score = _tld_score(domain)
    conn.execute(
        """
        INSERT OR REPLACE INTO domain_trust_cache
            (domain, trust_score, method, cached_at)
        VALUES (?, ?, 'heuristic', ?)
        """,
        (domain, score, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    logger.debug("domain_resolver: %s → %.1f (TLD=%s)", domain, score, domain.rsplit(".", 1)[-1] if "." in domain else "?")
    return score
