"""Dynamic trust resolver for domains absent from source_trust.csv.

Replaces the old TLD heuristic with four live signals, each independently
fetched and cached in SQLite with a 7-day TTL so pipeline re-runs stay fast
but stale scores don't persist indefinitely.

Signal pipeline (evaluated in order):
  1. Google Safe Browsing — hard floor: flagged domains always score 5.
  2. Wikidata SPARQL      — is this domain a recognised news organisation?
  3. OpenPageRank         — domain authority 0–10 (optional: needs OPEN_PAGE_RANK_KEY).
  4. Domain age (WHOIS)   — years since registration, capped at 15.
  5. DNS authentication   — SPF + DMARC presence (0.5 each).

Score range for unknown domains: [30, 82].
  - 30 = no positive signals (unknown, unverifiable)
  - 82 = all signals maxed out (strong but unlisted outlet — below top MBFC entries)

Weights (sum to 1.0):
  With OPR key    → wikidata 0.35 / opr 0.30 / age 0.20 / dns 0.15
  Without OPR key → wikidata 0.45 / age 0.30 / dns 0.25
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_CACHE_TTL_DAYS: int = 7

_SCORE_FLOOR: float = 30.0
_SCORE_CEIL: float = 82.0
_SAFE_BROWSING_SCORE: float = 5.0

_WEIGHTS_WITH_OPR: dict[str, float] = {
    "wikidata": 0.35,
    "opr":      0.30,
    "age":      0.20,
    "dns":      0.15,
}
_WEIGHTS_WITHOUT_OPR: dict[str, float] = {
    "wikidata": 0.45,
    "age":      0.30,
    "dns":      0.25,
}

# Wikidata Qids for news-related entity types (P31 instance-of)
_NEWS_QTYPES = (
    "wd:Q1193236",  # news website
    "wd:Q11033",    # mass media
    "wd:Q1002697",  # online newspaper
    "wd:Q1153191",  # news magazine
    "wd:Q15265344", # broadcaster
    "wd:Q7216866",  # news organization
    "wd:Q1047870",  # public broadcaster
)


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


# ---------------------------------------------------------------------------
# Signal fetchers — each returns a normalised float in [0, 1] (or bool).
# All failures are caught and logged; the caller never sees an exception.
# ---------------------------------------------------------------------------

def _safe_browsing_flagged(domain: str, api_key: str) -> bool:
    """Return True if Google Safe Browsing v4 flags this domain."""
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    payload = json.dumps({
        "client": {"clientId": "hot-topics-dashboard", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": f"http://{domain}/"}],
        },
    }).encode()
    try:
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            body = json.loads(resp.read())
            return bool(body.get("matches"))
    except Exception as exc:
        logger.warning("Safe Browsing check failed for %s: %s", domain, exc)
        return False  # fail open — don't penalise on API error


def _wikidata_signal(domain: str) -> float:
    """Return 1.0 if domain is a recognised news organisation in Wikidata."""
    types_clause = " ".join(_NEWS_QTYPES)
    sparql = f"""
    SELECT ?item WHERE {{
      ?item wdt:P856 ?url .
      FILTER(CONTAINS(LCASE(STR(?url)), "{domain}"))
      ?item wdt:P31 ?type .
      VALUES ?type {{ {types_clause} }}
    }} LIMIT 1
    """
    url = (
        "https://query.wikidata.org/sparql?query="
        + urllib.parse.quote(sparql.strip())
        + "&format=json"
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "hot-topics-dashboard/1.0 (research project)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return 1.0 if data.get("results", {}).get("bindings") else 0.0
    except Exception as exc:
        logger.warning("Wikidata lookup failed for %s: %s", domain, exc)
        return 0.0


def _opr_signal(domain: str, api_key: str) -> float:
    """Return OpenPageRank score normalised to [0, 1] (10-point scale → fraction)."""
    url = (
        "https://openpagerank.com/api/v1.0/getPageRank?"
        + urllib.parse.urlencode({"domains[0]": domain})
    )
    try:
        req = urllib.request.Request(url, headers={"API-OPR": api_key})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
            rank = data["response"][0].get("page_rank_decimal") or 0.0
            return min(float(rank) / 10.0, 1.0)
    except Exception as exc:
        logger.warning("OPR lookup failed for %s: %s", domain, exc)
        return 0.0


def _age_signal(domain: str) -> float:
    """Return domain age normalised to [0, 1], capped at 15 years."""
    try:
        import whois as _whois  # python-whois; optional import guards missing dep
        w = _whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            age_days = (datetime.now() - created.replace(tzinfo=None)).days
            return min(age_days / (15.0 * 365.25), 1.0)
    except Exception as exc:
        logger.debug("WHOIS failed for %s: %s", domain, exc)
    return 0.0


def _dns_signal(domain: str) -> float:
    """Return [0, 1] based on SPF (0.5) + DMARC (0.5) presence."""
    try:
        import dns.resolver as _dns

        spf = False
        dmarc = False

        try:
            for rdata in _dns.resolve(domain, "TXT"):
                if "v=spf1" in str(rdata):
                    spf = True
                    break
        except Exception:
            pass

        try:
            for rdata in _dns.resolve(f"_dmarc.{domain}", "TXT"):
                if "v=DMARC1" in str(rdata):
                    dmarc = True
                    break
        except Exception:
            pass

        return (0.5 if spf else 0.0) + (0.5 if dmarc else 0.0)
    except ImportError:
        logger.debug("dnspython not installed — DNS auth signal skipped for %s", domain)
        return 0.0


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def _compute_live_score(domain: str) -> tuple[float, str]:
    """Fetch all signals and return a composite trust score in [0, 100].

    Returns:
        (score, method_tag) for storage in the cache.
    """
    from src.utils.config import GOOGLE_SAFE_BROWSING_KEY, OPEN_PAGE_RANK_KEY

    # Safe Browsing is a hard override — checked first, short-circuits everything
    if GOOGLE_SAFE_BROWSING_KEY and _safe_browsing_flagged(domain, GOOGLE_SAFE_BROWSING_KEY):
        logger.info("domain_resolver: %s flagged by Safe Browsing → score=%.1f", domain, _SAFE_BROWSING_SCORE)
        return _SAFE_BROWSING_SCORE, "safe_browsing_flagged"

    signals: dict[str, float] = {
        "wikidata": _wikidata_signal(domain),
        "age":      _age_signal(domain),
        "dns":      _dns_signal(domain),
    }

    if OPEN_PAGE_RANK_KEY:
        signals["opr"] = _opr_signal(domain, OPEN_PAGE_RANK_KEY)
        weights = _WEIGHTS_WITH_OPR
    else:
        weights = _WEIGHTS_WITHOUT_OPR

    raw = sum(signals[k] * weights[k] for k in signals if k in weights)
    score = round(_SCORE_FLOOR + raw * (_SCORE_CEIL - _SCORE_FLOOR), 1)

    method_tag = "live:" + ",".join(f"{k}={v:.2f}" for k, v in sorted(signals.items()))
    logger.info("domain_resolver: %s → %.1f (%s)", domain, score, method_tag)
    return score, method_tag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_trust(domain: str, conn: sqlite3.Connection) -> float:
    """Return a dynamically resolved trust score for an unknown domain.

    Serves from the SQLite cache if the entry is younger than _CACHE_TTL_DAYS.
    On a cache miss or expired entry, fetches live signals, computes a score,
    persists it, and returns it.

    This function should only be called for domains already confirmed to be
    absent from source_trust.csv (source_trust.py handles that check).

    Args:
        domain: Normalised domain (no 'www.' prefix, lowercase).
        conn: Active SQLite connection with domain_trust_cache table present.

    Returns:
        Trust score in [0.0, 100.0].
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=_CACHE_TTL_DAYS)
    ).isoformat()

    row = conn.execute(
        "SELECT trust_score FROM domain_trust_cache WHERE domain = ? AND cached_at > ?",
        (domain, cutoff),
    ).fetchone()
    if row is not None:
        return float(row[0])

    score, method = _compute_live_score(domain)
    conn.execute(
        """
        INSERT OR REPLACE INTO domain_trust_cache
            (domain, trust_score, method, cached_at)
        VALUES (?, ?, ?, ?)
        """,
        (domain, score, method, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return score
