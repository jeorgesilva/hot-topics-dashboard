"""Runtime source metadata lookup for domains not in the static CSV or with low confidence.

Checks the static source_trust.csv first, then a 30-day SQLite cache, then
queries MBFC via SearXNG to retrieve factual reporting and bias ratings.

Public API:
    get_source_data(domain, csv_path, db_path) -> SourceData
    generate_disclaimer(data) -> str
    domain_in_static_csv(domain, csv_path) -> bool
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.utils.config import SEARXNG_URL

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CSV = _PROJECT_ROOT / "config" / "source_trust.csv"
_DEFAULT_DB = _PROJECT_ROOT / "data" / "dashboard.db"
_CACHE_TTL_DAYS = 30
_MBFC_SLEEP_SECONDS = 1.5


@dataclass
class SourceData:
    domain: str
    factual_rating: Optional[str]
    bias_label: Optional[str]
    source: str           # "MBFC" | "Presserat" | "BDZV" | "Correctiv" | "manual_estimate" | "unavailable"
    confidence: str       # "verified" | "low" | "unavailable"
    mbfc_url: Optional[str]
    notes: str
    fetched_at: str       # ISO timestamp


# ---------------------------------------------------------------------------
# SQLite cache management
# ---------------------------------------------------------------------------

def _init_cache(db_path: str) -> None:
    """Create the source_lookup_cache table if it does not exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_lookup_cache (
            domain         TEXT PRIMARY KEY,
            factual_rating TEXT,
            bias_label     TEXT,
            source         TEXT,
            confidence     TEXT,
            mbfc_url       TEXT,
            notes          TEXT,
            fetched_at     TEXT
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Static CSV loader
# ---------------------------------------------------------------------------

# Module-level cache — loaded once per (csv_path, process) lifetime.
_CSV_CACHE: dict[str, dict[str, SourceData]] = {}


def _load_csv(csv_path: str) -> dict[str, SourceData]:
    path = Path(csv_path)
    if not path.exists():
        logger.warning("source_trust.csv not found at %s", path)
        return {}
    result: dict[str, SourceData] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("domain", "").strip().lower()
            if not raw:
                continue
            domain = raw.removeprefix("www.")
            result[domain] = SourceData(
                domain=domain,
                factual_rating=row.get("factual_rating") or None,
                bias_label=row.get("bias_label") or None,
                source=row.get("source") or "manual_estimate",
                confidence=row.get("confidence") or "low",
                mbfc_url=row.get("mbfc_url") or None,
                notes=row.get("notes") or "",
                fetched_at="",
            )
    logger.debug("source_lookup: loaded %d CSV entries from %s", len(result), path)
    return result


def _get_csv(csv_path: str) -> dict[str, SourceData]:
    if csv_path not in _CSV_CACHE:
        _CSV_CACHE[csv_path] = _load_csv(csv_path)
    return _CSV_CACHE[csv_path]


# ---------------------------------------------------------------------------
# MBFC fetcher
# ---------------------------------------------------------------------------

def _fetch_from_mbfc(domain: str, searxng_base_url: str) -> SourceData:
    """Look up a domain on MBFC via SearXNG and parse the result page.

    Args:
        domain: Normalised domain (no www. prefix, lowercase).
        searxng_base_url: Base URL of the running SearXNG instance.

    Returns:
        SourceData populated from MBFC if found; source='unavailable' otherwise.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Step 1: Query SearXNG for an MBFC page about this domain.
    mbfc_url: str | None = None
    try:
        resp = requests.get(
            f"{searxng_base_url}/search",
            params={
                "q": f'"{domain}" site:mediabiasfactcheck.com',
                "format": "json",
                "categories": "general",
                "language": "en",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for result in (data.get("results") or []):
            url = (result.get("url") or "").strip()
            if "mediabiasfactcheck.com" in url:
                mbfc_url = url
                break
    except Exception as exc:
        logger.warning("source_lookup: SearXNG lookup failed for %s: %s", domain, exc)

    if not mbfc_url:
        logger.info("source_lookup: no MBFC result for %s", domain)
        return SourceData(
            domain=domain,
            factual_rating=None,
            bias_label=None,
            source="unavailable",
            confidence="unavailable",
            mbfc_url=None,
            notes="not found in MBFC via SearXNG",
            fetched_at=now,
        )

    # Rate-limit between the SearXNG call and the MBFC page fetch.
    time.sleep(_MBFC_SLEEP_SECONDS)

    # Step 2: Fetch the MBFC page and extract factual/bias labels.
    factual_rating: str | None = None
    bias_label: str | None = None
    try:
        page_resp = requests.get(
            mbfc_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; hot-topics-dashboard/1.0)"},
            timeout=12,
        )
        page_resp.raise_for_status()
        soup = BeautifulSoup(page_resp.text, "html.parser")
        body_text = soup.get_text(separator="\n")

        for line in body_text.splitlines():
            upper = line.strip().upper()
            if "FACTUAL REPORTING:" in upper and factual_rating is None:
                value = line.strip().split(":", 1)[-1].strip().upper()
                if value:
                    factual_rating = value
            if upper.startswith("BIAS:") and bias_label is None:
                value = line.strip().split(":", 1)[-1].strip().upper()
                if value:
                    bias_label = value
    except Exception as exc:
        logger.warning("source_lookup: MBFC page fetch failed for %s: %s", domain, exc)

    logger.info(
        "source_lookup: MBFC result for %s — factual=%s bias=%s url=%s",
        domain, factual_rating, bias_label, mbfc_url,
    )
    return SourceData(
        domain=domain,
        factual_rating=factual_rating,
        bias_label=bias_label,
        source="MBFC",
        confidence="verified",
        mbfc_url=mbfc_url,
        notes="",
        fetched_at=now,
    )


# ---------------------------------------------------------------------------
# Cache row deserialiser
# ---------------------------------------------------------------------------

def _row_to_source_data(row: sqlite3.Row) -> SourceData:
    return SourceData(
        domain=row["domain"],
        factual_rating=row["factual_rating"],
        bias_label=row["bias_label"],
        source=row["source"],
        confidence=row["confidence"],
        mbfc_url=row["mbfc_url"],
        notes=row["notes"] or "",
        fetched_at=row["fetched_at"] or "",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_source_data(
    domain: str,
    csv_path: str | None = None,
    db_path: str | None = None,
) -> SourceData:
    """Return source metadata for a domain.

    Resolution order:
    1. Static CSV — if confidence=verified, return immediately (no network call).
    2. SQLite runtime cache — return if entry is not older than 30 days.
    3. MBFC via SearXNG — fetch live data.
    4. Store result in cache and return.

    For domains in the CSV with confidence=low, if the MBFC fetch fails, the
    original CSV entry is returned rather than an 'unavailable' result — the
    existing low-confidence data is more informative than nothing.

    Args:
        domain: Domain name, e.g. 'bbc.com' or 'www.reuters.com'.
        csv_path: Path to source_trust.csv. Defaults to config/source_trust.csv.
        db_path: Path to the SQLite database for the runtime cache.

    Returns:
        SourceData for the domain.
    """
    _csv_path = str(csv_path or _DEFAULT_CSV)
    _db_path = str(db_path or _DEFAULT_DB)

    key = domain.lower().strip().removeprefix("www.")
    now = datetime.now(timezone.utc).isoformat()

    # 1. Static CSV.
    csv_db = _get_csv(_csv_path)
    csv_entry = csv_db.get(key)
    if csv_entry is not None and csv_entry.confidence == "verified":
        return csv_entry

    # Ensure the cache table exists.
    _init_cache(_db_path)

    # 2. SQLite runtime cache.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_CACHE_TTL_DAYS)).isoformat()
    try:
        conn = sqlite3.connect(_db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM source_lookup_cache WHERE domain = ? AND fetched_at > ?",
            (key, cutoff),
        ).fetchone()
        conn.close()
        if row is not None:
            return _row_to_source_data(row)
    except Exception as exc:
        logger.warning("source_lookup: cache read failed: %s", exc)

    # 3. Fetch from MBFC.
    searxng_base = SEARXNG_URL or "http://localhost:8080"
    data = _fetch_from_mbfc(key, searxng_base)

    # If MBFC failed and the domain was in the CSV, prefer the CSV entry.
    if data.source == "unavailable" and csv_entry is not None:
        return csv_entry

    # 4. Persist to cache.
    try:
        conn = sqlite3.connect(_db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO source_lookup_cache
                (domain, factual_rating, bias_label, source, confidence,
                 mbfc_url, notes, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                data.factual_rating,
                data.bias_label,
                data.source,
                data.confidence,
                data.mbfc_url,
                data.notes,
                data.fetched_at or now,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("source_lookup: cache write failed: %s", exc)

    return data


def domain_in_static_csv(domain: str, csv_path: str | None = None) -> bool:
    """Return True if the domain appears in the static source_trust.csv.

    Used by the dashboard to decide whether to show 'Source data unavailable'
    (known outlet, lookup failed) versus showing nothing (unknown domain).
    """
    key = domain.lower().strip().removeprefix("www.")
    return key in _get_csv(str(csv_path or _DEFAULT_CSV))


def generate_disclaimer(data: SourceData) -> str:
    """Generate a human-readable source disclaimer for display in the dashboard.

    Args:
        data: SourceData returned by get_source_data().

    Returns:
        Disclaimer string, or empty string when no useful text can be generated.
        The caller is responsible for the 'unavailable' edge cases described in
        the dashboard's _render_article_disclaimer helper.
    """
    if data.source == "MBFC" and data.confidence == "verified":
        if data.factual_rating and data.bias_label:
            rating_clause = (
                f" has a {data.factual_rating} factual reporting record"
                f" and a {data.bias_label} political bias."
            )
        elif data.factual_rating:
            rating_clause = f" has a {data.factual_rating} factual reporting record."
        elif data.bias_label:
            rating_clause = f" has a {data.bias_label} political bias."
        else:
            rating_clause = "."
        text = f"According to Media Bias/Fact Check, {data.domain}{rating_clause}"
        if data.mbfc_url:
            text += f" {data.mbfc_url}"
        return text

    if data.source == "MBFC" and data.confidence == "low":
        return (
            "This outlet was previously associated with a Media Bias/Fact Check entry"
            " but the rating has not been re-verified."
        )

    if data.source == "Presserat" and data.confidence == "verified":
        return (
            f"{data.domain} is a member of the German Press Council (Presserat)"
            " and bound by the Pressekodex."
        )

    if data.source == "BDZV" and data.confidence == "verified":
        return (
            f"{data.domain} is a member of the Federal Association of"
            " German Newspaper Publishers (BDZV)."
        )

    if data.source == "Correctiv" and data.confidence == "verified":
        return (
            "This outlet has been directly evaluated by Correctiv,"
            " a German non-profit fact-checking organisation."
        )

    if data.source in ("manual_estimate", "unavailable"):
        return (
            "No independent evaluation was found for this source"
            " in known fact-checking databases."
        )

    return ""
