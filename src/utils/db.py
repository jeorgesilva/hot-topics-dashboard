"""SQLite database helpers for the Hot Topics Dashboard.

Manages the raw_items table with the schema defined in Issue #1.
All NLP-derived columns (cleaned_text, tokens_json, etc.) are nullable
and filled later by Person A's pipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.models import RawItem

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "dashboard.db"


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection to the SQLite database.

    Args:
        db_path: Path to the .db file. Defaults to data/dashboard.db.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create the database and raw_items table if they don't exist.

    The schema matches Issue #1 exactly. NLP-derived columns are nullable
    so Person B's scrapers can write rows before Person A's pipeline runs.

    Args:
        db_path: Path to the .db file.

    Returns:
        sqlite3.Connection ready for use.
    """
    conn = get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_items (
            id              TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            description     TEXT,
            source          TEXT NOT NULL,
            url             TEXT NOT NULL,
            platform        TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            engagement_json TEXT NOT NULL,
            cleaned_text    TEXT,
            tokens_json     TEXT,
            lemmas_json     TEXT,
            entities_json   TEXT,
            keywords_json   TEXT,
            cluster_id      INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_raw_items_platform
        ON raw_items(platform)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_raw_items_timestamp
        ON raw_items(timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_raw_items_cluster
        ON raw_items(cluster_id)
    """)
    conn.commit()
    logger.info("Database initialized at %s", db_path or DEFAULT_DB_PATH)
    return conn


def insert_items(
    conn: sqlite3.Connection,
    items: list[RawItem],
) -> int:
    """Insert RawItems into the database, skipping duplicates.

    Uses INSERT OR IGNORE so items with existing IDs are silently skipped.

    Args:
        conn: Active database connection.
        items: List of RawItem dicts from scrapers.

    Returns:
        Number of new rows actually inserted.
    """
    before = _count_rows(conn)
    conn.executemany(
        """
        INSERT OR IGNORE INTO raw_items
            (id, title, description, source, url, platform, timestamp, engagement_json)
        VALUES
            (:id, :title, :description, :source, :url, :platform, :timestamp, :engagement_json)
        """,
        [
            {
                "id": item["id"],
                "title": item["title"],
                "description": item.get("description"),
                "source": item["source"],
                "url": item["url"],
                "platform": item["platform"],
                "timestamp": item["timestamp"],
                "engagement_json": json.dumps(item["engagement"]),
            }
            for item in items
        ],
    )
    conn.commit()
    after = _count_rows(conn)
    inserted = after - before
    skipped = len(items) - inserted
    if skipped:
        logger.info("Skipped %d duplicate items", skipped)
    return inserted


def get_items(
    conn: sqlite3.Connection,
    *,
    platform: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read items from the database with optional filters.

    Args:
        conn: Active database connection.
        platform: Filter by platform ("reddit", "youtube", "google_news", "newsapi").
        since: ISO 8601 timestamp — return only items newer than this.
        limit: Maximum number of rows to return.

    Returns:
        List of row dicts with engagement_json parsed back to a dict.
    """
    query = "SELECT * FROM raw_items WHERE 1=1"
    params: list = []

    if platform:
        query += " AND platform = ?"
        params.append(platform)
    if since:
        query += " AND timestamp > ?"
        params.append(since)

    query += " ORDER BY timestamp DESC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["engagement"] = json.loads(d.pop("engagement_json"))
        # Parse JSON columns if present
        for json_col in ("tokens_json", "lemmas_json", "entities_json", "keywords_json"):
            raw = d.pop(json_col, None)
            clean_key = json_col.replace("_json", "")
            d[clean_key] = json.loads(raw) if raw else None
        results.append(d)
    return results


def _count_rows(conn: sqlite3.Connection) -> int:
    """Return the total number of rows in raw_items."""
    return conn.execute("SELECT COUNT(*) FROM raw_items").fetchone()[0]
