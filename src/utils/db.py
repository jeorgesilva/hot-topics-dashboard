"""SQLite database helpers for the Hot Topics Dashboard.

Manages the raw_items table with the schema defined in Issue #1.
All NLP-derived columns (cleaned_text, tokens_json, etc.) are nullable
and filled later by Person A's pipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
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
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def run_schema_migrations(conn: sqlite3.Connection) -> None:
    """Apply incremental schema migrations. Safe to call multiple times.

    Each ALTER TABLE is wrapped in try/except because SQLite raises
    OperationalError when the column already exists — that is the expected
    idempotent behaviour.

    Args:
        conn: Active database connection.
    """
    migrations = [
        "ALTER TABLE raw_items    ADD COLUMN article_risk_score REAL",
        "ALTER TABLE topic_scores ADD COLUMN avg_article_risk   REAL",
        "ALTER TABLE topics       ADD COLUMN run_id INTEGER",
    ]
    for sql in migrations:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # column already exists — ok
    conn.commit()


def init_db(
    db_path: "Path | str | sqlite3.Connection | None" = None,
) -> sqlite3.Connection:
    """Create the database tables if they don't exist.

    Creates `raw_items` plus clustering tables (`topics`, `topic_sources`).
    NLP-derived columns on `raw_items` are nullable so scrapers can write rows
    before the NLP pipeline runs.

    Args:
        db_path: Path to the .db file, or an existing Connection (for tests).

    Returns:
        sqlite3.Connection ready for use.
    """
    if isinstance(db_path, sqlite3.Connection):
        conn = db_path
    else:
        conn = get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_items (
            id              TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            description     TEXT,
            body_text       TEXT,
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at   TEXT NOT NULL,
            completed_at TEXT,
            status       TEXT NOT NULL DEFAULT 'running'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            label       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            item_count  INTEGER NOT NULL DEFAULT 0,
            run_id      INTEGER REFERENCES pipeline_runs(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topic_sources (
            topic_id    INTEGER NOT NULL REFERENCES topics(id),
            item_id     TEXT    NOT NULL REFERENCES raw_items(id),
            PRIMARY KEY (topic_id, item_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topic_scores (
            topic_id                INTEGER PRIMARY KEY REFERENCES topics(id),
            avg_trust               REAL,
            trust_variance          REAL,
            coverage_breadth        INTEGER,
            coverage_ratio          REAL,
            avg_sentiment_extremity REAL,
            sensationalism_avg      REAL,
            framing_inconsistency   REAL,
            attribution_vagueness   REAL,
            fact_inconsistency      REAL,
            composite_risk          REAL,
            computed_at             TEXT,
            social_avg_trust               REAL,
            social_coverage_ratio          REAL,
            social_avg_sentiment_extremity REAL,
            social_sensationalism_avg      REAL,
            social_framing_inconsistency   REAL,
            social_attribution_vagueness   REAL,
            social_fact_inconsistency      REAL,
            social_risk                    REAL,
            narrative_divergence           REAL
        )
    """)
    # Dynamic trust resolver cache — keyed by domain, populated on first lookup.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS domain_trust_cache (
            domain      TEXT PRIMARY KEY,
            trust_score REAL NOT NULL,
            method      TEXT NOT NULL,
            cached_at   TEXT NOT NULL
        )
    """)

    # Session-5 and later migrations (idempotent).
    run_schema_migrations(conn)

    # Idempotent migration: add columns introduced after initial schema deploy.
    raw_existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(raw_items)").fetchall()
    }
    if "body_text" not in raw_existing:
        conn.execute("ALTER TABLE raw_items ADD COLUMN body_text TEXT")

    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(topic_scores)").fetchall()
    }
    for col, type_ in [
        ("attribution_vagueness",          "REAL"),
        ("fact_inconsistency",             "REAL"),
        ("social_avg_trust",               "REAL"),
        ("social_coverage_ratio",          "REAL"),
        ("social_avg_sentiment_extremity", "REAL"),
        ("social_sensationalism_avg",      "REAL"),
        ("social_framing_inconsistency",   "REAL"),
        ("social_attribution_vagueness",   "REAL"),
        ("social_fact_inconsistency",      "REAL"),
        ("social_risk",                    "REAL"),
        ("narrative_divergence",           "REAL"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE topic_scores ADD COLUMN {col} {type_}")
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
            (id, title, description, body_text, source, url, platform, timestamp, engagement_json)
        VALUES
            (:id, :title, :description, :body_text, :source, :url, :platform, :timestamp, :engagement_json)
        """,
        [
            {
                "id": item["id"],
                "title": item["title"],
                "description": item.get("description"),
                "body_text": item.get("body_text"),  # type: ignore[typeddict-item]
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


def start_run(conn: sqlite3.Connection) -> int:
    """Insert a pipeline_runs row with status 'running' and return its id.

    Args:
        conn: Active database connection.

    Returns:
        The new run_id (INTEGER PRIMARY KEY).
    """
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO pipeline_runs (started_at, status) VALUES (?, 'running')",
        (now,),
    )
    conn.commit()
    return cur.lastrowid


def complete_run(conn: sqlite3.Connection, run_id: int) -> None:
    """Mark a pipeline run as completed and record the finish time.

    Args:
        conn: Active database connection.
        run_id: ID previously returned by start_run.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE pipeline_runs SET completed_at = ?, status = 'completed' WHERE id = ?",
        (now, run_id),
    )
    conn.commit()


def _count_rows(conn: sqlite3.Connection) -> int:
    """Return the total number of rows in raw_items."""
    return conn.execute("SELECT COUNT(*) FROM raw_items").fetchone()[0]
