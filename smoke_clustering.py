"""Smoke test — DB schema + clustering pipeline.

Run from the project root:
    .venv/bin/python smoke_clustering.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.utils.db import init_db, insert_items
from src.utils.models import RawItem
from src.utils.clustering import cluster_items


SAMPLE_ITEMS: list[RawItem] = [
    # --- cluster A: COVID vaccine ---
    {
        "id": "r_1", "title": "Scientists develop new COVID-19 vaccine with high efficacy",
        "description": None, "source": "r/science", "url": "https://reddit.com/r/1",
        "platform": "reddit", "timestamp": "2026-05-15T09:00:00Z",
        "engagement": {"score": 4200, "comments": 310},
    },
    {
        "id": "r_2", "title": "New COVID vaccine shows 95% efficacy in clinical trials",
        "description": None, "source": "r/news", "url": "https://reddit.com/r/2",
        "platform": "reddit", "timestamp": "2026-05-15T09:30:00Z",
        "engagement": {"score": 3800, "comments": 290},
    },
    {
        "id": "yt_1", "title": "COVID-19 vaccine efficacy results — scientists react",
        "description": None, "source": "YouTube", "url": "https://youtube.com/v/1",
        "platform": "youtube", "timestamp": "2026-05-15T10:00:00Z",
        "engagement": {"score": 15000, "comments": 820},
    },
    # --- cluster B: wildfire ---
    {
        "id": "n_1", "title": "Wildfires spread across southern California amid heatwave",
        "description": None, "source": "bbc.com", "url": "https://bbc.com/1",
        "platform": "newsapi", "timestamp": "2026-05-15T08:00:00Z",
        "engagement": {"score": 0, "comments": 0},
    },
    {
        "id": "n_2", "title": "California wildfire spreads rapidly during extreme heatwave",
        "description": None, "source": "cnn.com", "url": "https://cnn.com/1",
        "platform": "newsapi", "timestamp": "2026-05-15T08:45:00Z",
        "engagement": {"score": 0, "comments": 0},
    },
    # --- cluster C: crypto (singleton after fuzzy) ---
    {
        "id": "r_3", "title": "Bitcoin price crashes 20% in overnight trading session",
        "description": None, "source": "r/investing", "url": "https://reddit.com/r/3",
        "platform": "reddit", "timestamp": "2026-05-15T07:00:00Z",
        "engagement": {"score": 9100, "comments": 650},
    },
    # --- near-duplicate pair (fuzzy fallback candidate) ---
    {
        "id": "g_1", "title": "SpaceX Starship launch attempt scrubbed due to weather",
        "description": None, "source": "google_news", "url": "https://news.google.com/1",
        "platform": "google_news", "timestamp": "2026-05-15T11:00:00Z",
        "engagement": {"score": 0, "comments": 0},
    },
    {
        "id": "g_2", "title": "SpaceX Starship launch scrubbed because of weather conditions",
        "description": None, "source": "google_news", "url": "https://news.google.com/2",
        "platform": "google_news", "timestamp": "2026-05-15T11:15:00Z",
        "engagement": {"score": 0, "comments": 0},
    },
]


def hr(char: str = "─", width: int = 60) -> None:
    print(char * width)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "smoke.db"
        conn = init_db(db_path)

        hr("═")
        print("  SMOKE TEST — DB + Clustering pipeline")
        hr("═")

        inserted = insert_items(conn, SAMPLE_ITEMS)
        print(f"\n✓ Inserted {inserted} items into raw_items\n")

        hr()
        print("  Running cluster_items(distance_threshold=0.35, fuzzy_threshold=75)")
        hr()

        n_topics = cluster_items(conn, distance_threshold=0.35, fuzzy_threshold=75)
        print(f"\n✓ Created {n_topics} topics\n")

        # ── topics table ────────────────────────────────────────────
        hr()
        print("  TOPICS")
        hr()
        topics = conn.execute(
            "SELECT id, label, item_count FROM topics ORDER BY item_count DESC"
        ).fetchall()
        for t in topics:
            print(f"  [{t['id']:>2}]  ({t['item_count']} items)  {t['label']}")

        # ── topic_sources breakdown ──────────────────────────────────
        hr()
        print("  TOPIC → ITEMS")
        hr()
        for t in topics:
            members = conn.execute(
                """
                SELECT ri.id, ri.platform, ri.title
                FROM topic_sources ts
                JOIN raw_items ri ON ri.id = ts.item_id
                WHERE ts.topic_id = ?
                ORDER BY ri.id
                """,
                (t["id"],),
            ).fetchall()
            print(f"\n  Topic {t['id']} — \"{t['label']}\"")
            for m in members:
                print(f"    • [{m['platform']:>12}]  {m['id']:>5}  {m['title']}")

        # ── raw_items cluster_id stamp ───────────────────────────────
        hr()
        print("  RAW_ITEMS  (id → cluster_id)")
        hr()
        rows = conn.execute(
            "SELECT id, cluster_id FROM raw_items ORDER BY cluster_id, id"
        ).fetchall()
        for row in rows:
            print(f"  {row['id']:>6}  →  cluster {row['cluster_id']}")

        print()
        hr("═")
        print("  SMOKE TEST PASSED")
        hr("═")
        print()

        conn.close()


if __name__ == "__main__":
    main()
