"""Topic clustering for raw_items.

Pipeline:
  1. Load all items from raw_items.
  2. Clean titles (lowercase, strip punctuation).
  3. TF-IDF vectorize; compute pairwise cosine distances.
  4. AgglomerativeClustering (average linkage, cosine distance_threshold ≈ 0.35).
  5. Fallback: merge singleton clusters via RapidFuzz token_sort_ratio ≥ 75.
  6. Rebuild topics + topic_sources; stamp cluster_id onto raw_items.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_title(title: str) -> str:
    t = _PUNCT_RE.sub(" ", title.lower())
    return _WHITESPACE_RE.sub(" ", t).strip()


def _fuzzy_merge(labels: list[int], cleaned: list[str], threshold: int) -> list[int]:
    """Absorb singleton clusters into the best-matching non-singleton cluster.

    Iterates until no more merges are possible so chains resolve correctly.
    """
    for _ in range(len(labels)):
        sizes = Counter(labels)
        singletons = [i for i, lbl in enumerate(labels) if sizes[lbl] == 1]
        if not singletons:
            break
        merged_any = False
        for idx in singletons:
            best_score = threshold - 1
            best_label = labels[idx]
            for other_idx, other_cleaned in enumerate(cleaned):
                if other_idx == idx:
                    continue
                score = fuzz.token_sort_ratio(cleaned[idx], other_cleaned)
                if score > best_score:
                    best_score = score
                    best_label = labels[other_idx]
            if best_label != labels[idx]:
                labels[idx] = best_label
                merged_any = True
        if not merged_any:
            break
    return labels


def _representative_title(titles: list[str]) -> str:
    """Return the longest title — tends to be the most descriptive."""
    return max(titles, key=len)


def _rebuild_topics(
    conn: sqlite3.Connection,
    ids: list[str],
    titles: list[str],
    labels: list[int],
) -> None:
    """Atomically replace topics + topic_sources and update raw_items.cluster_id."""
    now = datetime.now(timezone.utc).isoformat()

    clusters: dict[int, list[tuple[str, str]]] = {}
    for item_id, title, label in zip(ids, titles, labels):
        clusters.setdefault(label, []).append((item_id, title))

    conn.execute("DELETE FROM topic_sources")
    conn.execute("DELETE FROM topics")

    for cluster_id, members in sorted(clusters.items()):
        member_titles = [t for _, t in members]
        conn.execute(
            "INSERT INTO topics (id, label, created_at, item_count) VALUES (?, ?, ?, ?)",
            (cluster_id, _representative_title(member_titles), now, len(members)),
        )
        conn.executemany(
            "INSERT INTO topic_sources (topic_id, item_id) VALUES (?, ?)",
            [(cluster_id, item_id) for item_id, _ in members],
        )
        conn.executemany(
            "UPDATE raw_items SET cluster_id = ? WHERE id = ?",
            [(cluster_id, item_id) for item_id, _ in members],
        )

    conn.commit()
    logger.info("Rebuilt %d topics from %d items.", len(clusters), len(ids))


def cluster_items(
    conn: sqlite3.Connection,
    *,
    distance_threshold: float = 0.35,
    fuzzy_threshold: int = 75,
) -> int:
    """Cluster all raw_items by title similarity and persist results.

    Runs TF-IDF + agglomerative clustering (cosine distance). Singleton
    clusters are then merged via RapidFuzz token_sort_ratio as a fallback.
    Completely replaces the topics and topic_sources tables on each call.

    Args:
        conn: Active database connection (must have topics + topic_sources tables).
        distance_threshold: Cosine distance cut-off for agglomerative clustering.
        fuzzy_threshold: Minimum token_sort_ratio (0–100) to merge singletons.

    Returns:
        Number of distinct topics created.
    """
    rows = conn.execute(
        "SELECT id, title FROM raw_items ORDER BY timestamp DESC"
    ).fetchall()

    if not rows:
        logger.info("No items to cluster.")
        return 0

    ids = [r["id"] for r in rows]
    titles = [r["title"] for r in rows]
    cleaned = [_clean_title(t) for t in titles]

    if len(rows) == 1:
        labels: list[int] = [0]
    else:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(cleaned)
        dist_matrix = cosine_distances(tfidf)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clustering.fit_predict(dist_matrix).tolist()
        labels = _fuzzy_merge(labels, cleaned, fuzzy_threshold)

    # Compact label space after potential merges
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    labels = [remap[lbl] for lbl in labels]

    _rebuild_topics(conn, ids, titles, labels)
    return len(set(labels))
