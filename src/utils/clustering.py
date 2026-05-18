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
    """Merge singleton clusters with the best-matching item anywhere in the list.

    Two singletons can merge with each other. Iterates until stable so
    chains (A→B→C) resolve in a single call.
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
    """Replace topics + topic_sources and update raw_items.cluster_id atomically.

    Wraps all mutations in a single transaction; rolls back on any failure so
    the DB is never left in a partially rebuilt state.
    """
    now = datetime.now(timezone.utc).isoformat()

    clusters: dict[int, list[tuple[str, str]]] = {}
    for item_id, title, label in zip(ids, titles, labels):
        clusters.setdefault(label, []).append((item_id, title))

    with conn:
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
    Completely replaces the topics and topic_sources tables on each call,
    including clearing them when raw_items is empty.

    Args:
        conn: Active database connection created via db.get_connection() or
            db.init_db() — must have row_factory=sqlite3.Row and the
            topics/topic_sources tables present.
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
        with conn:
            conn.execute("DELETE FROM topic_sources")
            conn.execute("DELETE FROM topics")
        return 0

    ids = [r["id"] for r in rows]
    titles = [r["title"] for r in rows]
    cleaned = [_clean_title(t) for t in titles]

    # Items whose cleaned title is empty (blank or punctuation-only) bypass
    # TF-IDF and are assigned unique singleton labels to avoid empty-vocabulary errors.
    valid = [(i, c) for i, c in enumerate(cleaned) if c]
    empty_idx = [i for i, c in enumerate(cleaned) if not c]

    labels: list[int | None] = [None] * len(rows)

    if len(valid) >= 2:
        valid_positions, valid_cleaned = zip(*valid)
        valid_cleaned_list = list(valid_cleaned)
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )
            tfidf = vectorizer.fit_transform(valid_cleaned_list)
            dist_matrix = cosine_distances(tfidf)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=distance_threshold,
            )
            valid_labels = clustering.fit_predict(dist_matrix).tolist()
            valid_labels = _fuzzy_merge(valid_labels, valid_cleaned_list, fuzzy_threshold)
        except ValueError:
            logger.warning(
                "TF-IDF vectorization failed (empty vocabulary after stop-word removal); "
                "assigning singleton clusters."
            )
            valid_labels = list(range(len(valid)))
        for pos, idx in enumerate(valid_positions):
            labels[idx] = valid_labels[pos]
    elif len(valid) == 1:
        labels[valid[0][0]] = 0

    # Assign unique labels to empty-title items
    next_label = max((lbl for lbl in labels if lbl is not None), default=-1) + 1
    for idx in empty_idx:
        labels[idx] = next_label
        next_label += 1

    final_labels: list[int] = labels  # type: ignore[assignment]

    # Compact label space after merges and empty-item offsets
    unique = sorted(set(final_labels))
    remap = {old: new for new, old in enumerate(unique)}
    final_labels = [remap[lbl] for lbl in final_labels]

    _rebuild_topics(conn, ids, titles, final_labels)
    return len(set(final_labels))
