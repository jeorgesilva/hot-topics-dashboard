"""Tests for NLP cache behavior in run_nlp.py (Sessão 3).

Verifies that:
- Articles already having cleaned_text skip spaCy preprocessing.
- Articles without cleaned_text have the field populated after scoring.
- topic_scores are identical regardless of cache state, and cached
  cleaned_text values are never overwritten on subsequent runs.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.scoring.sentiment import score_articles
from src.utils.db import init_db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_pipeline(pos: float = 0.75, neg: float = 0.05):
    neutral = round(1.0 - pos - neg, 4)

    def predict(texts, batch_size: int = 16):
        return [
            [
                {"label": "positive", "score": pos},
                {"label": "neutral",  "score": neutral},
                {"label": "negative", "score": neg},
            ]
            for _ in texts
        ]

    return predict


def _make_item(item_id: str, cleaned_text: str | None = None) -> dict:
    return {
        "id": item_id,
        "title": f"Titel {item_id}",
        "description": "Beschreibung.",
        "body_text": None,
        "source": "tagesschau.de",
        "url": f"https://tagesschau.de/{item_id}",
        "platform": "rss",
        "timestamp": "2026-06-01T10:00:00Z",
        "engagement_json": "{}",
        "cleaned_text": cleaned_text,
    }


@pytest.fixture
def db_path(tmp_path):
    """Temp DB with two rss articles linked to one topic, no cleaned_text."""
    path = str(tmp_path / "test.db")
    conn = init_db(path)
    conn.execute("""
        INSERT INTO raw_items
            (id, title, description, body_text, source, url,
             platform, timestamp, engagement_json)
        VALUES
            ('art1', 'Artikel eins', 'Beschreibung.', NULL, 'tagesschau.de',
             'https://tagesschau.de/1', 'rss', '2026-06-01T10:00:00Z', '{}'),
            ('art2', 'Artikel zwei', 'Beschreibung.', NULL, 'spiegel.de',
             'https://spiegel.de/2', 'rss', '2026-06-01T10:00:00Z', '{}')
    """)
    conn.execute(
        "INSERT INTO topics (id, label, created_at, item_count) "
        "VALUES (1, 'TestTopic', '2026-06-01T00:00:00Z', 2)"
    )
    conn.execute(
        "INSERT INTO topic_sources (topic_id, item_id) VALUES (1, 'art1'), (1, 'art2')"
    )
    conn.commit()
    conn.close()
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_uncached_articles_are_processed(monkeypatch):
    """Articles without cleaned_text get cleaned_text populated after scoring."""
    def fake_preprocess(item):
        return {**item, "cleaned_text": "verarbeitet " + item["title"], "tokens": [], "lemmas": []}

    monkeypatch.setattr("src.scoring.sentiment.preprocess", fake_preprocess)
    monkeypatch.setattr("src.scoring.sentiment._get_pipeline", lambda: _fake_pipeline())

    article = _make_item("new_art", cleaned_text=None)
    results = score_articles([article])

    assert results[0]["cleaned_text"] == "verarbeitet Titel new_art"


def test_cached_articles_skip_spacy(monkeypatch):
    """Articles with cleaned_text bypass spaCy preprocessing entirely."""
    preprocess_calls: list[str] = []

    def tracking_preprocess(item):
        preprocess_calls.append(item["id"])
        return {**item, "cleaned_text": item["title"], "tokens": [], "lemmas": []}

    monkeypatch.setattr("src.scoring.sentiment.preprocess", tracking_preprocess)
    monkeypatch.setattr("src.scoring.sentiment._get_pipeline", lambda: _fake_pipeline())

    cached   = _make_item("cached_art",   cleaned_text="bereits gesäubert")
    uncached = _make_item("uncached_art", cleaned_text=None)

    results = score_articles([cached, uncached])

    assert "cached_art"   not in preprocess_calls, "spaCy should be skipped for cached articles"
    assert "uncached_art" in preprocess_calls,     "spaCy should run for uncached articles"
    assert results[0]["cleaned_text"] == "bereits gesäubert"


def test_scores_identical_with_and_without_cache(db_path, monkeypatch):
    """topic_scores are identical between runs and cached cleaned_text is never overwritten."""
    from src.scoring.run_nlp import run_nlp_pipeline

    def fake_score_articles(items, **kwargs):
        return [
            {
                **a,
                "cleaned_text": "gesäubert " + a["title"],
                "sentiment_label": "neutral",
                "sentiment_score": 0.6,
                "sentiment_extremity": 0.25,
                "sensationalism_score": 0.1,
            }
            for a in items
        ]

    monkeypatch.setattr("src.scoring.run_nlp.score_articles", fake_score_articles)
    monkeypatch.setattr("src.scoring.run_nlp.compute_framing", lambda *a, **kw: {
        "framing_inconsistency": 0.35,
        "fact_inconsistency": 0.20,
    })
    monkeypatch.setattr("src.scoring.run_nlp.get_trust_score", lambda *a, **kw: 70.0)
    monkeypatch.setattr("src.scoring.run_nlp._domain_from_url", lambda *a: "tagesschau.de")
    monkeypatch.setattr("src.scoring.run_nlp.score_attribution_vagueness", lambda *a, **kw: 0.1)

    # Run 1: all articles are uncached (cleaned_text IS NULL)
    run_nlp_pipeline(db_path=db_path)

    raw_conn = sqlite3.connect(db_path)
    assert all(
        r[0] is not None
        for r in raw_conn.execute("SELECT cleaned_text FROM raw_items").fetchall()
    ), "cleaned_text should be saved after run 1"
    scores1 = raw_conn.execute(
        "SELECT avg_sentiment_extremity, sensationalism_avg, "
        "framing_inconsistency, attribution_vagueness "
        "FROM topic_scores WHERE topic_id = 1"
    ).fetchone()
    # Overwrite cleaned_text with sentinel values so we can detect if run 2 touches them
    raw_conn.execute("UPDATE raw_items SET cleaned_text = 'sentinel_art1' WHERE id = 'art1'")
    raw_conn.execute("UPDATE raw_items SET cleaned_text = 'sentinel_art2' WHERE id = 'art2'")
    raw_conn.commit()
    raw_conn.close()

    # Run 2: all articles are cached (cleaned_text IS NOT NULL)
    run_nlp_pipeline(db_path=db_path)

    raw_conn = sqlite3.connect(db_path)
    ct_after_run2 = {
        r[0]: r[1]
        for r in raw_conn.execute("SELECT id, cleaned_text FROM raw_items").fetchall()
    }
    scores2 = raw_conn.execute(
        "SELECT avg_sentiment_extremity, sensationalism_avg, "
        "framing_inconsistency, attribution_vagueness "
        "FROM topic_scores WHERE topic_id = 1"
    ).fetchone()
    raw_conn.close()

    # cached cleaned_text values must not be overwritten
    assert ct_after_run2["art1"] == "sentinel_art1", "run 2 overwrote cached cleaned_text for art1"
    assert ct_after_run2["art2"] == "sentinel_art2", "run 2 overwrote cached cleaned_text for art2"

    # topic_scores identical between runs
    assert scores1 is not None and scores2 is not None
    for idx, col in enumerate(("avg_sentiment_extremity", "sensationalism_avg",
                                "framing_inconsistency", "attribution_vagueness")):
        v1, v2 = scores1[idx], scores2[idx]
        assert abs((v1 or 0.0) - (v2 or 0.0)) < 1e-4, (
            f"{col} differs between runs: {v1} vs {v2}"
        )
