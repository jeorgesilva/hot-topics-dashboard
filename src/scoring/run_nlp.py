"""NLP pipeline runner: score all topics with Person-A signals.

Reads articles for each topic from the DB, runs sentiment + sensationalism
scoring, framing inconsistency, and attribution vagueness, then writes
the aggregated signals into topic_scores. Call compute_scores.py afterward
to compute the composite risk.

Usage:
    python -m src.scoring.run_nlp
    python -m src.scoring.run_nlp --db-path data/dashboard.db
"""
from __future__ import annotations

import argparse
import logging
import statistics
from datetime import datetime, timezone

from src.scoring.article_scorer import score_article as compute_article_risk
from src.scoring.attribution import score_attribution_vagueness
from src.scoring.framing import compute_framing
from src.scoring.sentiment import score_articles
from src.scoring.source_trust import get_trust_score, _domain_from_url
from src.utils.db import get_connection, init_db

logger = logging.getLogger(__name__)

_NLP_MODELS: dict | None = None


def load_nlp_models() -> dict:
    """Load the three NLP models, caching them for the lifetime of the process.

    Second call returns the same cached instances — no repeated disk I/O or
    GPU warm-up. Call this once at process start (e.g. from the orchestrator)
    so the per-topic scoring functions skip their own cold-start penalty.

    Returns:
        dict with keys "spacy", "sentiment", "embeddings" corresponding to
        de_core_news_lg, german-sentiment-bert, and paraphrase-multilingual-MiniLM.
    """
    global _NLP_MODELS
    if _NLP_MODELS is None:
        from src.nlp.preprocessor import _get_nlp
        from src.scoring.framing import _get_model
        from src.scoring.sentiment import _get_pipeline

        _NLP_MODELS = {
            "spacy": _get_nlp(),
            "sentiment": _get_pipeline(),
            "embeddings": _get_model(),
        }
    return _NLP_MODELS


def _load_topic_articles(
    conn,
    topic_id: int,
    *,
    platform: str | None = None,
    exclude_platform: str | None = None,
) -> list[dict]:
    query = """
        SELECT ri.id, ri.title, ri.description, ri.body_text,
               ri.source, ri.url,
               ri.platform, ri.timestamp, ri.engagement_json,
               ri.cleaned_text
        FROM topic_sources ts
        JOIN raw_items ri ON ri.id = ts.item_id
        WHERE ts.topic_id = ?
    """
    params: list = [topic_id]
    if platform:
        query += " AND ri.platform = ?"
        params.append(platform)
    if exclude_platform:
        query += " AND ri.platform != ?"
        params.append(exclude_platform)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def run_nlp_pipeline(db_path=None) -> dict:
    """Score all topics with NLP signals and persist to topic_scores."""
    conn = init_db(db_path)
    topic_ids = [
        r["id"] for r in conn.execute(
            """
            SELECT id FROM topics
            WHERE COALESCE(run_id, -1) = COALESCE((SELECT MAX(run_id) FROM topics), -1)
            """
        ).fetchall()
    ]

    if not topic_ids:
        logger.warning("No topics found in DB.")
        conn.close()
        return {"topics_scored": 0}

    logger.info("Running NLP pipeline on %d topics...", len(topic_ids))

    from src.nlp.preprocessor import _get_nlp
    nlp_model = _get_nlp()

    scored = 0

    for topic_id in topic_ids:
        # ── verified track (all sources) ─────────────────────────────────────
        verified = _load_topic_articles(conn, topic_id)
        if not verified:
            logger.info("  topic %d: no verified articles, skipping", topic_id)
            continue

        cached_v   = [a for a in verified if     a.get("cleaned_text")]
        uncached_v = [a for a in verified if not a.get("cleaned_text")]
        logger.info(
            "cache NLP: %d/%d artigos já processados (tópico %d)",
            len(cached_v), len(verified), topic_id,
        )
        logger.info("  topic %d: scoring %d verified articles...", topic_id, len(verified))

        scored_verified = score_articles(verified)

        avg_sentiment_extremity = statistics.mean(
            a["sentiment_extremity"] for a in scored_verified
        )
        sensationalism_avg = statistics.mean(
            a["sensationalism_score"] for a in scored_verified
        )
        trust_map = {
            a["source"]: get_trust_score(_domain_from_url(a["url"]) or a["source"], conn=conn)
            for a in scored_verified
        }
        framing_result = compute_framing(scored_verified, trust_map)
        framing_inconsistency = framing_result["framing_inconsistency"]
        fact_inconsistency = framing_result["fact_inconsistency"]

        # attribution per article first so we can compute article_risk_score;
        # the topic mean is derived from the same list.
        per_article_attribution_v = [
            score_attribution_vagueness(a.get("cleaned_text") or a["title"], nlp=nlp_model)
            for a in scored_verified
        ]
        attribution_vagueness = statistics.mean(per_article_attribution_v)

        # article_risk_score — combine per-article signals into a single [0,1] score.
        for a, attr in zip(scored_verified, per_article_attribution_v):
            trust = trust_map.get(a["source"], 50.0)
            a["article_risk_score"] = compute_article_risk(
                trust_score=trust,
                sentiment_extremity=a["sentiment_extremity"],
                sensationalism_score=a["sensationalism_score"],
                attribution_vagueness=attr,
            )

        avg_article_risk = statistics.mean(
            a["article_risk_score"] for a in scored_verified
        )

        uncached_v_ids = {a["id"] for a in uncached_v}
        for a in scored_verified:
            if a.get("cleaned_text") and a["id"] in uncached_v_ids:
                conn.execute(
                    "UPDATE raw_items SET cleaned_text = ? WHERE id = ?",
                    (a["cleaned_text"], a["id"]),
                )

        conn.executemany(
            "UPDATE raw_items SET article_risk_score = ? WHERE id = ?",
            [
                (a["article_risk_score"], a["id"])
                for a in scored_verified
                if a.get("article_risk_score") is not None
            ],
        )

        conn.execute(
            """
            INSERT INTO topic_scores (topic_id, avg_sentiment_extremity,
                sensationalism_avg, framing_inconsistency, fact_inconsistency,
                attribution_vagueness, avg_article_risk)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id) DO UPDATE SET
                avg_sentiment_extremity = excluded.avg_sentiment_extremity,
                sensationalism_avg      = excluded.sensationalism_avg,
                framing_inconsistency   = excluded.framing_inconsistency,
                fact_inconsistency      = excluded.fact_inconsistency,
                attribution_vagueness   = excluded.attribution_vagueness,
                avg_article_risk        = excluded.avg_article_risk
            """,
            (
                topic_id,
                round(avg_sentiment_extremity, 4),
                round(sensationalism_avg, 4),
                round(framing_inconsistency, 4),
                round(fact_inconsistency, 4),
                round(attribution_vagueness, 4),
                round(avg_article_risk, 4),
            ),
        )
        logger.info(
            "    verified ✓ sentiment_ext=%.3f  sensationalism=%.3f  framing=%.3f  attribution=%.3f",
            avg_sentiment_extremity, sensationalism_avg, framing_inconsistency, attribution_vagueness,
        )

        conn.commit()
        scored += 1

    conn.close()
    logger.info("NLP pipeline complete: %d topics scored.", scored)
    return {"topics_scored": scored}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NLP scoring for all topics.")
    parser.add_argument("--db-path", default=None, help="SQLite DB path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_nlp_pipeline(args.db_path)
    print(f"Topics NLP-scored: {result['topics_scored']}")


if __name__ == "__main__":
    main()
