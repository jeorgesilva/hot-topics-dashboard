# NewsRadar

Real-time misinformation detection dashboard for German-language news. Discovers trending topics from the open web via broad search (SearXNG/DDG) with semantic clustering — no curated RSS preselection — scores misinformation risk via NLP, and displays results in Streamlit.

## Stack

- Python 3.11+, SQLite, spaCy `de_core_news_lg`, HuggingFace transformers, scikit-learn
- Sentence Transformers (`paraphrase-multilingual-mpnet-base-v2`) for framing analysis and topic clustering
- Streamlit + Plotly (frontend), English UI via `src/dashboard/i18n.py`

## Commands

- `streamlit run src/dashboard/app.py` — run dashboard locally (port 8501)
- `python -m pytest tests/ -v` — run all tests

Full pipeline defaults: **10 topics × 20 articles** per run.

Full pipeline (run in order):
1. `python -m src.scrapers.run_all` — broad discovery → semantic clustering → 10 topics × 20 articles
2. `python -m src.scoring.run_nlp` — sentiment, framing, attribution NLP scoring per topic; also computes `article_risk_score` via `article_scorer`
3. `python -m src.scoring.compute_scores` — composite_risk per topic

**Faster alternative** — single process, NLP models loaded once:
```
python -m src.orchestrator [--no-broad-search] [--no-newsapi] [--target-topics N] [--articles-per-topic N]
```

Typical runtime (10 topics × 20 articles, broad-search mode): **~8 min first run** (domain resolver makes live network calls for unknown domains), **~3–4 min on subsequent runs** once `domain_trust_cache` is warm (7-day TTL). Breakdown: ~2 min scraping, ~6 min NLP (sentiment + framing + domain resolver), <5 s composite scoring.

Broad search is **on by default**. To use curated RSS + NewsAPI instead:
```
python -m src.scrapers.run_all --no-broad-search
```

Quick smoke-test (fewer topics, faster):
```
python -m src.scrapers.run_all --target-topics 3 --articles-per-topic 15
```

First-time setup (after `pip install -r requirements.txt`):
- `python -m spacy download de_core_news_lg` — German NLP model (required)
- Copy `config/.env.template` to `.env` and fill in values:
  - `NEWSAPI_KEY` (free tier: https://newsapi.org/register — 100 req/day; only used in `--no-broad-search` mode)
  - `GOOGLE_SAFE_BROWSING_KEY` — required for domain trust hard-floor checks (Google Cloud Console → Safe Browsing API v4)
  - `OPEN_PAGE_RANK_KEY` — optional, improves domain authority signal (https://www.domcop.com/openpagerank/signup, free 1000 req/day)
- Optionally set `SEARXNG_URL` in `.env` for a self-hosted SearXNG instance (DDG is the fallback)
- To run SearXNG locally: `docker-compose up -d` — starts `hot-topics-searxng` on port 8080; config lives in `config/searxng/settings.yml`

## Domain trust scoring

`domain_resolver.py` scores domains absent from `config/source_trust.csv` (MBFC-curated) using four live signals:

| Signal | Source | With OPR | Without OPR |
|---|---|---|---|
| Recognised news org | Wikidata SPARQL (P856 + news type) | 0.35 | 0.45 |
| Domain authority | OpenPageRank API (optional) | 0.30 | — |
| Domain age | WHOIS via `python-whois` (capped at 15 y) | 0.20 | 0.30 |
| DNS authentication | SPF + DMARC records via `dnspython` | 0.15 | 0.25 |

Google Safe Browsing is checked first — flagged domains short-circuit to a score of 5 regardless of other signals.

Score range for unknown domains: **30–82** (cannot exceed top MBFC-rated outlets). Results are cached in `domain_trust_cache` (SQLite) with a **7-day TTL** so each domain is re-evaluated weekly.

## Broad discovery pipeline (default)

When `--broad-search` is active (the default), the scraper runs **Option A** — topics emerge from the open web with no editorial preselection:

1. **Discovery** — 7 broad German-news queries hit SearXNG/DDG → ~200–350 deduplicated articles (URL + title + snippet); search engines are asked to pre-filter to the last ~30 days (`time_range=month` / `df=m`)
2. **Clustering** — sentence-transformer embeddings + agglomerative clustering (cosine distance ≤ 0.35) groups articles into topic candidates; the article closest to each cluster centroid becomes the topic label
3. **Supplement** — clusters below `articles_per_topic` get a targeted `search_topic(label)` call to fill gaps
4. **Enrich** — trafilatura extracts full body text and real publication date; articles without body text are dropped; body text is capped at 50 000 chars to prevent spaCy OOM; paywall subscription CTAs and scrambled/obfuscated text (e.g. heise.de-style) are detected and dropped
5. **Recency filter** — `_filter_by_age` in `run_all.py` drops articles whose publication date is older than 14 calendar days; applied after enrichment in both broad-search and curated modes
6. **Qualify** — clusters with ≥ 10 articles with body text qualify; others are dropped

Tuning constants in `run_all.py`: `_CLUSTER_DISTANCE_THRESHOLD` (default 0.35) and `_CLUSTER_MIN_SIZE` (default 5).

## Article counts explained

- `--articles-per-topic N` controls target articles per topic. Default: 20.
- Topics that don't reach `min_qualify` articles with extractable body text are dropped.
- In broad-search mode `min_qualify` defaults to 10 (lower than 20 because not every URL yields extractable text).
- In `--no-broad-search --no-newsapi` mode, the curated RSS pool (~420 articles) is the only source.

## Scoring formulas

**Article risk** (`article_scorer.py`) — per-article score stored in `raw_items.article_risk_score`:
```
article_risk = 0.30 × (1 - trust/100) + 0.25 × sentiment_extremity
             + 0.25 × sensationalism  + 0.20 × attribution_vagueness
```

**Composite risk** (`compute_scores.py`) — per-topic score stored in `topic_scores.composite_risk`:
```
composite_risk = 0.40 × avg_article_risk + 0.35 × framing_inconsistency
               + 0.15 × (1 - coverage_ratio) + 0.10 × fact_inconsistency
```

`composite_risk > 0.50` is the misinformation flag threshold (`_MISINFO_THRESHOLD`).

## Architecture

- `src/orchestrator.py` — unified single-process entry point (runs all three pipeline steps with NLP models loaded once)
- `src/scrapers/` — `broad_search` (SearXNG/DDG discovery + `discover_articles_broad`), `google_rss_scraper` (used only in `--no-broad-search` mode), `rss_scraper` (45 curated German feeds, `--no-broad-search` only), `newsapi_scraper`, `youtube_scraper`, `article_fetcher` (trafilatura full-text + publication date extraction; paywall/scrambled-text detection), `run_all` (pipeline orchestration; `_cluster_into_topics` for broad mode, `_fetch_articles_for_topic` for curated mode; `_filter_by_age` 14-day recency filter)
- `src/nlp/` — spaCy preprocessing (`preprocessor`), NER (`ner`, de_core_news_lg), keyword extraction (`keywords`), search query builder (`topic_query`)
- `src/scoring/` — `source_trust` (MBFC CSV + coverage metrics), `source_lookup` (runtime MBFC lookup + per-article disclaimer generation), `domain_resolver` (live signals + 7-day TTL SQLite cache), `sentiment` (german-sentiment-bert), `framing` (sentence embeddings), `attribution` (vagueness patterns), `article_scorer` (per-article risk: 4 signals), `compute_scores` (composite_risk, social_risk, narrative_divergence)
- `src/dashboard/` — Streamlit SPA (home/topic/article views), i18n.py (English strings), Plotly charts; demo mode switchable via "▶ Try Demo" button (loads `data/demo.db`)
- `src/utils/` — DB helpers (SQLite), RawItem/ScoredTopic TypedDicts, dedup (RapidFuzz)
- `config/` — `.env.template`, `rss_sources.csv` (45 feeds, used only in `--no-broad-search` mode), `source_trust.csv` (MBFC), `searxng/settings.yml`
- `data/dashboard.db` — full run history + live scores (gitignored)
- `notebooks/` — validation, EDA, scoring evaluation

## Run history (Option B)

Each pipeline execution is an independent, non-destructive snapshot. The DB accumulates history across runs; no data is ever deleted.

### Schema

```
pipeline_runs  id, started_at, completed_at, status ('running'|'completed')
topics         id (AUTOINCREMENT), label, created_at, item_count, run_id → pipeline_runs
topic_sources  topic_id → topics, item_id → raw_items
raw_items      global dedup store — shared across all runs (INSERT OR IGNORE)
               + cleaned_text, tokens_json, lemmas_json, entities_json, keywords_json,
                 cluster_id, article_risk_score
topic_scores   topic_id → topics, all NLP + composite signals
               verified track: avg_trust, trust_variance, coverage_breadth, coverage_ratio,
                 avg_sentiment_extremity, sensationalism_avg, framing_inconsistency,
                 attribution_vagueness, fact_inconsistency, avg_article_risk, composite_risk
               social track: social_avg_trust, social_coverage_ratio,
                 social_avg_sentiment_extremity, social_sensationalism_avg,
                 social_framing_inconsistency, social_attribution_vagueness,
                 social_fact_inconsistency, social_risk
               cross-track: narrative_divergence
domain_trust_cache  domain, trust_score, method, cached_at (7-day TTL)
```

- `start_run(conn)` / `complete_run(conn, run_id)` in `src/utils/db.py` bookend each pipeline call.
- Scoring steps (`run_nlp`, `compute_scores`, `source_trust.score_coverage`) automatically target the **latest run** via `COALESCE(run_id, -1) = COALESCE((SELECT MAX(run_id) FROM topics), -1)`.
- The dashboard always displays the latest **completed** run (`WHERE status = 'completed'`); the home caption shows `Run #N · updated <timestamp> UTC`.
- Because `raw_items` is a global dedup store, the same article URL across two runs occupies one row — only topic metadata and scores are duplicated per run.

### History queries

```sql
-- How many times was "Angela Merkel" mentioned across all runs?
SELECT COUNT(*) FROM raw_items WHERE entities_json LIKE '%Angela Merkel%';

-- Composite risk for a topic across runs (requires matching by label):
SELECT pr.id AS run, pr.completed_at, ts.composite_risk
FROM topics t
JOIN pipeline_runs pr ON pr.id = t.run_id
JOIN topic_scores ts ON ts.topic_id = t.id
WHERE t.label LIKE '%Merkel%'
ORDER BY pr.id;
```

## Conventions

- Type hints on all function signatures
- Docstrings on public functions (Google style)
- One module = one responsibility. No 500-line files.
- Tests mirror src/ structure: `tests/test_scrapers/`, `tests/test_nlp/`, etc.
- Never commit API keys. Use `.env` + python-dotenv. See `config/.env.template`
- Branch naming: `feat/short-description`, `fix/short-description`
- Commit messages: imperative mood, max 72 chars first line
