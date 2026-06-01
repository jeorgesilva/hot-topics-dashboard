# Hot Topics — Misinformation Risk Dashboard

Real-time dashboard that monitors trending German-language topics across Reddit, curated RSS feeds, and NewsAPI — then scores each topic's misinformation risk using NLP-based analysis.

The dashboard is entirely in German and targets the DACH media landscape.

---

## Features

- **Dual-source pipeline** — aggregates verified journalism (45+ curated German RSS feeds) and social media (Reddit) into a unified topic model
- **7-signal risk scoring** — each topic gets a composite risk score built from source trustworthiness, sentiment extremity, coverage breadth, framing divergence, sensationalism, attribution vagueness, and fact inconsistency
- **Two-track analysis** — `composite_risk` (journalistic sources only) and `social_risk` (Reddit only) are scored separately, and `narrative_divergence` = |composite − social| surfaces topics where Reddit amplifies or distorts the journalistic framing
- **Framing inconsistency** — cosine distance between multilingual sentence embeddings of high-trust vs. low-trust source tiers detects narrative divergence at the NLP level
- **Domain trust resolver** — MBFC-curated CSV → TLD heuristic fallback → default; scores every domain 0–100
- **Interactive Streamlit dashboard** — risk radar, waterfall contribution chart, domain trust bar, per-article signal gauges, social-media track panel

---

## How it works

```
RSS feeds (45 sources)  ──┐
Reddit (5 subreddits)   ──┤  run_all.py   → topics.db (raw + clustered)
NewsAPI (German news)   ──┘

topics.db  →  run_nlp.py        → NLP scores per topic (sentiment, framing, attribution…)
           →  compute_scores.py → composite_risk, social_risk, narrative_divergence
           →  app.py            → Streamlit dashboard
```

---

## Quick start

### 1. Install dependencies

```bash
git clone https://github.com/jeorgesilva/hot-topics-dashboard.git
cd hot-topics-dashboard
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download de_core_news_lg   # German NLP model (required)
```

### 2. Configure API keys

```bash
cp config/.env.template .env
# Edit .env — minimum required: NEWSAPI_KEY (free tier at https://newsapi.org/register)
# Optional: REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET for live Reddit scraping
```

> **Note:** Without a Reddit API key the pipeline still runs using the RSS pool.
> Without `NEWSAPI_KEY` you can pass `--no-newsapi` to use the RSS pool only.

### 3. Run the pipeline

```bash
# Step 1 — scrape & cluster
python -m src.scrapers.run_all

# (optional flags)
python -m src.scrapers.run_all --target-topics 5 --articles-per-topic 20
python -m src.scrapers.run_all --no-newsapi   # RSS + Reddit only (no API quota used)

# Step 2 — NLP scoring
python -m src.scoring.run_nlp

# Step 3 — composite scores
python -m src.scoring.compute_scores

# Step 4 — dashboard
streamlit run src/dashboard/app.py
```

---

## Dashboard views

| View | URL param | What it shows |
|------|-----------|---------------|
| Home | `?view=home` | Topic ranking table, sentiment vs. sensationalism scatter, composite risk bar chart, expander with scoring methodology |
| Topic detail | `?view=topic&topic_id=N` | Risk radar, signal waterfall, social-media track (Reddit vs. journalism), domain trust bar, article list |
| Article detail | `?view=article&item_id=X` | Per-article signal gauges (sensationalism, attribution vagueness, clickbait density, caps ratio), full text |

---

## Risk signals

| Signal | Weight | Description |
|--------|--------|-------------|
| Source Distrust | 25 % | Share of coverage from low-trust domains (MBFC score < 40) |
| Sentiment Extremity | 20 % | Average emotional deviation from neutral across articles |
| Low Coverage | 20 % | Ratio of incredible domains among all covering domains |
| Framing Divergence | 15 % | Cosine distance between high-trust and low-trust tier embeddings |
| Sensationalism | 10 % | Density of caps, exclamation marks, clickbait phrases |
| Attribution Vagueness | 5 % | Frequency of vague sourcing ("experts say", "sources claim") |
| Fact Inconsistency | 5 % | Jaccard distance between named-entity sets of the two tiers |

A `composite_risk` ≥ 50 % is flagged as a potential misinformation signal.

---

## Project structure

```
hot-topics-dashboard/
├── src/
│   ├── scrapers/
│   │   ├── run_all.py              # Orchestrator (RSS → NewsAPI → Reddit → cluster)
│   │   ├── rss_scraper.py          # 45 curated German RSS feeds
│   │   ├── newsapi_scraper.py      # NewsAPI (German, 100 req/day free)
│   │   ├── reddit_scraper.py       # PRAW scraper (r/de, r/germany, r/nachrichten…)
│   │   ├── google_rss_scraper.py   # Google News RSS fallback
│   │   └── article_fetcher.py      # Full-text fetch for RSS items
│   ├── nlp/
│   │   ├── preprocessor.py         # spaCy tokenisation, cleaning
│   │   ├── ner.py                  # Named entity extraction (PERSON, ORG, LOC)
│   │   ├── sentiment.py            # HuggingFace german-sentiment-bert
│   │   └── keywords.py             # Topic keyword extraction
│   ├── scoring/
│   │   ├── run_nlp.py              # NLP scoring orchestrator (verified + social tracks)
│   │   ├── compute_scores.py       # Composite risk aggregator
│   │   ├── source_trust.py         # MBFC CSV loader, domain trust scorer, coverage metrics
│   │   ├── domain_resolver.py      # TLD heuristic fallback + SQLite cache
│   │   ├── framing.py              # Sentence embedding framing analysis
│   │   ├── sentiment.py            # Sentiment extremity scorer
│   │   └── attribution.py          # Attribution vagueness scorer
│   ├── dashboard/
│   │   ├── app.py                  # Streamlit SPA (home / topic / article views)
│   │   └── i18n.py                 # German UI string constants
│   └── utils/
│       ├── db.py                   # SQLite schema + helpers
│       ├── models.py               # TypedDicts (RawItem, ScoredTopic…)
│       ├── clustering.py           # TF-IDF + agglomerative clustering
│       └── dedup.py                # RapidFuzz near-duplicate removal
├── config/
│   ├── .env.template               # API key template
│   ├── rss_sources.csv             # 45 German RSS feeds with trust scores
│   └── source_trust.csv            # MBFC domain trust database
├── tests/                          # Mirrors src/ — run with pytest
├── notebooks/                      # EDA, scoring validation, precision/recall
├── data/
│   ├── raw/                        # Scraped JSON dumps (gitignored)
│   └── processed/                  # topics.db — clustered + scored output (gitignored)
└── requirements.txt
```

---

## RSS sources

The pipeline includes 45 curated German-language RSS feeds covering the DACH region:

**High trust (≥ 80):** Tagesschau, DW Deutsch, ZDF heute, NDR, WDR, Süddeutsche Zeitung, FAZ, Die Zeit, NZZ, Der Standard, ORF

**Mid trust (60–79):** Spiegel Online, Stern, Focus, t-online, Heise Online, Golem, Netzpolitik, correctiv.org, Mimikama

**Monitoring tier (< 60):** RT Deutsch, Epoch Times DE (included as reference for low-trust framing comparison)

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Data ingestion | `feedparser` (RSS), PRAW (Reddit), NewsAPI, Crawl4AI |
| NLP | spaCy `de_core_news_lg`, HuggingFace `oliverguhr/german-sentiment-bert` |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Scoring | scikit-learn, RapidFuzz, MBFC CSV |
| Frontend | Streamlit, Plotly |
| Storage | SQLite |

---

## Running tests

```bash
python -m pytest tests/ -v
```

Tests mirror `src/` structure and use in-memory SQLite fixtures. No real API calls are made during testing.

---

## License

MIT
