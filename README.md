# Hot Topics — Misinformation Risk Dashboard

Real-time dashboard that monitors trending German-language topics across curated RSS feeds and NewsAPI — then scores each topic's misinformation risk using NLP-based analysis.

The dashboard is entirely in German and targets the DACH media landscape.

---

## Features

- **Multi-source pipeline** — aggregates verified journalism from 45+ curated German RSS feeds and NewsAPI into a unified topic model
- **7-signal risk scoring** — each topic gets a composite risk score built from source trustworthiness, sentiment extremity, coverage breadth, framing divergence, sensationalism, attribution vagueness, and fact inconsistency
- **Framing inconsistency** — cosine distance between multilingual sentence embeddings of high-trust vs. low-trust source tiers detects narrative divergence at the NLP level
- **Domain trust resolver** — MBFC-curated CSV → TLD heuristic fallback → default; scores every domain 0–100
- **Interactive Streamlit dashboard** — risk radar, waterfall contribution chart, domain trust bar, per-article signal gauges

---

## How it works

```
RSS feeds (45 sources)  ──┐
NewsAPI (German news)   ──┘  run_all.py   → topics.db (raw + clustered)

topics.db  →  run_nlp.py        → NLP scores per topic (sentiment, framing, attribution…)
           →  compute_scores.py → composite_risk
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
```

> **Note:** Without `NEWSAPI_KEY` you can pass `--no-newsapi` to use the RSS pool only.

### 3. Run the pipeline

```bash
# Step 1 — scrape & cluster
python -m src.scrapers.run_all

# (optional flags)
python -m src.scrapers.run_all --target-topics 5 --articles-per-topic 20
python -m src.scrapers.run_all --no-newsapi   # RSS only (no API quota used)

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
| Topic detail | `?view=topic&topic_id=N` | Risk radar, signal waterfall, domain trust bar, article list |
| Article detail | `?view=article&item_id=X` | Per-article signal gauges (sensationalism, attribution vagueness, clickbait density, caps ratio), full text |

---

<!-- AUTO-GENERATED: scoring formulas, do not edit by hand -->
## Scoring formulas

Topics are scored on **two independent tiers** (Fase 7) instead of one blended number:

**Article risk** (`article_scorer.py`) — per-article score stored in `raw_items.article_risk_score`:
```
article_risk = 0.15 × source_distrust + 0.30 × sentiment_extremity
               + 0.30 × sensationalism + 0.25 × attribution_vagueness
```

**Linguistic-only risk** (`compute_scores.py`) — always computable once `run_nlp` has scored a topic; stored in `topic_scores.linguistic_only_risk`:
```
linguistic_only_risk = 0.55 × avg_article_risk + 0.10 × framing_inconsistency
                       + 0.35 × fact_inconsistency
```

**Evidence-grounded risk** (`compute_scores.py::compute_evidence_signals`) — fraction of RAG-verified claims (Fase 5, `claim_verifications`) that were `refuted` among claims with a definite verdict (`supported` ∪ `refuted`, excluding `not_enough_evidence`). Stored in `topic_scores.evidence_grounded_risk` (`NULL` when no claim has a definite verdict). `topic_scores.evidence_coverage` is the fraction of checked claims that had a definite verdict.

**Combined score** (`compute_scores.py::compute_overall_risk`) — the single sortable/flaggable number stored in `topic_scores.composite_risk`:
```
composite_risk = evidence_grounded_risk  if evidence_coverage > 0.30 (30 %)
                = linguistic_only_risk    otherwise
```

`composite_risk` > 0.50 (50 %) is the misinformation flag threshold (`_MISINFO_THRESHOLD`).

**Overall confidence** (`compute_scores.py::compute_overall_confidence`) — `topic_scores.overall_confidence` (`'high'` / `'medium'` / `'low'`), the average of `evidence_coverage` and the topic's mean Fase 6 `source_reliability.resolve_reliability()` confidence across its article domains: ≥ 0.65 (65 %) → high, ≥ 0.35 (35 %) → medium, else low.

Weights and thresholds are defined once in `src/scoring/weights.py` and this section is generated from that file by `scripts/render_docs.py` — do not hand-edit the numbers above.
<!-- END AUTO-GENERATED: scoring formulas -->

---

## Project structure

```
hot-topics-dashboard/
├── src/
│   ├── scrapers/
│   │   ├── run_all.py              # Orchestrator (RSS → NewsAPI → cluster)
│   │   ├── rss_scraper.py          # 45 curated German RSS feeds
│   │   ├── newsapi_scraper.py      # NewsAPI (German, 100 req/day free)
│   │   ├── google_rss_scraper.py   # Google News RSS fallback
│   │   └── article_fetcher.py      # Full-text fetch for RSS items
│   ├── nlp/
│   │   ├── preprocessor.py         # spaCy tokenisation, cleaning
│   │   ├── ner.py                  # Named entity extraction (PERSON, ORG, LOC)
│   │   ├── sentiment.py            # HuggingFace german-sentiment-bert
│   │   └── keywords.py             # Topic keyword extraction
│   ├── scoring/
│   │   ├── run_nlp.py              # NLP scoring orchestrator
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
| Data ingestion | `feedparser` (RSS), NewsAPI, Crawl4AI |
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
