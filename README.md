# Hot Topics — Misinformation Risk Dashboard

Real-time misinformation detection dashboard for German-language news. Topics are discovered from the open web via broad search (SearXNG/DDG) with semantic clustering — no curated RSS preselection required — scored for misinformation risk with NLP + RAG-based fact-checking, and displayed in a Streamlit dashboard. A curated RSS/NewsAPI mode is available as a fallback.

---

## Features

- **Broad topic discovery** — no editorial preselection: 7 broad German-news queries hit SearXNG (self-hosted, DDG fallback), results are deduplicated, then grouped into topics via sentence-embedding clustering
- **Curated fallback mode** (`--no-broad-search`) — 45 curated German RSS feeds + NewsAPI for a fully deterministic, editorially-scoped topic pool
- **Two-tier risk scoring** — `linguistic_only_risk` (always computable: source trust, sentiment, sensationalism, attribution vagueness, framing divergence, NLI fact inconsistency) and `evidence_grounded_risk` (fraction of RAG-verified claims refuted), combined into a single `composite_risk`
- **RAG claim verification** — claims extracted per article are checked against Google Fact Check Tools, Wikidata, and web search evidence via a multilingual NLI model (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`)
- **Truth-discovery source reliability** — per-domain reliability posterior built from the accumulated history of claim verdicts, blended with the domain trust prior
- **Domain trust resolver** — MBFC-curated CSV (114 entries) → live signals (Wikidata recognition, OpenPageRank, WHOIS domain age, SPF/DMARC) → Google Safe Browsing hard floor; scores every domain 0–100, cached with a 7-day TTL
- **Run history** — every pipeline execution is a non-destructive snapshot; the dashboard always shows the latest completed run, with full history queryable in SQLite
- **Interactive Streamlit dashboard** — topic ranking, risk radar, signal waterfall, domain trust bar, per-article signal gauges, evidence/claim cards, and a demo mode backed by a static sample DB

---

## How it works

```
Broad-search mode (default):
  SearXNG/DDG (7 broad queries) → dedupe → semantic clustering → topic candidates
    → targeted search_topic() to fill gaps → trafilatura full-text enrich
    → clusters with ≥ 10 articles qualify                          run_all.py

Curated mode (--no-broad-search):
  45 RSS feeds + NewsAPI (German) → topic clustering                run_all.py

topics.db  →  run_nlp.py        → sentiment, framing, attribution, article_risk_score
           →  compute_scores.py → linguistic_only_risk, evidence_grounded_risk, composite_risk
           →  app.py            → Streamlit dashboard
```

`src/orchestrator.py` runs all three steps in a single process (NLP models loaded once) — the faster alternative to invoking each script separately.

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
```

| Variable | Required? | Notes |
|---|---|---|
| `GOOGLE_SAFE_BROWSING_KEY` | Required | Domain trust hard-floor checks |
| `NEWSAPI_KEY` | Only for `--no-broad-search` | Free tier: 100 req/day |
| `SEARXNG_URL` | Optional | Self-hosted SearXNG for broad-search mode; DuckDuckGo is the fallback |
| `OPEN_PAGE_RANK_KEY` | Optional | Improves unknown-domain trust scoring; free 1000 req/day |
| `GOOGLE_FACT_CHECK_API_KEY` | Optional | Improves claim-verification evidence retrieval |
| `GNEWS_API_KEY` | Optional | Alternative to NewsAPI |

To run SearXNG locally: `docker-compose up -d` (starts `hot-topics-searxng` on port 8080; config in `config/searxng/settings.yml`).

### 3. Run the pipeline

```bash
# Single-process (recommended) — broad-search mode is on by default
python -m src.orchestrator --target-topics 10 --articles-per-topic 20

# Or run each step separately:
python -m src.scrapers.run_all              # discovery → clustering → topics.db
python -m src.scoring.run_nlp               # sentiment, framing, attribution scoring
python -m src.scoring.compute_scores        # linguistic_only_risk, evidence_grounded_risk, composite_risk

# Curated RSS + NewsAPI instead of broad search:
python -m src.scrapers.run_all --no-broad-search

# Quick smoke test (fewer topics, faster):
python -m src.scrapers.run_all --target-topics 3 --articles-per-topic 15
```

Typical runtime (10 topics × 20 articles, broad-search mode): ~8 min on the first run (domain resolver makes live network calls for unknown domains), ~3–4 min on subsequent runs once the 7-day domain trust cache is warm.

### 4. Run the dashboard

```bash
streamlit run src/dashboard/app.py
```

A **demo mode** toggle is available in the dashboard settings, backed by a static `data/demo.db` sample so the UI can be explored without running the pipeline first.

---

## Dashboard views

| View | URL param | What it shows |
|------|-----------|---------------|
| Home | `?view=home` | Topic ranking table, sentiment vs. sensationalism scatter, composite risk bar chart, expander with scoring methodology |
| Topic detail | `?view=topic&topic_id=N` | Two-tier risk metrics (linguistic-only, evidence-grounded, coverage, confidence), claim/evidence cards, risk radar, signal waterfall, domain trust bar, article list |
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
│   ├── orchestrator.py              # Single-process entry point (all 3 pipeline steps)
│   ├── scrapers/
│   │   ├── broad_search.py          # SearXNG/DDG discovery, search_topic() gap-fill
│   │   ├── run_all.py               # Pipeline orchestration (broad + curated modes)
│   │   ├── rss_scraper.py           # 45 curated German RSS feeds (--no-broad-search only)
│   │   ├── newsapi_scraper.py       # NewsAPI (German, 100 req/day free)
│   │   ├── google_rss_scraper.py    # Google News RSS (--no-broad-search only)
│   │   ├── youtube_scraper.py       # YouTube scraper (disabled by default)
│   │   └── article_fetcher.py       # trafilatura full-text extraction
│   ├── nlp/
│   │   ├── preprocessor.py          # spaCy tokenisation, cleaning
│   │   ├── ner.py                   # Named entity extraction (de_core_news_lg)
│   │   ├── keywords.py              # Topic keyword extraction
│   │   ├── topic_query.py           # Search query builder
│   │   ├── embeddings.py            # Shared sentence-embedding model loader
│   │   ├── nli.py                   # Shared multilingual NLI pipeline
│   │   └── claim_extractor.py       # Per-article claim extraction (Fase 5)
│   ├── scoring/
│   │   ├── run_nlp.py               # NLP scoring orchestrator
│   │   ├── compute_scores.py        # Two-tier risk aggregator
│   │   ├── source_trust.py          # MBFC CSV loader, coverage metrics
│   │   ├── domain_resolver.py       # Live trust signals + 7-day TTL SQLite cache
│   │   ├── source_lookup.py         # Runtime MBFC lookup + disclaimer generator
│   │   ├── framing.py               # Sentence-embedding framing analysis
│   │   ├── contradiction.py         # NLI-based cross-tier fact_inconsistency
│   │   ├── sentiment.py             # Sentiment extremity scorer
│   │   ├── attribution.py           # Attribution vagueness scorer
│   │   ├── article_scorer.py        # Per-article risk (4 signals)
│   │   ├── evidence_retriever.py    # Claim evidence retrieval (Fase 5)
│   │   ├── claim_verifier.py        # NLI-based claim verification (Fase 5)
│   │   ├── source_reliability.py    # Truth-discovery reliability posterior (Fase 6)
│   │   └── weights.py               # All scoring weights/thresholds (single source of truth)
│   ├── dashboard/
│   │   ├── app.py                   # Streamlit SPA (home / topic / article views)
│   │   └── i18n.py                  # UI string constants
│   └── utils/
│       ├── db.py                    # SQLite schema + helpers, run history
│       ├── models.py                # TypedDicts (RawItem, ScoredTopic…)
│       └── dedup.py                 # RapidFuzz near-duplicate removal
├── config/
│   ├── .env.template                # API key template
│   ├── rss_sources.csv              # 45 curated German RSS feeds
│   ├── source_trust.csv             # MBFC domain trust database (114 entries)
│   └── searxng/settings.yml         # Self-hosted SearXNG config
├── scripts/
│   └── render_docs.py               # Regenerates the "Scoring formulas" section above
├── tests/                           # Mirrors src/ — run with pytest
├── notebooks/                       # EDA, scoring validation, precision/recall
├── data/
│   ├── dashboard.db                 # Full run history + live scores (gitignored)
│   └── demo.db                      # Static sample DB for dashboard demo mode
└── requirements.txt
```

---

## RSS sources (curated fallback mode)

`config/rss_sources.csv` includes 45 curated German-language RSS feeds covering the DACH region:

**High trust (≥ 80):** Tagesschau, DW Deutsch, ZDF heute, NDR, WDR, Süddeutsche Zeitung, FAZ, Die Zeit, NZZ, Der Standard, ORF

**Mid trust (60–79):** Spiegel Online, Stern, Focus, t-online, Heise Online, Golem, Netzpolitik, correctiv.org, Mimikama

**Monitoring tier (< 60):** RT Deutsch, Epoch Times DE (included as reference for low-trust framing comparison)

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Discovery | SearXNG (self-hosted) / DuckDuckGo fallback, `feedparser` (RSS), NewsAPI, `trafilatura` (full-text) |
| NLP | spaCy `de_core_news_lg`, HuggingFace `oliverguhr/german-sentiment-bert` |
| Embeddings / NLI | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |
| Domain trust | MBFC CSV, Wikidata SPARQL, OpenPageRank, WHOIS (`python-whois`), SPF/DMARC (`dnspython`), Google Safe Browsing |
| Scoring | scikit-learn, RapidFuzz |
| Frontend | Streamlit, Plotly |
| Storage | SQLite |

---

## Running tests

```bash
python -m pytest tests/ -v
```

Tests mirror `src/` structure and mock all external API calls — no real network requests are made during testing.

---

## License

MIT
