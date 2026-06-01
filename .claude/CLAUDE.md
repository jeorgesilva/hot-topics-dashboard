# Hot Topics Dashboard

Real-time misinformation detection dashboard for German-language news. Scrapes trending topics from curated RSS feeds (45 sources), Reddit (5 subreddits), and optionally NewsAPI — clusters them, scores misinformation risk via NLP, and displays results in Streamlit.

## Stack

- Python 3.11+, SQLite, spaCy `de_core_news_lg`, HuggingFace transformers, scikit-learn
- Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) for framing analysis
- Streamlit + Plotly (frontend), German UI via `src/dashboard/i18n.py`

## Commands

- `streamlit run src/dashboard/app.py` — run dashboard locally (port 8501)
- `python -m pytest tests/ -v` — run all tests

Full pipeline defaults: **10 topics × 40 articles** (20 verified + 20 Reddit) per run.

Full pipeline (run in order):
1. `python -m src.scrapers.run_all` — Google RSS (50 seeds) → RSS pool + Reddit → 10 topics × 20 verified articles
2. `python -m src.scoring.run_nlp` — sentiment, framing, attribution NLP scoring per topic (two tracks: verified + social)
3. `python -m src.scoring.compute_scores` — composite_risk, social_risk, narrative_divergence per topic

Without NewsAPI quota (RSS + Reddit pool only):
```
python -m src.scrapers.run_all --no-newsapi
```

Quick smoke-test (fewer topics, faster):
```
python -m src.scrapers.run_all --no-newsapi --target-topics 3 --articles-per-topic 15
```

First-time setup (after `pip install -r requirements.txt`):
- `python -m spacy download de_core_news_lg` — German NLP model (required)
- Copy `config/.env.template` to `.env` and fill in `NEWSAPI_KEY`
  (free tier: https://newsapi.org/register — 100 requests/day; optional if using --no-newsapi)

## Article counts explained

- `--articles-per-topic N` controls **verified** articles (RSS/NewsAPI) per topic. Default: 20.
- `--reddit-per-topic N` controls **Reddit** posts per topic. Default: 20.
- Total per topic = verified + Reddit = **40** with defaults.
- Topics that don't reach `--articles-per-topic` verified articles are dropped.
- With `--no-newsapi`, the RSS pool (~420 articles) is the only verified source; broad topics (Iran, Merz) qualify easily, niche topics may not.

## Architecture

- `src/scrapers/` — RSS (45 curated German feeds), Reddit (PRAW), Google News RSS, NewsAPI, article_fetcher (trafilatura full-text)
- `src/nlp/` — spaCy preprocessing, NER (de_core_news_lg), sentiment (german-sentiment-bert)
- `src/scoring/` — source trust (MBFC CSV + TLD heuristic), framing (sentence embeddings), composite risk formula
- `src/dashboard/` — Streamlit SPA (home/topic/article views), i18n.py (German strings), Plotly charts
- `src/utils/` — DB helpers (SQLite), RawItem/ScoredTopic TypedDicts, dedup (RapidFuzz)
- `config/` — `.env.template`, `rss_sources.csv` (45 feeds), `source_trust.csv` (MBFC)
- `data/dashboard.db` — live topics + scores (gitignored)
- `notebooks/` — validation, EDA, scoring evaluation

## Conventions

- Type hints on all function signatures
- Docstrings on public functions (Google style)
- One module = one responsibility. No 500-line files.
- Tests mirror src/ structure: `tests/test_scrapers/`, `tests/test_nlp/`, etc.
- Never commit API keys. Use `.env` + python-dotenv. See `config/.env.template`
- Branch naming: `feature/short-description`, `fix/short-description`
- Commit messages: imperative mood, max 72 chars first line
