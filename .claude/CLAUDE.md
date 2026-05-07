# Misinfo Dashboard

Real-time misinformation detection dashboard. Scrapes trending topics from Reddit, YouTube, and DuckDuckGo news, clusters them, scores misinformation risk via NLP, and displays results in Streamlit.

## Stack

- Python 3.11+, SQLite, spaCy, HuggingFace transformers, scikit-learn
- Streamlit + Plotly (frontend), n8n via Docker (automation)
- Optional: Ollama (Mistral 7B), Qdrant, google-lens-python

## Commands

- `streamlit run src/dashboard/app.py` — run dashboard locally
- `python -m pytest tests/ -v` — run all tests
- `python src/scrapers/run_all.py` — run scraper pipeline
- `python src/scoring/compute_scores.py` — recompute risk scores

## Architecture

- `src/scrapers/` — Reddit (PRAW), YouTube (API v3), DuckDuckGo (Crawl4AI)
- `src/nlp/` — spaCy preprocessing, NER, sentiment (HuggingFace), Ollama prompts
- `src/scoring/` — source trust DB, cross-referencing, composite risk formula
- `src/dashboard/` — Streamlit app, Plotly charts, alert logic
- `src/utils/` — DB helpers, clustering (TF-IDF + agglomerative), dedup (RapidFuzz)
- `config/` — API keys template, source trust CSV, thresholds
- `data/raw/` — scraped JSON dumps (gitignored)
- `data/processed/` — clustered + scored output (gitignored)
- `notebooks/` — validation, EDA, scoring evaluation

## Conventions

- Type hints on all function signatures
- Docstrings on public functions (Google style)
- One module = one responsibility. No 500-line files.
- Tests mirror src/ structure: `tests/test_scrapers/`, `tests/test_nlp/`, etc.
- Never commit API keys. Use `.env` + python-dotenv. See `config/.env.template`
- Branch naming: `feature/short-description`, `fix/short-description`
- Commit messages: imperative mood, max 72 chars first line
