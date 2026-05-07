# Hot Topic & Misinformation Intelligence Dashboard

Real-time dashboard that monitors trending political and cyber-related topics across Reddit, YouTube, and web news — then scores each topic's misinformation risk using NLP-based analysis.

## How it works

1. **Scrape** — Collects trending content from Reddit, YouTube, and DuckDuckGo news
2. **Cluster** — Groups related articles into unified topics using TF-IDF + agglomerative clustering
3. **Score** — Evaluates each topic for misinformation risk (sentiment, source credibility, cross-references, linguistic markers)
4. **Display** — Streamlit dashboard with interactive charts, risk alerts, and transparent score breakdowns

## Quick start

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/misinfo-dashboard.git
cd misinfo-dashboard
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure API keys
cp config/.env.template .env
# Edit .env with your API keys

# Run the scraper pipeline
python src/scrapers/run_all.py

# Launch the dashboard
streamlit run src/dashboard/app.py
```

## Project structure

```
misinfo-dashboard/
├── .claude/CLAUDE.md       # Claude Code project instructions
├── src/
│   ├── scrapers/           # Reddit, YouTube, DuckDuckGo scrapers
│   ├── nlp/                # spaCy preprocessing, sentiment, NER, Ollama
│   ├── scoring/            # Source trust DB, cross-ref, composite risk score
│   ├── dashboard/          # Streamlit app + Plotly charts
│   └── utils/              # DB helpers, clustering, dedup
├── config/                 # .env template, source trust CSV, thresholds
├── data/                   # Raw scrapes + processed output (gitignored)
├── notebooks/              # Validation, EDA, scoring evaluation
├── tests/                  # Mirrors src/ structure
├── docs/                   # Architecture diagram, presentation
└── requirements.txt
```

## Tech stack

| Layer | Tools |
|-------|-------|
| Data ingestion | PRAW, YouTube Data API v3, Crawl4AI / Playwright |
| NLP | spaCy, HuggingFace Transformers (RoBERTa), Ollama + Mistral 7B |
| Scoring | scikit-learn, RapidFuzz, MediaBiasFactCheck (curated CSV) |
| Frontend | Streamlit, Plotly |
| Automation | n8n (self-hosted via Docker) |
| Storage | SQLite |

## Team

| Role | Focus |
|------|-------|
| Person A | Computational linguistics & NLP (text cleaning, NER, sentiment, Ollama prompts, stylometry) |
| Person B | Data science & AI engineering (scrapers, database, clustering, dashboard, deployment) |

## License

MIT
