"""English UI string constants for the Hot Topics Dashboard."""
from __future__ import annotations

# ── app-level ──────────────────────────────────────────────────────────────────
PAGE_TITLE = "Hot Topics Dashboard"
APP_TITLE = "🔍 Hot Topics"
APP_CAPTION = "Misinformation Risk Dashboard"

# ── home view ──────────────────────────────────────────────────────────────────
METRIC_TOPICS_ANALYSED = "Topics Analysed"
METRIC_TOPICS_ANALYSED_HELP = (
    "Total number of identified topic clusters. "
    "Each cluster groups articles covering the same subject."
)
METRIC_HIGH_RISK = "High-Risk Topics"
METRIC_AVG_TRUST = "Avg Source Trust"
METRIC_AVG_TRUST_HELP = (
    "Mean trust score (0–100) across all scored articles, based on MBFC. "
    "≥ 60 = credible · 40–59 = neutral · < 40 = unreliable."
)
METRIC_AVG_RISK = "Avg Composite Risk"
METRIC_AVG_RISK_HELP = (
    "Weighted average across 3 NLP signals. "
    "40 % article risk (source trust, sentiment, sensationalism, attribution) + "
    "35 % framing divergence + 10 % fact inconsistency. "
    "Higher = more misinformation risk."
)

SECTION_TOPIC_RANKING = "Topic Ranking"
LABEL_ARTICLES = "Articles"
LABEL_RISK = "Risk"
LABEL_RELIABILITY = "Reliability"
LABEL_UNSCORED = "unscored"
LABEL_DIV_TOOLTIP = (
    "Narrative Divergence: how strongly Reddit framing differs from verified sources"
)

CAPTION_PIPELINE_MISSING = (
    "No scored topics found at `{db_path}`.\n\n"
    "Run the pipeline first:\n"
    "```\npython src/scrapers/run_all.py\n"
    "python src/scoring/compute_scores.py\n```"
)

CAPTION_ONLY_SCORED = (
    "Only fully scored topics (NLP pipeline completed) are listed here."
)

# ── charts (home) ──────────────────────────────────────────────────────────────
CHART_SENTIMENT_VS_SENS = "Sentiment vs. Sensationalism"
CHART_COMPOSITE_RISK = "Composite Risk by Topic"
CHART_RISK_THRESHOLD_LABEL = "Threshold"
AXIS_SENTIMENT = "Sentiment Extremity"
AXIS_SENSATIONALISM = "Sensationalism"
AXIS_RISK = "Risk"
AXIS_COMPOSITE_RISK = "Composite Risk (0–100 %)"

# ── topic detail ───────────────────────────────────────────────────────────────
BTN_BACK = "← Back"
TOPIC_NOT_FOUND = "Topic {topic_id} not found."
TOPIC_NOT_SCORED = (
    "This topic has not been scored yet. "
    "Run `python src/scoring/compute_scores.py`."
)
CAPTION_COMPOSITE_RISK = "Composite risk: **{risk:.1f} %**"
CAPTION_ARTICLES = "{n} articles"
CAPTION_SCORED_AT = "scored {ts}"

SECTION_SIGNAL_BREAKDOWN = "Signal Breakdown"
SIGNAL_BREAKDOWN_CAPTION = "Hover over a card for a full explanation."

SECTION_SOCIAL_TRACK = "Social Media Track (Reddit)"
SOCIAL_NO_DATA = (
    "No Reddit articles linked to this topic — social risk track unavailable. "
    "Run the pipeline with Reddit enabled to populate this section."
)

METRIC_VERIFIED_RISK = "Verified Risk"
METRIC_VERIFIED_RISK_HELP = "Composite risk from NewsAPI/RSS articles (journalistic sources)."
METRIC_SOCIAL_RISK = "Social Risk"
METRIC_SOCIAL_RISK_HELP = "Composite risk from Reddit posts about this topic."
METRIC_NARRATIVE_DIV = "Narrative Divergence"
METRIC_NARRATIVE_DIV_HELP = (
    "Absolute difference between verified and social risk. "
    "High divergence means Reddit discussions frame the topic very differently "
    "from journalistic sources — a potential misinformation signal."
)

DIVERGENCE_HIGH = "high divergence — indicative of coordinated social amplification."
DIVERGENCE_MED = "moderate divergence — social framing deviates from press coverage."
DIVERGENCE_LOW = "low divergence — social and press coverage largely aligned."
DIVERGENCE_PREFIX = "Narrative Divergence"
DIVERGENCE_VS = "Verified risk {v:.1f} % vs. Social risk {s:.1f} %"

CAPTION_SOCIAL_GRADE = "Social grade:"
CAPTION_SOCIAL_BASED_ON = "Based on Reddit posts about this topic."

SECTION_RISK_RADAR = "Risk Radar"
RADAR_CHART_TITLE = "Risk Radar"
RADAR_CAPTION = (
    "Each axis shows a normalised risk signal (0–100 %). "
    "Larger area = higher misinformation risk. "
    "Hover over a point to see the exact value."
)
RADAR_CATEGORIES: list[str] = [
    "Source Distrust",
    "Sentiment",
    "Framing",
    "Sensationalism",
]

SECTION_DOMAIN_TRUST = "Domain Trust Scores"
DOMAIN_TRUST_CAPTION = (
    "Trust scores from Media Bias/Fact Check (MBFC). "
    "🟢 ≥ 60 = credible · 🟠 40–59 = neutral · 🔴 < 40 = unreliable. "
    "High spread between green and red is a misinformation signal."
)
DOMAIN_TRUST_NO_DATA = "No domain data available."

SECTION_ARTICLES = "Articles"
ARTICLES_NONE = "No articles found for this topic."

# ── signal names ───────────────────────────────────────────────────────────────
SIGNAL_NAMES: dict[str, str] = {
    "Article Risk":          "📊 Article Risk",
    "Source Distrust":       "🏛️ Source Distrust",
    "Sentiment Extremity":   "😤 Sentiment Extremity",
    "Framing Divergence":    "🔀 Framing Divergence",
    "Sensationalism":        "📢 Sensationalism",
    "Attribution Vagueness": "⚠️ Attribution Vagueness",
    "Fact Inconsistency":    "📋 Fact Inconsistency",
}

SIGNAL_DETAIL_LABELS: dict[str, str] = {
    "Source Distrust":     "Avg trust {val:.1f}",
    "Sentiment Extremity": "Signal {val:.1f} %",
    "Framing Divergence":  "Signal {val:.1f} %",
    "Sensationalism":      "Signal {val:.1f} %",
}

# ── signal tooltips ────────────────────────────────────────────────────────────
EXPANDER_ARTICLE_RISK_DETAIL = "📊 Article Risk Breakdown (4 sub-signals)"

SIGNAL_TOOLTIPS: dict[str, str] = {
    "Article Risk": (
        "Average per-article risk across all articles in this topic cluster. "
        "Bundles 4 article-level sub-signals: "
        "source distrust (30 %), sentiment extremity (25 %), sensationalism (25 %), "
        "attribution vagueness (20 %). Weight in composite risk: 40 %."
    ),
    "Source Distrust": (
        "Fraction of coverage from low-trust sources: score = (1 − avg_trust / 100). "
        "Based on MBFC trust scores (0–100) per domain. "
        "🟢 ≥ 60 credible · 🟠 40–59 neutral · 🔴 < 40 unreliable. "
        "Weight in article risk: 30 % (effective composite weight: 12 %)."
    ),
    "Sentiment Extremity": (
        "Average emotional intensity of articles — how far sentiment deviates from neutral. "
        "High = articles use strongly polarised, emotionally charged language. "
        "Measured with oliverguhr/german-sentiment-bert. "
        "Weight in article risk: 25 % (effective composite weight: 10 %)."
    ),
    "Framing Divergence": (
        "How differently articles in this topic cluster frame the same story. "
        "Measured as pairwise cosine distance between sentence-transformer embeddings. "
        "High = articles tell very different versions of the same topic — "
        "a key misinformation signal. Weight in composite risk: 35 %."
    ),
    "Sensationalism": (
        "Density of ALL-CAPS words, exclamation marks, and clickbait phrases across articles. "
        "High = heavy sensationalist rhetoric. "
        "Weight in article risk: 25 % (effective composite weight: 10 %)."
    ),
    "Attribution Vagueness": (
        "Use of vague sourcing language ('some say', 'sources claim', 'experts believe'). "
        "High = most claims lack specific, named attribution — a misinformation signal. "
        "Weight in article risk: 20 % (effective composite weight: 8 %)."
    ),
    "Fact Inconsistency": (
        "Inconsistency of named entities (persons, places, organisations) "
        "across articles in the topic cluster. "
        "High = articles cite very different facts — a misinformation signal. "
        "Weight in composite risk: 10 %."
    ),
}

# ── breakdown chart ────────────────────────────────────────────────────────────
BREAKDOWN_CHART_TITLE = "Risk Contribution per Signal"
BREAKDOWN_CHART_CAPTION = (
    "The stacked bar shows the weighted contribution of each signal to the composite risk. "
    "Longer segments = greater influence on the risk score."
)
BREAKDOWN_SIGNAL_COLOURS: list[str] = [
    "#3498db",  # Article Risk
    "#e74c3c",  # Framing Divergence
    "#e67e22",  # Fact Inconsistency
]

# ── article view ───────────────────────────────────────────────────────────────
SECTION_SIGNAL_ANALYSIS = "Signal Analysis"
ARTICLE_ANALYSIS_LABEL = "Article Analysis"
GAUGE_SENSATIONALISM = "Sensationalism"
GAUGE_ATTRIBUTION = "Attribution Vagueness"
GAUGE_CLICKBAIT = "Clickbait Patterns"
GAUGE_CAPS = "ALL-CAPS Density"

ARTICLE_FULL_TEXT = "Full Article Text"
ARTICLE_SUMMARY = "Summary"
ARTICLE_DESCRIPTION = "Description"
ARTICLE_NO_TEXT = "*No text available.*"

# ── expander explanations ──────────────────────────────────────────────────────
EXPANDER_HOW_RISK = "ℹ️ How is the risk score calculated?"
EXPANDER_HOW_RISK_TEXT = """\
The **composite risk** (0–100 %) is a weighted sum of 3 signals:

| Signal | Weight | Description |
|---|---|---|
| 📊 Article Risk | 40 % | Avg per-article risk (bundles 4 sub-signals — see below) |
| 🔀 Framing Divergence | 35 % | How inconsistently articles frame the same topic |
| 📋 Fact Inconsistency | 10 % | Named-entity divergence across articles |

**Article Risk** (40 %) is itself composed of:

| Sub-Signal | Weight (article) | Effective weight |
|---|---|---|
| 🏛️ Source Distrust | 30 % | 12 % |
| 😤 Sentiment Extremity | 25 % | 10 % |
| 📢 Sensationalism | 25 % | 10 % |
| ⚠️ Attribution Vagueness | 20 % | 8 % |

A risk ≥ 50 % is flagged as a potential misinformation signal.
"""

EXPANDER_SOCIAL_TRACK = "ℹ️ What is the Social Media Track?"
EXPANDER_SOCIAL_TRACK_TEXT = """\
The **Social Media Track** scores Reddit posts separately from journalistic sources.

- **Verified Risk** — computed from NewsAPI/RSS/Google News articles
- **Social Risk** — computed from Reddit posts
- **Narrative Divergence** — |verified − social risk|

High divergence (≥ 30 %) can indicate coordinated social amplification or **narrative hijacking**:
Reddit discussions frame the topic fundamentally differently from the press.
"""

EXPANDER_DOMAIN_TRUST = "ℹ️ Where do trust scores come from?"
EXPANDER_DOMAIN_TRUST_TEXT = """\
Trust scores are resolved in this order:

1. **Google Safe Browsing** — domains flagged as malicious receive a hard floor of 5/100, bypassing all other signals
2. **Media Bias/Fact Check (MBFC)** — manually curated trust scores for 100+ German and international news outlets; the primary source for well-known domains
3. **Live scoring for unknown domains** — four signals are combined and cached for 7 days:
   - **Recognised news organisation** (Wikidata lookup via P856 + news category)
   - **Domain authority** (OpenPageRank API — optional; improves score accuracy when configured)
   - **Domain age** (WHOIS lookup, capped at 15 years)
   - **DNS authentication** (SPF + DMARC records)

   Unknown domain scores are capped at 82 — they cannot exceed top-rated MBFC outlets.

Thresholds: 🟢 ≥ 60 credible · 🟠 40–59 neutral · 🔴 < 40 unreliable
"""

EXPANDER_RADAR = "ℹ️ What does the Risk Radar show?"
EXPANDER_RADAR_TEXT = """\
The radar chart visualises 4 risk signals simultaneously:

- **Source Distrust** — how much coverage comes from low-trust sources (1 − avg_trust / 100)
- **Sentiment** — average emotional intensity of articles
- **Framing** — how inconsistently articles frame the topic (cosine distance between embeddings)
- **Sensationalism** — density of ALL-CAPS, exclamation marks, and clickbait phrases

Large area = high overall risk. A sharp spike on a single axis shows which signal is driving the risk.
"""
