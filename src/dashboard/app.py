"""Hot Topics Misinformation Dashboard — 3-view SPA routing.

Views:
    ?view=home              — daily topic ranking (default)
    ?view=topic&topic_id=N  — topic detail with radar + article list
    ?view=article&item_id=X — per-article signal analysis

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running via `streamlit run src/dashboard/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.scoring.attribution import score_attribution_vagueness
from src.scoring.compute_scores import _MISINFO_THRESHOLD, _WEIGHTS, grade_topic
from src.scoring.sentiment import _clickbait_score, _sensationalism
from src.scoring.source_trust import _domain_from_url, get_trust_score
from src.utils.db import init_db

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "data" / "dashboard.db"

_HIGH_RISK = _MISINFO_THRESHOLD

_GRADE_COLOUR: dict[str, str] = {
    "A": "#2ecc71",
    "B": "#82b74b",
    "C": "#f1c40f",
    "D": "#e67e22",
    "F": "#e74c3c",
}

_RISK_COLORSCALE = [
    [0.0, "#2ecc71"],
    [0.4, "#f1c40f"],
    [0.7, "#e67e22"],
    [1.0, "#e74c3c"],
]

PLATFORM_ICONS: dict[str, str] = {
    "reddit":      "🔴",
    "youtube":     "▶️",
    "newsapi":     "📰",
    "google_news": "🌐",
    "duckduckgo":  "🦆",
}


# ── data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_scored_topics(db_path: str) -> pd.DataFrame:
    """Load all topics with scores. Returns empty DataFrame if unavailable."""
    try:
        conn = init_db(db_path)
        rows = conn.execute(
            """
            SELECT
                t.id            AS topic_id,
                t.label         AS topic,
                t.item_count    AS articles,
                t.created_at,
                ts.avg_sentiment_extremity  AS sentiment_extremity,
                ts.sensationalism_avg       AS sensationalism,
                ts.framing_inconsistency,
                ts.avg_trust,
                ts.trust_variance,
                ts.coverage_breadth,
                ts.coverage_ratio,
                ts.attribution_vagueness,
                ts.fact_inconsistency,
                ts.composite_risk,
                ts.computed_at,
                ts.social_avg_trust,
                ts.social_coverage_ratio,
                ts.social_avg_sentiment_extremity,
                ts.social_sensationalism_avg,
                ts.social_framing_inconsistency,
                ts.social_attribution_vagueness,
                ts.social_fact_inconsistency,
                ts.social_risk,
                ts.narrative_divergence,
                (SELECT GROUP_CONCAT(DISTINCT ri.platform)
                 FROM topic_sources ts2
                 JOIN raw_items ri ON ri.id = ts2.item_id
                 WHERE ts2.topic_id = t.id) AS platforms,
                (SELECT GROUP_CONCAT(ri.keywords_json, '||')
                 FROM topic_sources ts2
                 JOIN raw_items ri ON ri.id = ts2.item_id
                 WHERE ts2.topic_id = t.id
                   AND ri.keywords_json IS NOT NULL) AS keywords_raw
            FROM topics t
            LEFT JOIN topic_scores ts ON t.id = ts.topic_id
            ORDER BY ts.composite_risk DESC NULLS LAST, t.created_at DESC
            """
        ).fetchall()
        conn.close()
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    scored = df["composite_risk"].notna()
    df.loc[scored, "grade"] = df.loc[scored, "composite_risk"].apply(grade_topic)
    df.loc[scored, "reliability_pct"] = ((1.0 - df.loc[scored, "composite_risk"]) * 100).round(1)
    df["coverage_ratio_pct"] = (df["coverage_ratio"].fillna(0) * 100).round(1)
    df["bubble_size"] = df["coverage_breadth"].fillna(0).clip(lower=1)
    return df


@st.cache_data(ttl=60)
def load_topic_articles(db_path: str, topic_id: int) -> list[dict]:
    """Load articles for a topic with per-article trust and sensationalism."""
    try:
        conn = init_db(db_path)
        rows = conn.execute(
            """
            SELECT ri.id, ri.title, ri.description, ri.source, ri.url,
                   ri.platform, ri.timestamp, ri.engagement_json, ri.cleaned_text
            FROM topic_sources ts
            JOIN raw_items ri ON ri.id = ts.item_id
            WHERE ts.topic_id = ?
            ORDER BY ri.timestamp DESC
            """,
            (topic_id,),
        ).fetchall()
        conn.close()
    except Exception:
        return []

    result = []
    for row in rows:
        d = dict(row)
        try:
            d["engagement"] = json.loads(d.pop("engagement_json", "{}") or "{}")
        except Exception:
            d["engagement"] = {}
        domain = _domain_from_url(d.get("url", "")) or d.get("source", "")
        d["trust_score"] = get_trust_score(domain)
        text = d.get("cleaned_text") or d.get("title", "")
        d["sensationalism_score"] = _sensationalism(text)
        result.append(d)
    return result


@st.cache_data(ttl=60)
def load_article(db_path: str, item_id: str) -> dict | None:
    """Load a single article by ID, with its topic context."""
    try:
        conn = init_db(db_path)
        row = conn.execute(
            """
            SELECT ri.*,
                   (SELECT ts.topic_id
                    FROM topic_sources ts WHERE ts.item_id = ri.id LIMIT 1) AS topic_id,
                   (SELECT t.label
                    FROM topics t
                    JOIN topic_sources ts2 ON ts2.topic_id = t.id
                    WHERE ts2.item_id = ri.id LIMIT 1) AS topic_label
            FROM raw_items ri
            WHERE ri.id = ?
            """,
            (item_id,),
        ).fetchone()
        conn.close()
    except Exception:
        return None

    if not row:
        return None

    d = dict(row)
    try:
        d["engagement"] = json.loads(d.pop("engagement_json", "{}") or "{}")
    except Exception:
        d["engagement"] = {}
    return d


# ── helpers ───────────────────────────────────────────────────────────────────

def _truncate(label: str, n: int = 55) -> str:
    return label if len(label) <= n else label[:n] + "…"


def _parse_keywords(keywords_raw: str | None, n: int = 5) -> list[str]:
    """Flatten and deduplicate keywords from a '||'-joined list of JSON arrays."""
    if not keywords_raw:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for chunk in keywords_raw.split("||"):
        try:
            for kw in json.loads(chunk):
                key = str(kw).lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    result.append(str(kw).strip())
        except Exception:
            pass
    return result[:n]


def _platform_icons(platforms_str: str | None) -> str:
    if not platforms_str:
        return ""
    parts = [p.strip() for p in platforms_str.split(",")]
    return " ".join(PLATFORM_ICONS.get(p, "🔗") for p in parts if p)


def _grade_badge_html(grade: str) -> str:
    colour = _GRADE_COLOUR.get(grade, "#999")
    return (
        f"<span style='background:{colour};color:#fff;padding:3px 10px;"
        f"border-radius:4px;font-weight:bold;font-size:1.0em'>{grade}</span>"
    )


def _risk_badge_html(risk: float) -> str:
    colour = "#e74c3c" if risk >= _HIGH_RISK else "#e67e22" if risk >= 0.40 else "#2ecc71"
    return (
        f"<span style='background:{colour};color:#fff;padding:2px 8px;"
        f"border-radius:4px;font-size:0.85em'>{risk:.3f}</span>"
    )


# ── home view ─────────────────────────────────────────────────────────────────

def render_home(df: pd.DataFrame, db_path: str) -> None:
    col_title, col_settings = st.columns([6, 1])
    with col_title:
        st.title("🔍 Hot Topics")
        st.caption("Misinformation risk dashboard")
    with col_settings:
        with st.expander("⚙️"):
            new_path = st.text_input("Database", value=db_path, label_visibility="collapsed")
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.session_state["db_path"] = new_path
                st.rerun()

    if df.empty:
        st.warning(
            f"No scored topics found at `{db_path}`.\n\n"
            "Run the pipeline first:\n"
            "```\npython src/scrapers/run_all.py\n"
            "python src/scoring/compute_scores.py\n```"
        )
        return

    scored = df[df["composite_risk"].notna()]
    flagged = scored[scored["composite_risk"] >= _HIGH_RISK]

    if not flagged.empty:
        lines = "  \n".join(
            f"- **{r['topic']}** &nbsp; {_grade_badge_html(str(r.get('grade','?')))} "
            f"risk {r['composite_risk']:.3f}"
            for _, r in flagged.iterrows()
        )
        st.error(
            f"**{len(flagged)} high-risk topic(s) detected** (risk ≥ {_HIGH_RISK}):\n\n{lines}",
            icon="🚨",
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Topics analysed", len(df),
        help="Total number of unique topic clusters identified in the scraped articles. "
             "Each cluster groups articles about the same subject.",
    )
    c2.metric(
        "High-risk topics", int(len(flagged)),
        help=f"Topics whose composite risk score exceeds {_HIGH_RISK} (50%). "
             "These are likely misinformation vectors based on source trust, sentiment, "
             "framing divergence and sensationalism signals.",
    )
    c3.metric(
        "Avg source trust",
        f"{scored['avg_trust'].mean():.1f} / 100" if not scored.empty else "—",
        help="Mean trust score (0–100) across all scored articles, based on Media Bias/Fact Check "
             "(MBFC) ratings. Scores ≥ 60 = credible, 40–59 = neutral, < 40 = unreliable.",
    )
    c4.metric(
        "Avg composite risk",
        f"{scored['composite_risk'].mean():.1%}" if not scored.empty else "—",
        help="Weighted average of 7 NLP and coverage signals across all scored topics. "
             "Formula: 25% source distrust + 20% sentiment + 20% low coverage + "
             "15% framing divergence + 10% sensationalism + 5% attribution vagueness + "
             "5% fact inconsistency. Higher % = more misinformation risk.",
    )

    st.markdown("---")
    st.subheader("Topic Ranking")
    for _, row in df.iterrows():
        _render_topic_card(row)

    if not scored.empty:
        st.markdown("---")
        col_scatter, col_bar = st.columns(2)
        with col_scatter:
            _render_scatter(scored)
        with col_bar:
            _render_risk_bar(scored)


def _render_topic_card(row: pd.Series) -> None:
    topic_id = row["topic_id"]
    topic = row["topic"]
    articles = int(row.get("articles", 0) or 0)
    risk_raw = row.get("composite_risk")
    risk = None if pd.isna(risk_raw) else float(risk_raw)
    grade_raw = row.get("grade")
    grade = "?" if pd.isna(grade_raw) else str(grade_raw)
    rel_raw = row.get("reliability_pct")
    reliability_pct = 50.0 if pd.isna(rel_raw) else float(rel_raw)
    platforms_str = str(row.get("platforms", "") or "")
    grade_colour = _GRADE_COLOUR.get(grade, "#999")
    icons = _platform_icons(platforms_str)
    risk_str = _risk_badge_html(risk) if risk is not None else "<em style='color:#aaa'>unscored</em>"

    div_raw = row.get("narrative_divergence")
    div_val = None if pd.isna(div_raw) else float(div_raw)
    div_badge = ""
    if div_val is not None:
        div_colour = "#e74c3c" if div_val >= 0.3 else "#e67e22" if div_val >= 0.15 else "#2ecc71"
        div_badge = (
            f"<span title='Narrative divergence: how much Reddit framing differs from verified sources' "
            f"style='background:{div_colour};color:#fff;padding:2px 7px;"
            f"border-radius:4px;font-size:0.78em;margin-left:6px;cursor:help'>"
            f"Δ {div_val:.2f}</span>"
        )

    keywords = _parse_keywords(row.get("keywords_raw"))
    kw_section = "".join(
        f"<span style='background:#1a1a1a;color:#ccc;border:1px solid #444;"
        f"border-radius:3px;padding:1px 6px;font-size:0.72em;margin-right:4px'>{kw}</span>"
        for kw in keywords
    )
    kw_row = f"<div style='margin-top:5px'>{kw_section}</div>" if kw_section else ""
    bar_inner = f"<div style='background:{grade_colour};height:100%;width:{reliability_pct:.1f}%'></div>"
    bar = f"<div style='background:#333;border-radius:4px;height:5px;overflow:hidden'>{bar_inner}</div>"
    rel_label = f"<span style='font-size:0.75em;color:#aaa'>Reliability {reliability_pct:.1f}%</span>"
    title_row = (
        f"{_grade_badge_html(grade)}"
        f"<strong style='font-size:1.0em;color:#fff;margin-left:8px'>{_truncate(topic, 70)}</strong>"
        f"{div_badge}"
        f"<span style='color:#aaa;font-size:0.85em;margin-left:8px'>{articles} articles &nbsp;{icons}</span>"
        f"<span style='margin-left:auto'>Risk: {risk_str}</span>"
    )
    card_html = (
        f"<div style='border-left:4px solid {grade_colour};padding:10px 16px;"
        f"margin-bottom:6px;background:#000;border-radius:0 6px 6px 0'>"
        f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>{title_row}</div>"
        f"{kw_row}"
        f"<div style='margin-top:8px'>{bar}{rel_label}</div>"
        f"</div>"
    )

    col_card, col_nav = st.columns([11, 1])
    with col_card:
        st.markdown(card_html, unsafe_allow_html=True)
    with col_nav:
        if st.button("→", key=f"nav_t_{topic_id}", help="View topic"):
            st.query_params["view"] = "topic"
            st.query_params["topic_id"] = str(topic_id)
            st.rerun()


def _render_scatter(df: pd.DataFrame) -> None:
    needed = ["sentiment_extremity", "sensationalism", "composite_risk"]
    plot_df = df.dropna(subset=needed).copy()
    if plot_df.empty:
        return
    plot_df["label"] = plot_df["topic"].apply(lambda t: _truncate(t, 40))
    plot_df["grade"] = plot_df["composite_risk"].apply(grade_topic)
    fig = px.scatter(
        plot_df,
        x="sentiment_extremity",
        y="sensationalism",
        color="composite_risk",
        color_continuous_scale=_RISK_COLORSCALE,
        range_color=[0.0, 1.0],
        size="articles",
        hover_name="label",
        hover_data={"grade": True, "composite_risk": ":.2f", "articles": True},
        labels={
            "sentiment_extremity": "Sentiment Extremity",
            "sensationalism": "Sensationalism",
            "composite_risk": "Risk",
        },
        title="Sentiment vs Sensationalism",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=20), height=320)
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_bar(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("composite_risk", ascending=True).copy()
    grades = plot_df.get("grade", pd.Series(["?"] * len(plot_df), index=plot_df.index))
    fig = go.Figure(go.Bar(
        x=plot_df["composite_risk"],
        y=plot_df["topic"].apply(lambda t: _truncate(t, 40)),
        orientation="h",
        marker_color=[_GRADE_COLOUR.get(str(g), "#999") for g in grades],
        text=grades,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Risk: %{x:.4f}<br>Grade: %{text}<extra></extra>",
    ))
    fig.add_vline(
        x=_HIGH_RISK, line_dash="dash", line_color="#e74c3c",
        annotation_text=f"Threshold ({_HIGH_RISK})",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Composite Risk by Topic",
        xaxis=dict(title="Composite risk (0–100%)", range=[0, 1.05]),
        yaxis_title=None,
        margin=dict(l=0, r=50, t=40, b=20),
        height=max(280, len(plot_df) * 40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Only topics that have been fully scored (NLP pipeline completed) appear here.")


# ── topic detail view ─────────────────────────────────────────────────────────

def render_topic(topic_id: int, db_path: str) -> None:
    def _back():
        st.query_params.clear()
        st.rerun()

    df = load_scored_topics(db_path)
    topic_rows = df[df["topic_id"] == topic_id] if not df.empty else pd.DataFrame()

    col_back, col_hdr = st.columns([1, 9])
    with col_back:
        if st.button("← Back", key="back_home"):
            _back()

    if topic_rows.empty:
        with col_hdr:
            st.error(f"Topic {topic_id} not found.")
        return

    row = topic_rows.iloc[0]
    grade = str(row.get("grade", "?"))
    risk = row.get("composite_risk")
    articles = int(row.get("articles", 0) or 0)
    icons = _platform_icons(str(row.get("platforms", "") or ""))
    computed_at = str(row.get("computed_at", ""))[:19].replace("T", " ")

    with col_hdr:
        st.markdown(
            f"## {_grade_badge_html(grade)} &nbsp; {row['topic']}",
            unsafe_allow_html=True,
        )
        if risk is not None:
            st.caption(
                f"Composite risk: **{risk * 100:.1f}%** &nbsp;·&nbsp; "
                f"{articles} articles &nbsp;·&nbsp; {icons} &nbsp;·&nbsp; "
                f"scored {computed_at}"
            )

    st.markdown("---")

    if risk is None:
        st.info("This topic has not been scored yet. Run `python src/scoring/compute_scores.py`.")
        return

    _render_score_breakdown(row)
    st.markdown("---")

    _render_social_track(row)
    st.markdown("---")

    col_radar, col_domain = st.columns(2)
    with col_radar:
        _render_radar(row)
    with col_domain:
        _render_domain_trust_bar(db_path, topic_id)

    st.markdown("---")
    st.subheader("Articles")
    articles_data = load_topic_articles(db_path, topic_id)
    if not articles_data:
        st.info("No articles found for this topic.")
    else:
        for a in articles_data:
            _render_article_row(a, topic_id)


_SIGNAL_TOOLTIPS: dict[str, str] = {
    "Source Distrust": (
        "Measures how much of this topic's coverage comes from low-trust sources. "
        "Weight: 25% of composite risk. "
        "High % = most articles come from unreliable outlets. "
        "Based on MBFC trust scores (0–100) per domain."
    ),
    "Sentiment Extremity": (
        "Average emotional intensity of articles — how far sentiment deviates from neutral. "
        "Weight: 20% of composite risk. "
        "High % = articles use strongly polarised, emotionally charged language. "
        "Computed by a RoBERTa sentiment model (|positive − negative| probability)."
    ),
    "Low Coverage": (
        "Fraction of the topic's coverage that comes from non-credible domains. "
        "Weight: 20% of composite risk. "
        "High % = story only covered by low-trust outlets, a classic misinformation pattern."
    ),
    "Framing Divergence": (
        "How differently high-trust vs low-trust sources frame the same topic. "
        "Weight: 15% of composite risk. "
        "Measured as cosine distance between MiniLM embeddings of the two source tiers. "
        "High % = credible and unreliable sources tell very different stories."
    ),
    "Sensationalism": (
        "Density of ALL-CAPS words, exclamation marks, loaded terms (e.g. 'bombshell', "
        "'shocking') and clickbait patterns across all articles. "
        "Weight: 10% of composite risk. "
        "High % = strong sensationalist rhetoric."
    ),
}


def _render_score_breakdown(row: pd.Series) -> None:
    st.subheader("Signal Breakdown")
    st.caption("Hover over each card for a full explanation of the signal.")
    risk = float(row.get("composite_risk", 0) or 0)
    avg_trust = float(row.get("avg_trust", 50) or 50)
    sentiment = float(row.get("sentiment_extremity", 0) or 0)
    coverage_ratio = float(row.get("coverage_ratio", 0) or 0)
    framing = float(row.get("framing_inconsistency", 0) or 0)
    sensationalism = float(row.get("sensationalism", 0) or 0)

    signals = [
        (
            "🏛️ Source Distrust",
            _WEIGHTS["avg_trust"] * (1.0 - avg_trust / 100.0),
            _WEIGHTS["avg_trust"],
            f"avg trust {avg_trust:.1f}%",
            _SIGNAL_TOOLTIPS["Source Distrust"],
        ),
        (
            "😤 Sentiment Extremity",
            _WEIGHTS["avg_sentiment_extremity"] * sentiment,
            _WEIGHTS["avg_sentiment_extremity"],
            f"signal {sentiment * 100:.1f}%",
            _SIGNAL_TOOLTIPS["Sentiment Extremity"],
        ),
        (
            "📡 Low Coverage",
            _WEIGHTS["coverage_ratio"] * (1.0 - coverage_ratio),
            _WEIGHTS["coverage_ratio"],
            f"{coverage_ratio * 100:.1f}% credible domains",
            _SIGNAL_TOOLTIPS["Low Coverage"],
        ),
        (
            "🔀 Framing Divergence",
            _WEIGHTS["framing_inconsistency"] * framing,
            _WEIGHTS["framing_inconsistency"],
            f"signal {framing * 100:.1f}%",
            _SIGNAL_TOOLTIPS["Framing Divergence"],
        ),
        (
            "📢 Sensationalism",
            _WEIGHTS["sensationalism_avg"] * sensationalism,
            _WEIGHTS["sensationalism_avg"],
            f"signal {sensationalism * 100:.1f}%",
            _SIGNAL_TOOLTIPS["Sensationalism"],
        ),
    ]

    cols = st.columns(5)
    for col, (label, contribution, weight, detail, tooltip) in zip(cols, signals):
        share_pct = (contribution / risk * 100) if risk > 0 else 0.0
        c = "#e74c3c" if contribution > 0.10 else "#e67e22" if contribution > 0.05 else "#2ecc71"
        col.markdown(
            f"""<div title="{tooltip}" style='border:1px solid #dee2e6;border-radius:8px;
            padding:12px;text-align:center;min-height:130px;cursor:help'>
              <div style='font-size:0.78em;color:#555;margin-bottom:4px'>{label}</div>
              <div style='font-size:1.6em;font-weight:bold;color:{c}'>{contribution * 100:.1f}%</div>
              <div style='font-size:0.75em;color:#888'>{detail}</div>
              <div style='font-size:0.68em;color:#aaa;margin-top:4px'>
                weight {weight * 100:.0f}% · {share_pct:.0f}% of total risk
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_social_track(row: pd.Series) -> None:
    """Render the social (Reddit) risk track and narrative divergence panel."""
    social_risk_raw = row.get("social_risk")
    social_risk = None if pd.isna(social_risk_raw) else float(social_risk_raw)
    div_raw = row.get("narrative_divergence")
    div_val = None if pd.isna(div_raw) else float(div_raw)

    verified_risk_raw = row.get("composite_risk")
    verified_risk = 0.0 if verified_risk_raw is None or pd.isna(verified_risk_raw) else float(verified_risk_raw)
    st.subheader("Social Media Track (Reddit)")

    if social_risk is None:
        st.info(
            "No Reddit articles were linked to this topic — social risk track unavailable. "
            "Re-run the pipeline with Reddit enabled to populate this section."
        )
        return

    social_grade = grade_topic(social_risk)
    social_grade_colour = _GRADE_COLOUR.get(social_grade, "#999")
    div_colour = "#e74c3c" if (div_val or 0) >= 0.3 else "#e67e22" if (div_val or 0) >= 0.15 else "#2ecc71"

    col_v, col_s, col_d = st.columns(3)
    col_v.metric(
        "Verified Risk",
        f"{verified_risk * 100:.1f}%",
        help="Composite risk computed from NewsAPI/RSS articles (journalistic sources).",
    )
    col_s.metric(
        "Social Risk",
        f"{social_risk * 100:.1f}%",
        delta=f"{(social_risk - verified_risk) * 100:+.1f}% vs verified",
        delta_color="inverse",
        help="Composite risk computed from Reddit posts for this topic.",
    )
    if div_val is not None:
        col_d.metric(
            "Narrative Divergence",
            f"{div_val * 100:.1f}%",
            help=(
                "Absolute gap between verified and social risk scores. "
                "High divergence means Reddit discussions frame this topic very "
                "differently from journalistic sources — a potential misinformation signal."
            ),
        )

    social_avg_trust = float(row.get("social_avg_trust") or 50)
    social_sentiment = float(row.get("social_avg_sentiment_extremity") or 0)
    social_coverage = float(row.get("social_coverage_ratio") or 0)
    social_framing = float(row.get("social_framing_inconsistency") or 0)
    social_sensationalism = float(row.get("social_sensationalism_avg") or 0)

    signals = [
        ("🏛️ Source Distrust",     _WEIGHTS["avg_trust"] * (1.0 - social_avg_trust / 100.0),     f"avg trust {social_avg_trust:.1f}", _SIGNAL_TOOLTIPS["Source Distrust"]),
        ("😤 Sentiment Extremity", _WEIGHTS["avg_sentiment_extremity"] * social_sentiment,         f"signal {social_sentiment * 100:.1f}%", _SIGNAL_TOOLTIPS["Sentiment Extremity"]),
        ("📡 Low Coverage",        _WEIGHTS["coverage_ratio"] * (1.0 - social_coverage),           f"{social_coverage * 100:.1f}% credible", _SIGNAL_TOOLTIPS["Low Coverage"]),
        ("🔀 Framing Divergence",  _WEIGHTS["framing_inconsistency"] * social_framing,             f"signal {social_framing * 100:.1f}%", _SIGNAL_TOOLTIPS["Framing Divergence"]),
        ("📢 Sensationalism",      _WEIGHTS["sensationalism_avg"] * social_sensationalism,         f"signal {social_sensationalism * 100:.1f}%", _SIGNAL_TOOLTIPS["Sensationalism"]),
    ]

    cols = st.columns(5)
    for col, (label, contribution, detail, tooltip) in zip(cols, signals):
        share_pct = (contribution / social_risk * 100) if social_risk > 0 else 0.0
        c = "#e74c3c" if contribution > 0.10 else "#e67e22" if contribution > 0.05 else "#2ecc71"
        col.markdown(
            f"""<div title="{tooltip}" style='border:1px solid #dee2e6;border-radius:8px;
            padding:12px;text-align:center;min-height:110px;cursor:help'>
              <div style='font-size:0.78em;color:#555;margin-bottom:4px'>{label}</div>
              <div style='font-size:1.6em;font-weight:bold;color:{c}'>{contribution * 100:.1f}%</div>
              <div style='font-size:0.75em;color:#888'>{detail}</div>
              <div style='font-size:0.68em;color:#aaa;margin-top:4px'>{share_pct:.0f}% of social risk</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.caption(
        f"Social grade: {_grade_badge_html(social_grade)} &nbsp; "
        f"Based on Reddit posts for this topic.",
        unsafe_allow_html=True,
    )

    if div_val is not None:
        st.markdown(
            f"<div style='margin-top:12px;padding:10px 14px;border-left:4px solid {div_colour};"
            f"background:#0a0a0a;border-radius:0 6px 6px 0'>"
            f"<strong style='color:{div_colour}'>Narrative Divergence: {div_val * 100:.1f}%</strong>"
            f"<span style='color:#aaa;font-size:0.85em;margin-left:10px'>"
            f"Verified risk {verified_risk * 100:.1f}% vs Social risk {social_risk * 100:.1f}%"
            + (" — high divergence suggests coordinated social amplification." if div_val >= 0.3
               else " — moderate divergence, social framing differs from press coverage." if div_val >= 0.15
               else " — low divergence, social and press coverage broadly aligned.")
            + "</span></div>",
            unsafe_allow_html=True,
        )


def _render_radar(row: pd.Series) -> None:
    avg_trust = float(row.get("avg_trust", 50) or 50)
    sentiment = float(row.get("sentiment_extremity", 0) or 0)
    coverage_ratio = float(row.get("coverage_ratio", 0) or 0)
    framing = float(row.get("framing_inconsistency", 0) or 0)
    sensationalism = float(row.get("sensationalism", 0) or 0)

    categories = [
        "Source Distrust",
        "Sentiment",
        "Low Coverage",
        "Framing",
        "Sensationalism",
    ]
    values = [
        1.0 - avg_trust / 100.0,
        sentiment,
        1.0 - coverage_ratio,
        framing,
        sensationalism,
    ]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.2)",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=30, r=30, t=50, b=30),
        height=300,
        title=dict(text="Risk Radar", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each axis shows a risk signal normalised to 0–100%. "
        "A larger filled area means higher overall misinformation risk. "
        "Hover over a point to see its exact value."
    )


def _render_domain_trust_bar(db_path: str, topic_id: int) -> None:
    articles = load_topic_articles(db_path, topic_id)
    if not articles:
        st.info("No domain data available.")
        return

    seen: dict[str, float] = {}
    for a in articles:
        domain = _domain_from_url(a.get("url", "")) or a.get("source", "unknown")
        if domain and domain not in seen:
            seen[domain] = get_trust_score(domain)

    if not seen:
        return

    domain_df = (
        pd.DataFrame([{"domain": d, "trust": s} for d, s in seen.items()])
        .sort_values("trust", ascending=True)
        .tail(15)
    )
    colours = [
        "#2ecc71" if s >= 60 else "#e67e22" if s >= 40 else "#e74c3c"
        for s in domain_df["trust"]
    ]
    fig = go.Figure(go.Bar(
        x=domain_df["trust"],
        y=domain_df["domain"],
        orientation="h",
        marker_color=colours,
        hovertemplate="%{y}: %{x:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Domain Trust Scores", x=0.5),
        xaxis=dict(title="Trust score (0–100)", range=[0, 100]),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=0, r=10, t=50, b=20),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Trust scores from Media Bias/Fact Check (MBFC). "
        "🟢 ≥ 60 = credible · 🟠 40–59 = neutral · 🔴 < 40 = unreliable. "
        "High divergence between green and red bars is a misinformation signal."
    )


def _render_article_row(article: dict, topic_id: int) -> None:
    platform = article.get("platform", "")
    icon = PLATFORM_ICONS.get(platform, "🔗")
    title = article.get("title", "Unknown")
    source = article.get("source", "")
    url = article.get("url", "#")
    trust = float(article.get("trust_score", 50))
    sens = float(article.get("sensationalism_score", 0))
    item_id = article.get("id", "")
    trust_colour = "#2ecc71" if trust >= 60 else "#e67e22" if trust >= 40 else "#e74c3c"

    col_icon, col_content, col_nav = st.columns([1, 10, 1])
    with col_icon:
        st.markdown(
            f"<div style='font-size:1.4em;text-align:center;padding-top:8px'>{icon}</div>",
            unsafe_allow_html=True,
        )
    with col_content:
        st.markdown(
            f"**[{_truncate(title, 85)}]({url})**  \n"
            f"<span style='font-size:0.85em;color:#666'>"
            f"{source}"
            f" &nbsp;·&nbsp; <span style='color:{trust_colour}'>trust {trust:.0f}</span>"
            f" &nbsp;·&nbsp; sensationalism {sens:.2f}"
            f"</span>",
            unsafe_allow_html=True,
        )
    with col_nav:
        if item_id and st.button("→", key=f"nav_a_{item_id}", help="Analyse article"):
            st.query_params["view"] = "article"
            st.query_params["item_id"] = item_id
            st.query_params["topic_id"] = str(topic_id)
            st.rerun()


# ── article analysis view ─────────────────────────────────────────────────────

def render_article(item_id: str, db_path: str) -> None:
    article = load_article(db_path, item_id)

    topic_id = article.get("topic_id") if article else st.query_params.get("topic_id")
    topic_label = article.get("topic_label", "Topic") if article else "Topic"

    col_home, col_s1, col_topic, col_s2, col_art = st.columns([2, 0.4, 4, 0.4, 5])
    with col_home:
        if st.button("🏠 Home", key="bc_home"):
            st.query_params.clear()
            st.rerun()
    with col_s1:
        st.markdown("<span style='color:#aaa'>›</span>", unsafe_allow_html=True)
    with col_topic:
        btn_label = _truncate(str(topic_label), 32)
        if topic_id and st.button(btn_label, key="bc_topic"):
            st.query_params["view"] = "topic"
            st.query_params["topic_id"] = str(topic_id)
            st.rerun()
        elif not topic_id:
            st.markdown(btn_label)
    with col_s2:
        st.markdown("<span style='color:#aaa'>›</span>", unsafe_allow_html=True)
    with col_art:
        st.markdown("**Article Analysis**")

    st.markdown("---")

    if article is None:
        st.error(f"Article `{item_id}` not found in the database.")
        return

    platform = article.get("platform", "")
    icon = PLATFORM_ICONS.get(platform, "🔗")
    title = article.get("title", "Unknown")
    source = article.get("source", "")
    url = article.get("url", "#")
    timestamp = str(article.get("timestamp", ""))[:10]
    description = article.get("description", "")
    cleaned_text = article.get("cleaned_text") or article.get("title", "")

    st.markdown(f"## {icon} [{title}]({url})")
    st.caption(f"**{source}** &nbsp;·&nbsp; {timestamp} &nbsp;·&nbsp; {platform}")

    with st.expander("Description", expanded=True):
        st.write(cleaned_text or description or "*No text available.*")

    st.markdown("---")
    st.subheader("Signal Analysis")

    sens_score = _sensationalism(cleaned_text)
    click_score = _clickbait_score(cleaned_text)
    attr_score = score_attribution_vagueness(cleaned_text)

    words = cleaned_text.split()
    long_words = [w for w in words if len(w) > 3]
    caps_score = min(
        sum(1 for w in long_words if w.isupper()) / len(long_words) if long_words else 0.0,
        1.0,
    )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    _render_gauge(col1, "Sensationalism", sens_score)
    _render_gauge(col2, "Attribution Vagueness", attr_score)
    _render_gauge(col3, "Clickbait Patterns", click_score)
    _render_gauge(col4, "ALL-CAPS Density", caps_score)


def _render_gauge(container, title: str, value: float) -> None:
    bar_colour = "#e74c3c" if value >= 0.5 else "#e67e22" if value >= 0.25 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": bar_colour},
            "steps": [
                {"range": [0.00, 0.25], "color": "#d5f5e3"},
                {"range": [0.25, 0.50], "color": "#fef9e7"},
                {"range": [0.50, 1.00], "color": "#fdebd0"},
            ],
            "threshold": {
                "line": {"color": "#e74c3c", "width": 2},
                "thickness": 0.75,
                "value": 0.5,
            },
        },
        title={"text": title},
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=60, b=20))
    container.plotly_chart(fig, use_container_width=True)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Hot Topics Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "db_path" not in st.session_state:
        st.session_state["db_path"] = str(_DEFAULT_DB)

    db_path: str = st.session_state["db_path"]

    params = st.query_params
    view = params.get("view", "home")

    if view == "topic":
        try:
            topic_id = int(params.get("topic_id", ""))
        except (ValueError, TypeError):
            st.query_params.clear()
            st.rerun()
            return
        render_topic(topic_id, db_path)

    elif view == "article":
        item_id = params.get("item_id", "")
        if not item_id:
            st.query_params.clear()
            st.rerun()
            return
        render_article(item_id, db_path)

    else:
        df = load_scored_topics(db_path)
        render_home(df, db_path)


if __name__ == "__main__":
    main()
