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
                (SELECT GROUP_CONCAT(DISTINCT ri.platform)
                 FROM topic_sources ts2
                 JOIN raw_items ri ON ri.id = ts2.item_id
                 WHERE ts2.topic_id = t.id) AS platforms
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
    c1.metric("Topics analysed", len(df))
    c2.metric("High-risk topics", int(len(flagged)), help=f"composite_risk ≥ {_HIGH_RISK}")
    c3.metric(
        "Avg source trust",
        f"{scored['avg_trust'].mean():.1f} / 100" if not scored.empty else "—",
    )
    c4.metric(
        "Avg composite risk",
        f"{scored['composite_risk'].mean():.3f}" if not scored.empty else "—",
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

    col_card, col_nav = st.columns([11, 1])
    with col_card:
        st.markdown(
            f"""<div style='border-left:4px solid {grade_colour};padding:10px 16px;
            margin-bottom:6px;background:#000;border-radius:0 6px 6px 0'>
              <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap'>
                {_grade_badge_html(grade)}
                <strong style='font-size:1.0em;color:#fff'>{_truncate(topic, 70)}</strong>
                <span style='color:#aaa;font-size:0.85em'>{articles} articles &nbsp;{icons}</span>
                <span style='margin-left:auto'>Risk: {risk_str}</span>
              </div>
              <div style='margin-top:8px'>
                <div style='background:#333;border-radius:4px;height:5px;overflow:hidden'>
                  <div style='background:{grade_colour};height:100%;width:{reliability_pct:.1f}%'></div>
                </div>
                <span style='font-size:0.75em;color:#aaa'>Reliability {reliability_pct:.1f}%</span>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
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
        xaxis=dict(title="Composite risk", range=[0, 1.05]),
        yaxis_title=None,
        margin=dict(l=0, r=50, t=40, b=20),
        height=max(280, len(plot_df) * 40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


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
                f"Composite risk: **{risk:.4f}** &nbsp;·&nbsp; "
                f"{articles} articles &nbsp;·&nbsp; {icons} &nbsp;·&nbsp; "
                f"scored {computed_at}"
            )

    st.markdown("---")

    if risk is None:
        st.info("This topic has not been scored yet. Run `python src/scoring/compute_scores.py`.")
        return

    _render_score_breakdown(row)
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


def _render_score_breakdown(row: pd.Series) -> None:
    st.subheader("Signal Breakdown")
    risk = float(row.get("composite_risk", 0) or 0)
    avg_trust = float(row.get("avg_trust", 50) or 50)
    sentiment = float(row.get("sentiment_extremity", 0) or 0)
    coverage_ratio = float(row.get("coverage_ratio", 0) or 0)
    framing = float(row.get("framing_inconsistency", 0) or 0)
    sensationalism = float(row.get("sensationalism", 0) or 0)

    contributions = {
        "🏛️ Source\nDistrust": (
            _WEIGHTS["avg_trust"] * (1.0 - avg_trust / 100.0),
            _WEIGHTS["avg_trust"],
            f"{avg_trust:.1f}/100 trust",
        ),
        "😤 Sentiment\nExtremity": (
            _WEIGHTS["avg_sentiment_extremity"] * sentiment,
            _WEIGHTS["avg_sentiment_extremity"],
            f"score {sentiment:.3f}",
        ),
        "📡 Low\nCoverage": (
            _WEIGHTS["coverage_ratio"] * (1.0 - coverage_ratio),
            _WEIGHTS["coverage_ratio"],
            f"{coverage_ratio * 100:.1f}% credible",
        ),
        "🔀 Framing\nDivergence": (
            _WEIGHTS["framing_inconsistency"] * framing,
            _WEIGHTS["framing_inconsistency"],
            f"score {framing:.3f}",
        ),
        "📢 Sensational-\nism": (
            _WEIGHTS["sensationalism_avg"] * sensationalism,
            _WEIGHTS["sensationalism_avg"],
            f"score {sensationalism:.3f}",
        ),
    }

    cols = st.columns(5)
    for col, (label, (contribution, weight, detail)) in zip(cols, contributions.items()):
        pct = (contribution / risk * 100) if risk > 0 else 0.0
        c = "#e74c3c" if contribution > 0.10 else "#e67e22" if contribution > 0.05 else "#2ecc71"
        col.markdown(
            f"""<div style='border:1px solid #dee2e6;border-radius:8px;padding:12px;
            text-align:center;min-height:120px'>
              <div style='font-size:0.78em;color:#555;white-space:pre-line;margin-bottom:4px'>{label}</div>
              <div style='font-size:1.6em;font-weight:bold;color:{c}'>{contribution:.3f}</div>
              <div style='font-size:0.75em;color:#888'>{detail}</div>
              <div style='font-size:0.68em;color:#aaa;margin-top:4px'>w={weight} · {pct:.0f}% of risk</div>
            </div>""",
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
        xaxis=dict(title="Trust (0–100)", range=[0, 100]),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=0, r=10, t=50, b=20),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


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

    if description:
        with st.expander("Description"):
            st.write(description)

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

    if cleaned_text:
        st.markdown("---")
        with st.expander("Cleaned text"):
            st.text(cleaned_text)


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
