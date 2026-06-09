"""NewsRadar Misinformation Dashboard — 3-view SPA routing.

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

from src.dashboard import i18n
from src.scoring.article_scorer import score_article
from src.scoring.attribution import score_attribution_vagueness
from src.scoring.compute_scores import _MISINFO_THRESHOLD, _WEIGHTS
from src.scoring.sentiment import _clickbait_score, _sensationalism
from src.scoring.source_lookup import domain_in_static_csv, generate_disclaimer, get_source_data
from src.scoring.source_trust import _domain_from_url, get_trust_score
from src.utils.db import init_db

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "data" / "dashboard.db"
_DEMO_DB    = _PROJECT_ROOT / "data" / "demo.db"

_HIGH_RISK = _MISINFO_THRESHOLD

def _reliability_colour(reliability_pct: float) -> str:
    if reliability_pct >= 80:
        return "#2ecc71"
    if reliability_pct >= 60:
        return "#82b74b"
    if reliability_pct >= 40:
        return "#f1c40f"
    if reliability_pct >= 20:
        return "#e67e22"
    return "#e74c3c"

_RISK_COLORSCALE = [
    [0.0, "#2ecc71"],
    [0.4, "#f1c40f"],
    [0.7, "#e67e22"],
    [1.0, "#e74c3c"],
]

PLATFORM_ICONS: dict[str, str] = {
    "reddit":      "🔴",
    # "youtube":     "▶️",  # disabled — not used in current pipeline
    "newsapi":     "📰",
    "google_news": "🌐",
    "duckduckgo":  "🦆",
    "rss":         "📡",
}


# ── data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_scored_topics(db_path: str) -> pd.DataFrame:
    """Load all topics with scores for the latest completed run.

    Returns empty DataFrame if unavailable.
    """
    try:
        conn = init_db(db_path)

        latest_run = conn.execute(
            "SELECT MAX(id) FROM pipeline_runs WHERE status = 'completed'"
        ).fetchone()[0]

        # Backward compat: if pipeline_runs is empty, show all topics.
        run_filter = f"AND t.run_id = {latest_run}" if latest_run is not None else ""

        rows = conn.execute(
            f"""
            SELECT
                t.id            AS topic_id,
                t.label         AS topic,
                t.item_count    AS articles,
                t.created_at,
                t.run_id,
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
            WHERE 1=1 {run_filter}
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
    df.loc[scored, "reliability_pct"] = ((1.0 - df.loc[scored, "composite_risk"]) * 100).round(1)
    df["coverage_ratio_pct"] = (df["coverage_ratio"].fillna(0) * 100).round(1)
    df["bubble_size"] = df["coverage_breadth"].fillna(0).clip(lower=1)
    return df


@st.cache_data(ttl=60)
def load_latest_run(db_path: str) -> dict | None:
    """Return metadata for the latest completed pipeline run, or None."""
    try:
        conn = init_db(db_path)
        row = conn.execute(
            """
            SELECT id, started_at, completed_at
            FROM pipeline_runs
            WHERE status = 'completed'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_topic_articles(db_path: str, topic_id: int) -> list[dict]:
    """Load articles for a topic with per-article trust and sensationalism."""
    try:
        conn = init_db(db_path)
        rows = conn.execute(
            """
            SELECT ri.id, ri.title, ri.description, ri.body_text,
                   ri.source, ri.url,
                   ri.platform, ri.timestamp, ri.engagement_json, ri.cleaned_text
            FROM topic_sources ts
            JOIN raw_items ri ON ri.id = ts.item_id
            WHERE ts.topic_id = ?
            ORDER BY ri.timestamp DESC
            """,
            (topic_id,),
        ).fetchall()
    except Exception:
        return []

    result = []
    try:
        for row in rows:
            d = dict(row)
            try:
                d["engagement"] = json.loads(d.pop("engagement_json", "{}") or "{}")
            except Exception:
                d["engagement"] = {}
            domain = _domain_from_url(d.get("url", "")) or d.get("source", "")
            d["trust_score"] = get_trust_score(domain, conn=conn)
            text = d.get("cleaned_text") or d.get("title", "")
            d["sensationalism_score"] = _sensationalism(text)
            result.append(d)
    finally:
        conn.close()
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


def _pct_badge_html(reliability_pct: float) -> str:
    colour = _reliability_colour(reliability_pct)
    return (
        f"<span style='background:{colour};color:#fff;padding:3px 10px;"
        f"border-radius:4px;font-weight:bold;font-size:1.0em'>{reliability_pct:.0f} %</span>"
    )


def _risk_pct_badge_html(risk: float) -> str:
    colour = "#e74c3c" if risk >= _HIGH_RISK else "#e67e22" if risk >= 0.40 else "#2ecc71"
    risk_pct = risk * 100
    return (
        f"<span style='background:{colour};color:#fff;padding:3px 10px;"
        f"border-radius:4px;font-weight:bold;font-size:1.0em'>{risk_pct:.0f} %</span>"
    )


def _risk_badge_html(risk: float) -> str:
    colour = "#e74c3c" if risk >= _HIGH_RISK else "#e67e22" if risk >= 0.40 else "#2ecc71"
    return (
        f"<span style='background:{colour};color:#fff;padding:2px 8px;"
        f"border-radius:4px;font-size:0.85em'>{risk:.3f}</span>"
    )


# ── demo mode ─────────────────────────────────────────────────────────────────

def _render_demo_banner() -> None:
    """Persistent banner + exit button shown on every view while in demo mode."""
    col_msg, col_btn = st.columns([8, 2])
    with col_msg:
        st.info(i18n.DEMO_BANNER, icon="🎭")
    with col_btn:
        st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
        if st.button(i18n.DEMO_EXIT_BTN, use_container_width=True, key="demo_exit_global"):
            st.session_state["demo_mode"] = False
            st.session_state["db_path"] = str(_DEFAULT_DB)
            st.query_params.clear()
            st.cache_data.clear()
            st.rerun()


# ── home view ─────────────────────────────────────────────────────────────────

def render_home(df: pd.DataFrame, db_path: str) -> None:
    col_title, col_demo, col_settings = st.columns([5, 2, 1], vertical_alignment="bottom")
    with col_title:
        st.title(i18n.APP_TITLE)
        latest_run = load_latest_run(db_path)
        if latest_run and latest_run.get("completed_at"):
            from datetime import datetime, timezone, timedelta
            _BERLIN = timezone(timedelta(hours=2))
            _raw = str(latest_run["completed_at"]).replace(" ", "T")
            if not _raw.endswith("Z") and "+" not in _raw[10:]:
                _raw += "+00:00"
            run_ts = datetime.fromisoformat(_raw).astimezone(_BERLIN).strftime("%Y-%m-%d %H:%M:%S")
            run_id = latest_run["id"]
            st.caption(f"{i18n.APP_CAPTION} &nbsp;·&nbsp; Run #{run_id} · updated {run_ts} UTC+2")
        else:
            st.caption(i18n.APP_CAPTION)
    with col_demo:
        if not st.session_state.get("demo_mode"):
            if st.button(i18n.DEMO_BTN, use_container_width=True, help=i18n.DEMO_BTN_HELP):
                if not _DEMO_DB.exists():
                    st.error(i18n.DEMO_DB_MISSING)
                else:
                    st.session_state["demo_mode"] = True
                    st.session_state["db_path"] = str(_DEMO_DB)
                    st.cache_data.clear()
                    st.rerun()
    with col_settings:
        with st.expander("⚙️"):
            new_path = st.text_input("Database", value=db_path, label_visibility="collapsed")
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.session_state["db_path"] = new_path
                st.rerun()

    if df.empty:
        st.warning(i18n.CAPTION_PIPELINE_MISSING.format(db_path=db_path))
        return

    scored = df[df["composite_risk"].notna()]
    flagged = scored[scored["composite_risk"] >= _HIGH_RISK]

    if not flagged.empty:
        lines = "  \n".join(
            f"- **{r['topic']}** — risk {float(r['composite_risk']) * 100:.0f} %"
            for _, r in flagged.iterrows()
        )
        st.error(
            f"**{len(flagged)} high-risk topic(s) detected** (risk ≥ {_HIGH_RISK}):\n\n{lines}",
            icon="🚨",
        )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        i18n.METRIC_TOPICS_ANALYSED,
        len(df),
        help=i18n.METRIC_TOPICS_ANALYSED_HELP,
    )
    c2.metric(
        i18n.METRIC_HIGH_RISK,
        int(len(flagged)),
        help=(
            f"Topics whose composite risk exceeds {_HIGH_RISK} (50 %). "
            "Based on source trust, sentiment, framing divergence and sensationalism."
        ),
    )
    c3.metric(
        i18n.METRIC_AVG_RISK,
        f"{scored['composite_risk'].mean():.1%}" if not scored.empty else "—",
        help=i18n.METRIC_AVG_RISK_HELP,
    )

    with st.expander(i18n.EXPANDER_HOW_RISK):
        st.markdown(i18n.EXPANDER_HOW_RISK_TEXT)

    st.markdown("---")
    st.subheader(i18n.SECTION_TOPIC_RANKING)
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
    platforms_str = str(row.get("platforms", "") or "")
    risk_colour = (
        "#e74c3c" if risk is not None and risk >= _HIGH_RISK
        else "#e67e22" if risk is not None and risk >= 0.40
        else "#2ecc71"
    )
    icons = _platform_icons(platforms_str)
    risk_badge = (
        _risk_pct_badge_html(risk) if risk is not None
        else f"<em style='color:#aaa'>{i18n.LABEL_UNSCORED}</em>"
    )

    keywords = _parse_keywords(row.get("keywords_raw"))
    kw_section = "".join(
        f"<span style='background:#2a2a2a;color:#e0e0e0;border:1px solid #555;"
        f"border-radius:3px;padding:1px 6px;font-size:0.72em;margin-right:4px'>{kw}</span>"
        for kw in keywords
    )
    kw_row = f"<div style='margin-top:5px'>{kw_section}</div>" if kw_section else ""
    title_row = (
        f"{risk_badge}"
        f"<strong style='font-size:1.0em;color:#fff;margin-left:8px'>{_truncate(topic, 70)}</strong>"
        f"<span style='color:#aaa;font-size:0.85em;margin-left:8px'>{articles} {i18n.LABEL_ARTICLES} &nbsp;{icons}</span>"
    )
    card_html = (
        f"<div style='border-left:4px solid {risk_colour};padding:10px 16px;"
        f"margin-bottom:6px;background:#000;border-radius:0 6px 6px 0'>"
        f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>{title_row}</div>"
        f"{kw_row}"
        f"</div>"
    )

    col_card, col_nav = st.columns([11, 1])
    with col_card:
        st.markdown(card_html, unsafe_allow_html=True)
    with col_nav:
        if st.button("→", key=f"nav_t_{topic_id}", help="Open topic"):
            st.query_params["view"] = "topic"
            st.query_params["topic_id"] = str(topic_id)
            st.rerun()


def _render_scatter(df: pd.DataFrame) -> None:
    needed = ["sentiment_extremity", "sensationalism", "composite_risk"]
    plot_df = df.dropna(subset=needed).copy()
    if plot_df.empty:
        return
    plot_df["label"] = plot_df["topic"].apply(lambda t: _truncate(t, 40))
    plot_df["reliability_pct"] = ((1.0 - plot_df["composite_risk"]) * 100).round(1)
    fig = px.scatter(
        plot_df,
        x="sentiment_extremity",
        y="sensationalism",
        color="composite_risk",
        color_continuous_scale=_RISK_COLORSCALE,
        range_color=[0.0, 1.0],
        size="articles",
        hover_name="label",
        hover_data={"reliability_pct": ":.1f", "composite_risk": ":.2f", "articles": True},
        labels={
            "sentiment_extremity": i18n.AXIS_SENTIMENT,
            "sensationalism": i18n.AXIS_SENSATIONALISM,
            "composite_risk": i18n.AXIS_RISK,
        },
        title=i18n.CHART_SENTIMENT_VS_SENS,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=20),
        height=320,
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
    )
    fig.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_bar(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("composite_risk", ascending=True).copy()
    risk_vals = plot_df["composite_risk"]
    fig = go.Figure(go.Bar(
        x=risk_vals,
        y=plot_df["topic"].apply(lambda t: _truncate(t, 40)),
        orientation="h",
        marker_color=[
            "#e74c3c" if r >= _HIGH_RISK else "#e67e22" if r >= 0.40 else "#2ecc71"
            for r in risk_vals
        ],
        text=[f"{r * 100:.0f} %" for r in risk_vals],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Risk: %{x:.0%}<extra></extra>",
    ))
    fig.add_vline(
        x=_HIGH_RISK, line_dash="dash", line_color="#e74c3c",
        annotation_text=f"{i18n.CHART_RISK_THRESHOLD_LABEL} ({_HIGH_RISK:.0%})",
        annotation_position="top right",
    )
    fig.update_layout(
        title=i18n.CHART_COMPOSITE_RISK,
        xaxis=dict(title=i18n.AXIS_COMPOSITE_RISK, range=[0, 1.15], tickformat=".0%"),
        yaxis_title=None,
        margin=dict(l=0, r=50, t=40, b=20),
        height=max(280, len(plot_df) * 40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(i18n.CAPTION_ONLY_SCORED)


# ── topic detail view ─────────────────────────────────────────────────────────

def render_topic(topic_id: int, db_path: str) -> None:
    def _back():
        st.query_params.clear()
        st.rerun()

    df = load_scored_topics(db_path)
    topic_rows = df[df["topic_id"] == topic_id] if not df.empty else pd.DataFrame()

    col_back, col_hdr = st.columns([1, 9])
    with col_back:
        if st.button(i18n.BTN_BACK, key="back_home"):
            _back()

    if topic_rows.empty:
        with col_hdr:
            st.error(i18n.TOPIC_NOT_FOUND.format(topic_id=topic_id))
        return

    row = topic_rows.iloc[0]
    risk = row.get("composite_risk")
    articles = int(row.get("articles", 0) or 0)
    icons = _platform_icons(str(row.get("platforms", "") or ""))
    from datetime import datetime, timezone, timedelta
    _BERLIN = timezone(timedelta(hours=2))
    _raw_ca = str(row.get("computed_at") or "").replace(" ", "T")
    if _raw_ca and not _raw_ca.endswith("Z") and "+" not in _raw_ca[10:]:
        _raw_ca += "+00:00"
    computed_at = (
        datetime.fromisoformat(_raw_ca).astimezone(_BERLIN).strftime("%Y-%m-%d %H:%M:%S")
        if _raw_ca else ""
    )

    with col_hdr:
        st.markdown(f"## {row['topic']}")
        if risk is not None:
            st.caption(
                i18n.CAPTION_COMPOSITE_RISK.format(risk=risk * 100)
                + f" &nbsp;·&nbsp; "
                + i18n.CAPTION_ARTICLES.format(n=articles)
                + f" &nbsp;·&nbsp; {icons} &nbsp;·&nbsp; "
                + i18n.CAPTION_SCORED_AT.format(ts=computed_at)
            )

    st.markdown("---")

    if risk is None:
        st.info(i18n.TOPIC_NOT_SCORED)
        return

    _render_score_breakdown(row)
    st.markdown("---")

    col_radar, col_domain = st.columns(2)
    with col_radar:
        _render_radar(row)
    with col_domain:
        _render_domain_trust_bar(db_path, topic_id)

    st.markdown("---")
    st.subheader(i18n.SECTION_ARTICLES)
    articles_data = load_topic_articles(db_path, topic_id)
    if not articles_data:
        st.info(i18n.ARTICLES_NONE)
    else:
        for a in articles_data:
            _render_article_row(a, topic_id, db_path)


def _render_score_breakdown(row: pd.Series) -> None:
    st.subheader(i18n.SECTION_SIGNAL_BREAKDOWN)
    st.caption(i18n.SIGNAL_BREAKDOWN_CAPTION)

    risk = float(row.get("composite_risk", 0) or 0)
    avg_article_risk = float(row.get("avg_article_risk", 0) or 0)
    framing = float(row.get("framing_inconsistency", 0) or 0)
    fact = float(row.get("fact_inconsistency", 0) or 0)

    # Sub-signals bundled inside avg_article_risk
    avg_trust = float(row.get("avg_trust", 50) or 50)
    sentiment = float(row.get("sentiment_extremity", 0) or 0)
    sensationalism = float(row.get("sensationalism", 0) or 0)
    attribution = float(row.get("attribution_vagueness", 0) or 0)

    signals = [
        (
            i18n.SIGNAL_NAMES["Article Risk"],
            _WEIGHTS["avg_article_risk"] * avg_article_risk,
            _WEIGHTS["avg_article_risk"],
            f"Avg article risk {avg_article_risk * 100:.1f} %",
            i18n.SIGNAL_TOOLTIPS["Article Risk"],
        ),
        (
            i18n.SIGNAL_NAMES["Framing Divergence"],
            _WEIGHTS["framing_inconsistency"] * framing,
            _WEIGHTS["framing_inconsistency"],
            f"Signal {framing * 100:.1f} %",
            i18n.SIGNAL_TOOLTIPS["Framing Divergence"],
        ),
        (
            i18n.SIGNAL_NAMES["Fact Inconsistency"],
            _WEIGHTS["fact_inconsistency"] * fact,
            _WEIGHTS["fact_inconsistency"],
            f"Signal {fact * 100:.1f} %",
            i18n.SIGNAL_TOOLTIPS["Fact Inconsistency"],
        ),
    ]

    cols = st.columns(3)
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
                Weight {weight * 100:.0f} % · {share_pct:.0f} % of composite risk
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander(i18n.EXPANDER_ARTICLE_RISK_DETAIL):
        st.caption(i18n.SIGNAL_BREAKDOWN_CAPTION)
        sub_signals = [
            (i18n.SIGNAL_NAMES["Source Distrust"],      1.0 - avg_trust / 100.0, 0.30, f"Avg trust {avg_trust:.1f}",    i18n.SIGNAL_TOOLTIPS["Source Distrust"]),
            (i18n.SIGNAL_NAMES["Sentiment Extremity"],  sentiment,               0.25, f"{sentiment * 100:.1f} %",      i18n.SIGNAL_TOOLTIPS["Sentiment Extremity"]),
            (i18n.SIGNAL_NAMES["Sensationalism"],       sensationalism,          0.25, f"{sensationalism * 100:.1f} %", i18n.SIGNAL_TOOLTIPS["Sensationalism"]),
            (i18n.SIGNAL_NAMES["Attribution Vagueness"],attribution,             0.20, f"{attribution * 100:.1f} %",    i18n.SIGNAL_TOOLTIPS["Attribution Vagueness"]),
        ]
        sub_cols = st.columns(4)
        for sub_col, (label, raw_val, weight, detail, tooltip) in zip(sub_cols, sub_signals):
            c = "#e74c3c" if raw_val >= 0.6 else "#e67e22" if raw_val >= 0.3 else "#2ecc71"
            sub_col.markdown(
                f"""<div title="{tooltip}" style='border:1px solid #2a2a2a;border-radius:6px;
                padding:10px;text-align:center;cursor:help'>
                  <div style='font-size:0.75em;color:#666;margin-bottom:4px'>{label}</div>
                  <div style='font-size:1.2em;font-weight:bold;color:{c}'>{raw_val * 100:.1f}%</div>
                  <div style='font-size:0.7em;color:#888'>{detail}</div>
                  <div style='font-size:0.65em;color:#aaa;margin-top:2px'>Weight in article risk: {weight * 100:.0f} %</div>
                </div>""",
                unsafe_allow_html=True,
            )

    _render_risk_waterfall(row)

    with st.expander(i18n.EXPANDER_HOW_RISK):
        st.markdown(i18n.EXPANDER_HOW_RISK_TEXT)


def _render_risk_waterfall(row: pd.Series) -> None:
    """Horizontal stacked bar showing each signal's weighted contribution to composite_risk."""
    risk = float(row.get("composite_risk", 0) or 0)
    if risk <= 0:
        return

    avg_article_risk = float(row.get("avg_article_risk", 0) or 0)
    framing = float(row.get("framing_inconsistency", 0) or 0)
    fact = float(row.get("fact_inconsistency", 0) or 0)

    contributions = [
        (_WEIGHTS["avg_article_risk"] * avg_article_risk,           i18n.SIGNAL_NAMES["Article Risk"]),
        (_WEIGHTS["framing_inconsistency"] * framing,               i18n.SIGNAL_NAMES["Framing Divergence"]),
        (_WEIGHTS["fact_inconsistency"] * fact,                     i18n.SIGNAL_NAMES["Fact Inconsistency"]),
    ]

    fig = go.Figure()
    for (val, name), colour in zip(contributions, i18n.BREAKDOWN_SIGNAL_COLOURS):
        fig.add_trace(go.Bar(
            name=name,
            x=[val],
            y=["Composite Risk"],
            orientation="h",
            marker_color=colour,
            text=f"{val * 100:.1f} %" if val > 0.005 else "",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>{name}</b><br>Contribution: {val * 100:.2f} %<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text=i18n.BREAKDOWN_CHART_TITLE, x=0, font=dict(size=14)),
        xaxis=dict(
            title=None,
            range=[0, max(risk * 1.1, 0.05)],
            tickformat=".0%",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(visible=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=160,
        margin=dict(l=0, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(i18n.BREAKDOWN_CHART_CAPTION)



def _render_radar(row: pd.Series) -> None:
    avg_trust = float(row.get("avg_trust", 50) or 50)
    sentiment = float(row.get("sentiment_extremity", 0) or 0)
    framing = float(row.get("framing_inconsistency", 0) or 0)
    sensationalism = float(row.get("sensationalism", 0) or 0)

    categories = i18n.RADAR_CATEGORIES
    values = [
        1.0 - avg_trust / 100.0,
        sentiment,
        framing,
        sensationalism,
    ]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.18)",
        line=dict(color="#e74c3c", width=2.5),
        mode="lines+markers",
        marker=dict(color="#e74c3c", size=7, line=dict(color="#fff", width=1)),
        hovertemplate="<b>%{theta}</b><br>%{r:.0%}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(color="rgba(255,255,255,0.55)", size=10),
                gridcolor="rgba(255,255,255,0.1)",
                linecolor="rgba(255,255,255,0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(color="rgba(255,255,255,0.85)", size=12),
                gridcolor="rgba(255,255,255,0.12)",
                linecolor="rgba(255,255,255,0.18)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=50, r=50, t=55, b=30),
        height=320,
        title=dict(
            text=i18n.RADAR_CHART_TITLE,
            x=0.5,
            xanchor="center",
            font=dict(size=15, color="rgba(255,255,255,0.9)"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(i18n.RADAR_CAPTION)
    with st.expander(i18n.EXPANDER_RADAR):
        st.markdown(i18n.EXPANDER_RADAR_TEXT)


def _render_domain_trust_bar(db_path: str, topic_id: int) -> None:
    articles = load_topic_articles(db_path, topic_id)
    if not articles:
        st.info(i18n.DOMAIN_TRUST_NO_DATA)
        return

    seen: dict[str, float] = {}
    for a in articles:
        domain = _domain_from_url(a.get("url", "")) or a.get("source", "unknown")
        if domain and domain not in seen:
            seen[domain] = a.get("trust_score", get_trust_score(domain))

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
        title=dict(text=i18n.SECTION_DOMAIN_TRUST, x=0.5, xanchor="center",
                   font=dict(size=15, color="rgba(255,255,255,0.9)")),
        xaxis=dict(title="Trust Score (0–100)", range=[0, 100],
                   gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="rgba(255,255,255,0.6)")),
        yaxis=dict(tickfont=dict(size=10, color="rgba(255,255,255,0.85)"),
                   gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=0, r=10, t=55, b=20),
        height=320,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(i18n.DOMAIN_TRUST_CAPTION)
    with st.expander(i18n.EXPANDER_DOMAIN_TRUST):
        st.markdown(i18n.EXPANDER_DOMAIN_TRUST_TEXT)


def _render_article_disclaimer(domain: str, db_path: str) -> None:
    """Render a muted source-evaluation caption below an article card.

    Shows nothing for unknown domains where MBFC lookup failed.
    Shows 'Source data unavailable' for known CSV outlets where lookup failed.
    Shows the full disclaimer otherwise.
    """
    if not domain:
        return
    try:
        data = get_source_data(domain, db_path=db_path)
    except Exception:
        return
    if data.source == "unavailable" and data.confidence == "unavailable":
        if domain_in_static_csv(domain):
            st.caption("ℹ️ Source data unavailable")
        return
    disclaimer = generate_disclaimer(data)
    if disclaimer:
        st.caption(f"ℹ️ {disclaimer}")


def _render_article_row(article: dict, topic_id: int, db_path: str = str(_DEFAULT_DB)) -> None:
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
            f" &nbsp;·&nbsp; <span style='color:{trust_colour}'>Trust {trust:.0f}</span>"
            f" &nbsp;·&nbsp; Sensationalism {sens:.0%}"
            f"</span>",
            unsafe_allow_html=True,
        )
        _render_article_disclaimer(
            _domain_from_url(article.get("url", "")) or article.get("source", ""),
            db_path,
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
        st.markdown(f"**{i18n.ARTICLE_ANALYSIS_LABEL}**")

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
    description = article.get("description") or ""
    body_text = article.get("body_text") or ""
    cleaned_text = article.get("cleaned_text") or article.get("title", "")

    st.markdown(f"### {icon} [{title}]({url})")
    st.caption(f"**{source}** &nbsp;·&nbsp; {timestamp}")
    _render_article_disclaimer(
        _domain_from_url(url) or source,
        db_path,
    )

    def _plain_text(text: str) -> None:
        st.markdown(
            f"<div style='font-size:0.9em;line-height:1.6;white-space:pre-wrap'>{text}</div>",
            unsafe_allow_html=True,
        )

    if body_text:
        with st.expander(i18n.ARTICLE_FULL_TEXT, expanded=True):
            _plain_text(body_text)
        if description:
            with st.expander(i18n.ARTICLE_SUMMARY, expanded=False):
                _plain_text(description)
    else:
        with st.expander(i18n.ARTICLE_DESCRIPTION, expanded=True):
            _plain_text(description or cleaned_text or i18n.ARTICLE_NO_TEXT)

    st.markdown("---")
    st.subheader(i18n.SECTION_SIGNAL_ANALYSIS)

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
    _render_gauge(col1, i18n.GAUGE_SENSATIONALISM, sens_score)
    _render_gauge(col2, i18n.GAUGE_ATTRIBUTION, attr_score)
    _render_gauge(col3, i18n.GAUGE_CLICKBAIT, click_score)
    _render_gauge(col4, i18n.GAUGE_CAPS, caps_score)


def _render_gauge(container, title: str, value: float) -> None:
    bar_colour = "#e74c3c" if value >= 0.5 else "#e67e22" if value >= 0.25 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"valueformat": ".0f", "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": bar_colour},
            "steps": [
                {"range": [0, 25],  "color": "#d5f5e3"},
                {"range": [25, 50], "color": "#fef9e7"},
                {"range": [50, 100],"color": "#fdebd0"},
            ],
            "threshold": {
                "line": {"color": "#e74c3c", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": title},
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=60, b=20))
    container.plotly_chart(fig, use_container_width=True)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title=i18n.PAGE_TITLE,
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "db_path" not in st.session_state:
        st.session_state["db_path"] = str(_DEFAULT_DB)
    if "demo_mode" not in st.session_state:
        st.session_state["demo_mode"] = False

    db_path: str = st.session_state["db_path"]

    if st.session_state.get("demo_mode"):
        _render_demo_banner()

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
