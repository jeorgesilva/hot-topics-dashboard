"""Hot Topics Misinformation Dashboard.

Run with:
    streamlit run src/dashboard/app.py

Person A panel: sentiment extremity, sensationalism, framing inconsistency.
Person B panel: source trust, coverage breadth, composite risk (stub — to be filled).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.scoring.compute_scores import grade_topic

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "data" / "dashboard.db"

# Risk thresholds for visual callouts
_HIGH_RISK = 0.60
_MODERATE_RISK = 0.40

# Colour palette consistent across all charts (low=green, high=red)
_RISK_COLORSCALE = [
    [0.0,  "#2ecc71"],  # green
    [0.4,  "#f1c40f"],  # yellow
    [0.7,  "#e67e22"],  # orange
    [1.0,  "#e74c3c"],  # red
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _get_conn(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=60)
def load_scored_topics(db_path: str) -> pd.DataFrame:
    """Load topics with their scores from the DB; returns empty DataFrame if unavailable."""
    conn = _get_conn(Path(db_path))
    if conn is None:
        return pd.DataFrame()
    try:
        rows = conn.execute(
            """
            SELECT
                t.id            AS topic_id,
                t.label         AS topic,
                t.item_count    AS articles,
                t.created_at,
                ts.avg_sentiment_extremity  AS sentiment_extremity,
                ts.sensationalism_avg       AS sensationalism,
                ts.framing_inconsistency    AS framing_inconsistency,
                ts.avg_trust                AS avg_trust,
                ts.trust_variance           AS trust_variance,
                ts.coverage_breadth         AS coverage_breadth,
                ts.coverage_ratio           AS coverage_ratio,
                ts.composite_risk,
                ts.computed_at
            FROM topics t
            LEFT JOIN topic_scores ts ON t.id = ts.topic_id
            ORDER BY ts.composite_risk DESC NULLS LAST, t.created_at DESC
            """
        ).fetchall()
        conn.close()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception:
        conn.close()
        return pd.DataFrame()


def _truncate(label: str, n: int = 55) -> str:
    return label if len(label) <= n else label[:n] + "…"


def _risk_badge(risk: float | None) -> str:
    if risk is None:
        return "—"
    grade = grade_topic(risk)
    color = (
        "#e74c3c" if risk >= _HIGH_RISK
        else "#e67e22" if risk >= _MODERATE_RISK
        else "#2ecc71"
    )
    return f'<span style="color:{color};font-weight:700">{grade} ({risk:.2f})</span>'


# ── sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> Path:
    st.sidebar.title("Hot Topics Dashboard")
    st.sidebar.caption("Misinformation risk scoring")

    custom = st.sidebar.text_input(
        "Database path",
        value=str(_DEFAULT_DB),
        help="Absolute path to dashboard.db",
    )
    db_path = Path(custom)

    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Pipeline commands**\n"
        "```\npython src/scrapers/run_all.py\n"
        "python src/scoring/compute_scores.py\n```"
    )
    return db_path


# ── empty state ───────────────────────────────────────────────────────────────

def _render_empty(db_path: Path) -> None:
    st.warning(
        f"No scored topics found at `{db_path}`.\n\n"
        "Run the pipeline first:\n"
        "```\npython src/scrapers/run_all.py\n"
        "python src/scoring/compute_scores.py\n```"
    )


# ── KPI cards ─────────────────────────────────────────────────────────────────

def _render_kpi_row(df: pd.DataFrame) -> None:
    scored = df[df["composite_risk"].notna()]
    high_risk = (scored["composite_risk"] >= _HIGH_RISK).sum()
    avg_sensationalism = (
        df["sensationalism"].mean() if "sensationalism" in df else float("nan")
    )
    avg_framing = (
        df["framing_inconsistency"].mean()
        if "framing_inconsistency" in df
        else float("nan")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics analysed", len(df))
    c2.metric(
        "High-risk topics",
        int(high_risk),
        delta=None,
        help=f"composite_risk ≥ {_HIGH_RISK}",
    )
    c3.metric(
        "Avg sensationalism",
        f"{avg_sensationalism:.2f}" if pd.notna(avg_sensationalism) else "—",
    )
    c4.metric(
        "Avg framing gap",
        f"{avg_framing:.2f}" if pd.notna(avg_framing) else "—",
    )


# ── Person A charts ───────────────────────────────────────────────────────────

def _hbar(
    df: pd.DataFrame,
    col: str,
    title: str,
    x_label: str,
) -> go.Figure:
    """Horizontal bar chart with risk colour scale, sorted descending."""
    plot_df = (
        df[["topic", col]]
        .dropna(subset=[col])
        .sort_values(col, ascending=True)
        .copy()
    )
    plot_df["label"] = plot_df["topic"].apply(_truncate)

    fig = px.bar(
        plot_df,
        x=col,
        y="label",
        orientation="h",
        color=col,
        color_continuous_scale=_RISK_COLORSCALE,
        range_color=[0.0, 1.0],
        labels={col: x_label, "label": ""},
        title=title,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=10, t=40, b=20),
        height=max(220, 36 * len(plot_df)),
        yaxis=dict(tickfont=dict(size=11)),
        xaxis=dict(range=[0, 1]),
    )
    return fig


def _render_person_a_charts(df: pd.DataFrame) -> None:
    st.subheader("Sentiment & Sensationalism")

    cols_needed = ["sentiment_extremity", "sensationalism", "framing_inconsistency"]
    missing = [c for c in cols_needed if c not in df.columns or df[c].isna().all()]
    if missing:
        st.info(
            "Person A scores not yet computed. "
            "Run `python src/scoring/compute_scores.py` to populate."
        )
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            _hbar(df, "sentiment_extremity", "Sentiment Extremity", "Extremity (0–1)"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            _hbar(df, "sensationalism", "Sensationalism Score", "Score (0–1)"),
            use_container_width=True,
        )

    st.plotly_chart(
        _hbar(df, "framing_inconsistency", "Framing Inconsistency (high-trust vs low-trust tiers)", "Cosine distance (0–1)"),
        use_container_width=True,
    )


# ── scatter: sentiment vs sensationalism ──────────────────────────────────────

def _render_scatter(df: pd.DataFrame) -> None:
    needed = ["sentiment_extremity", "sensationalism", "composite_risk"]
    if any(c not in df.columns or df[c].isna().all() for c in needed):
        return

    plot_df = df.dropna(subset=needed).copy()
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
        title="Sentiment Extremity vs Sensationalism (bubble size = article count)",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=20), height=380)
    st.plotly_chart(fig, use_container_width=True)


# ── topic table ───────────────────────────────────────────────────────────────

def _render_topic_table(df: pd.DataFrame) -> None:
    st.subheader("Topic Scores")

    display_cols = {
        "topic":                  "Topic",
        "articles":               "Articles",
        "composite_risk":         "Risk",
        "sentiment_extremity":    "Sentiment",
        "sensationalism":         "Sensationalism",
        "framing_inconsistency":  "Framing gap",
        "avg_trust":              "Avg trust",
        "coverage_ratio":         "Coverage ratio",
    }
    available = {k: v for k, v in display_cols.items() if k in df.columns}
    table_df = df[list(available.keys())].rename(columns=available).copy()

    float_cols = [v for k, v in available.items() if k not in ("topic", "articles")]
    for col in float_cols:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "—"
            )

    st.dataframe(table_df, use_container_width=True, hide_index=True)


# ── high-risk callout ─────────────────────────────────────────────────────────

def _render_high_risk_callout(df: pd.DataFrame) -> None:
    if "composite_risk" not in df.columns:
        return
    flagged = df[df["composite_risk"] >= _HIGH_RISK].sort_values(
        "composite_risk", ascending=False
    )
    if flagged.empty:
        return

    st.error(f"**{len(flagged)} high-risk topic(s) detected** (risk ≥ {_HIGH_RISK})")
    for _, row in flagged.iterrows():
        risk = row["composite_risk"]
        grade = grade_topic(risk)
        label = _truncate(row["topic"], 80)
        st.markdown(f"- **{grade}** `{risk:.2f}` — {label}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Hot Topics Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    db_path = _sidebar()
    df = load_scored_topics(str(db_path))

    st.title("Hot Topics Misinformation Dashboard")
    if df.empty:
        _render_empty(db_path)
        return

    _render_high_risk_callout(df)
    _render_kpi_row(df)

    st.markdown("---")

    tab_a, tab_b = st.tabs(["Sentiment & Framing (Person A)", "Trust & Coverage (Person B)"])

    with tab_a:
        _render_person_a_charts(df)
        st.markdown("---")
        _render_scatter(df)
        st.markdown("---")
        _render_topic_table(df)

    with tab_b:
        st.info(
            "Person B panel — source trust, coverage breadth, and composite risk "
            "breakdown will be implemented here."
        )


if __name__ == "__main__":
    main()
