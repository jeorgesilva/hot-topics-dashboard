"""Hot Topics Misinformation Dashboard.

Run with:
    streamlit run src/dashboard/app.py

Person A tab: sentiment extremity, sensationalism, framing inconsistency charts.
Person B tab: source trust, coverage breadth, composite risk table and charts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.scoring.compute_scores import _MISINFO_THRESHOLD, grade_topic
from src.utils.db import init_db

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "data" / "dashboard.db"

_HIGH_RISK = _MISINFO_THRESHOLD
_MODERATE_RISK = 0.40

_RISK_COLORSCALE = [
    [0.0, "#2ecc71"],
    [0.4, "#f1c40f"],
    [0.7, "#e67e22"],
    [1.0, "#e74c3c"],
]

_GRADE_COLOUR: dict[str, str] = {
    "A": "#2ecc71",
    "B": "#82b74b",
    "C": "#f1c40f",
    "D": "#e67e22",
    "F": "#e74c3c",
}


# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_scored_topics(db_path: str) -> pd.DataFrame:
    """Load all topics with their scores. Returns empty DataFrame if unavailable."""
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
                ts.composite_risk,
                ts.computed_at
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _truncate(label: str, n: int = 55) -> str:
    return label if len(label) <= n else label[:n] + "…"


# ── sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> str:
    st.sidebar.title("Hot Topics Dashboard")
    st.sidebar.caption("Misinformation risk scoring")

    db_path = st.sidebar.text_input(
        "Database path",
        value=str(_DEFAULT_DB),
        help="Absolute path to dashboard.db",
    )

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

def _render_empty(db_path: str) -> None:
    st.warning(
        f"No scored topics found at `{db_path}`.\n\n"
        "Run the pipeline first:\n"
        "```\npython src/scrapers/run_all.py\n"
        "python src/scoring/compute_scores.py\n```"
    )


# ── KPI row ───────────────────────────────────────────────────────────────────

def _render_kpi_row(df: pd.DataFrame) -> None:
    scored = df[df["composite_risk"].notna()]
    high_risk_count = (scored["composite_risk"] >= _HIGH_RISK).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics analysed", len(df))
    c2.metric("High-risk topics", int(high_risk_count), help=f"composite_risk ≥ {_HIGH_RISK}")
    c3.metric(
        "Avg source trust",
        f"{scored['avg_trust'].mean():.1f} / 100" if not scored.empty else "—",
    )
    c4.metric(
        "Avg composite risk",
        f"{scored['composite_risk'].mean():.3f}" if not scored.empty else "—",
    )


# ── high-risk callout ─────────────────────────────────────────────────────────

def _render_high_risk_callout(df: pd.DataFrame) -> None:
    scored = df[df["composite_risk"].notna()]
    flagged = scored[scored["composite_risk"] >= _HIGH_RISK].sort_values(
        "composite_risk", ascending=False
    )
    if flagged.empty:
        return
    st.error(f"**{len(flagged)} high-risk topic(s) detected** (risk ≥ {_HIGH_RISK})")
    for _, row in flagged.iterrows():
        grade = row.get("grade", "?")
        st.markdown(f"- **{grade}** `{row['composite_risk']:.2f}` — {_truncate(row['topic'], 80)}")


# ── Person A charts ───────────────────────────────────────────────────────────

def _hbar(df: pd.DataFrame, col: str, title: str, x_label: str) -> go.Figure:
    plot_df = (
        df[["topic", col]]
        .dropna(subset=[col])
        .sort_values(col, ascending=True)
        .copy()
    )
    plot_df["label"] = plot_df["topic"].apply(_truncate)
    fig = px.bar(
        plot_df, x=col, y="label", orientation="h",
        color=col, color_continuous_scale=_RISK_COLORSCALE, range_color=[0.0, 1.0],
        labels={col: x_label, "label": ""}, title=title,
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
    if any(c not in df.columns or df[c].isna().all() for c in cols_needed):
        st.info(
            "Person A scores not yet available. "
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
        _hbar(df, "framing_inconsistency", "Framing Inconsistency", "Cosine distance (0–1)"),
        use_container_width=True,
    )


def _render_sentiment_scatter(df: pd.DataFrame) -> None:
    needed = ["sentiment_extremity", "sensationalism", "composite_risk"]
    if any(c not in df.columns or df[c].isna().all() for c in needed):
        return
    plot_df = df.dropna(subset=needed).copy()
    plot_df["label"] = plot_df["topic"].apply(lambda t: _truncate(t, 40))
    plot_df["grade"] = plot_df["composite_risk"].apply(grade_topic)
    fig = px.scatter(
        plot_df, x="sentiment_extremity", y="sensationalism",
        color="composite_risk", color_continuous_scale=_RISK_COLORSCALE,
        range_color=[0.0, 1.0], size="articles",
        hover_name="label",
        hover_data={"grade": True, "composite_risk": ":.2f", "articles": True},
        labels={
            "sentiment_extremity": "Sentiment Extremity",
            "sensationalism": "Sensationalism",
            "composite_risk": "Risk",
        },
        title="Sentiment Extremity vs Sensationalism (bubble = article count)",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=20), height=380)
    st.plotly_chart(fig, use_container_width=True)


# ── Person B panel ────────────────────────────────────────────────────────────

def _render_person_b_panel(df: pd.DataFrame) -> None:
    scored = df[df["composite_risk"].notna()].copy()

    if scored.empty:
        st.info(
            "No composite risk scores yet. "
            "Run `python src/scoring/compute_scores.py` to populate."
        )
        return

    # Risk alert
    flagged = scored[scored["composite_risk"] >= _HIGH_RISK]
    if not flagged.empty:
        lines = "  \n".join(
            f"- **{row['topic']}** — Grade {row['grade']}, risk {row['composite_risk']:.3f}"
            for _, row in flagged.iterrows()
        )
        st.error(
            f"**{len(flagged)} topic(s) flagged as potential misinformation** "
            f"(composite risk ≥ {_HIGH_RISK}):\n\n{lines}"
        )
    else:
        st.success("No topics currently exceed the misinformation risk threshold.")

    st.markdown("---")

    # Topic table
    st.subheader("Topic reliability table")
    table_df = scored[[
        "topic", "grade", "reliability_pct", "avg_trust",
        "trust_variance", "coverage_breadth", "coverage_ratio_pct",
        "composite_risk", "articles",
    ]].rename(columns={
        "topic":              "Topic",
        "grade":              "Grade",
        "reliability_pct":    "Reliability %",
        "avg_trust":          "Avg Trust",
        "trust_variance":     "Trust Variance",
        "coverage_breadth":   "Credible Domains",
        "coverage_ratio_pct": "Coverage %",
        "composite_risk":     "Composite Risk",
        "articles":           "Articles",
    })
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Reliability %":  st.column_config.NumberColumn(format="%.1f"),
            "Avg Trust":      st.column_config.NumberColumn(format="%.1f"),
            "Trust Variance": st.column_config.NumberColumn(format="%.2f"),
            "Coverage %":     st.column_config.NumberColumn(format="%.1f"),
            "Composite Risk": st.column_config.ProgressColumn(
                format="%.4f", min_value=0.0, max_value=1.0
            ),
        },
    )

    st.markdown("---")

    # Charts
    col_bar, col_scatter = st.columns(2)

    with col_bar:
        st.subheader("Composite risk by topic")
        bar_df = scored.sort_values("composite_risk", ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=bar_df["composite_risk"],
            y=bar_df["topic"],
            orientation="h",
            marker_color=[_GRADE_COLOUR[g] for g in bar_df["grade"]],
            text=bar_df["grade"],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Composite risk: %{x:.4f}<br>"
                "Grade: %{text}<extra></extra>"
            ),
        ))
        fig_bar.add_vline(
            x=_HIGH_RISK, line_dash="dash", line_color="#e74c3c",
            annotation_text=f"Threshold ({_HIGH_RISK})",
            annotation_position="top right",
        )
        fig_bar.update_layout(
            xaxis=dict(title="Composite risk", range=[0, 1.05]),
            yaxis_title=None,
            margin=dict(l=0, r=50, t=20, b=40),
            height=max(320, len(scored) * 44),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_scatter:
        st.subheader("Source trust vs credible coverage")
        fig_sc = px.scatter(
            scored,
            x="avg_trust",
            y="coverage_ratio_pct",
            size="bubble_size",
            color="composite_risk",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 1],
            hover_name="topic",
            hover_data={
                "grade":              True,
                "avg_trust":          ":.1f",
                "coverage_ratio_pct": ":.1f",
                "coverage_breadth":   True,
                "composite_risk":     ":.4f",
                "bubble_size":        False,
            },
            labels={
                "avg_trust":          "Avg source trust (0–100)",
                "coverage_ratio_pct": "Credible domain coverage (%)",
                "composite_risk":     "Composite risk",
            },
            size_max=32,
        )
        fig_sc.update_layout(
            margin=dict(l=0, r=0, t=20, b=40),
            height=max(320, len(scored) * 44),
            coloraxis_colorbar=dict(title="Risk"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # Grade legend
    st.markdown("---")
    with st.expander("Grade scale reference"):
        leg_cols = st.columns(5)
        for i, (grade, desc) in enumerate([
            ("A", "Reliability ≥ 80%"),
            ("B", "Reliability 60–79%"),
            ("C", "Reliability 40–59%"),
            ("D", "Reliability 20–39%"),
            ("F", "Reliability < 20%"),
        ]):
            leg_cols[i].markdown(
                f"<div style='background:{_GRADE_COLOUR[grade]};padding:8px;"
                f"border-radius:6px;text-align:center'>"
                f"<b>{grade}</b><br><small>{desc}</small></div>",
                unsafe_allow_html=True,
            )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Hot Topics Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    db_path = _sidebar()
    df = load_scored_topics(db_path)

    st.title("🔍 Hot Topics Misinformation Dashboard")

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
        _render_sentiment_scatter(df)

    with tab_b:
        _render_person_b_panel(df)


if __name__ == "__main__":
    main()
