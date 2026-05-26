"""Streamlit dashboard — trust score, coverage breadth, and composite risk.

Displays all scored topics from the SQLite database with:
  - KPI row: topic count, avg trust, high-risk count, avg composite risk
  - Risk alert banner for topics above the misinfo threshold
  - Sortable topic table with reliability grades
  - Composite-risk bar chart (colour-coded by grade)
  - Trust vs credible-coverage scatter plot

Run from the project root:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.scoring.compute_scores import _MISINFO_THRESHOLD, grade_topic
from src.utils.db import init_db

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hot Topics — Reliability Dashboard",
    page_icon="🔍",
    layout="wide",
)

_GRADE_COLOUR: dict[str, str] = {
    "A": "#2ecc71",
    "B": "#82b74b",
    "C": "#f1c40f",
    "D": "#e67e22",
    "F": "#e74c3c",
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_conn():
    return init_db()


def _load_topics() -> pd.DataFrame:
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT
            t.id            AS topic_id,
            t.label         AS topic,
            t.item_count,
            ts.avg_trust,
            ts.trust_variance,
            ts.coverage_breadth,
            ts.coverage_ratio,
            ts.composite_risk,
            ts.computed_at
        FROM topics t
        JOIN topic_scores ts ON ts.topic_id = t.id
        WHERE ts.composite_risk IS NOT NULL
        ORDER BY ts.composite_risk DESC
        """
    ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["grade"] = df["composite_risk"].apply(grade_topic)
    df["reliability_pct"] = ((1.0 - df["composite_risk"]) * 100).round(1)
    df["coverage_ratio_pct"] = (df["coverage_ratio"] * 100).round(1)
    df["bubble_size"] = df["coverage_breadth"].clip(lower=1)
    return df


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔍 Hot Topics — Reliability Dashboard")
st.caption("Source trust · Coverage breadth · Composite risk score")

df = _load_topics()

if df.empty:
    st.info(
        "No scored topics found yet. "
        "Run `python src/scoring/compute_scores.py` to populate scores."
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

high_risk_df = df[df["composite_risk"] >= _MISINFO_THRESHOLD]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Topics tracked", len(df))
k2.metric("Avg source trust", f"{df['avg_trust'].mean():.1f} / 100")
k3.metric("High-risk topics", len(high_risk_df))
k4.metric("Avg composite risk", f"{df['composite_risk'].mean():.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Risk alert banner
# ---------------------------------------------------------------------------

if not high_risk_df.empty:
    flagged = "  \n".join(
        f"- **{row['topic']}** — Grade {row['grade']}, "
        f"risk {row['composite_risk']:.3f}"
        for _, row in high_risk_df.iterrows()
    )
    st.error(f"**{len(high_risk_df)} topic(s) flagged as potential misinformation** "
             f"(composite risk ≥ {_MISINFO_THRESHOLD}):\n\n{flagged}")
else:
    st.success("No topics currently exceed the misinformation risk threshold.")

st.divider()

# ---------------------------------------------------------------------------
# Topic table
# ---------------------------------------------------------------------------

st.subheader("Topic reliability table")

table_df = df[[
    "topic", "grade", "reliability_pct", "avg_trust",
    "trust_variance", "coverage_breadth", "coverage_ratio_pct",
    "composite_risk", "item_count",
]].rename(columns={
    "topic":              "Topic",
    "grade":              "Grade",
    "reliability_pct":    "Reliability %",
    "avg_trust":          "Avg Trust",
    "trust_variance":     "Trust Variance",
    "coverage_breadth":   "Credible Domains",
    "coverage_ratio_pct": "Coverage %",
    "composite_risk":     "Composite Risk",
    "item_count":         "Articles",
})

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Reliability %":   st.column_config.NumberColumn(format="%.1f"),
        "Avg Trust":       st.column_config.NumberColumn(format="%.1f"),
        "Trust Variance":  st.column_config.NumberColumn(format="%.2f"),
        "Coverage %":      st.column_config.NumberColumn(format="%.1f"),
        "Composite Risk":  st.column_config.ProgressColumn(
            format="%.4f", min_value=0.0, max_value=1.0
        ),
    },
)

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

col_bar, col_scatter = st.columns(2)

# Bar chart: composite risk per topic, coloured by grade
with col_bar:
    st.subheader("Composite risk by topic")
    bar_df = df.sort_values("composite_risk", ascending=True)

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
        x=_MISINFO_THRESHOLD,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Threshold ({_MISINFO_THRESHOLD})",
        annotation_position="top right",
    )
    fig_bar.update_layout(
        xaxis=dict(title="Composite risk", range=[0, 1.05]),
        yaxis_title=None,
        margin=dict(l=0, r=50, t=20, b=40),
        height=max(320, len(df) * 44),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Scatter: avg_trust vs coverage ratio, bubble = credible domain count
with col_scatter:
    st.subheader("Source trust vs credible coverage")
    fig_sc = px.scatter(
        df,
        x="avg_trust",
        y="coverage_ratio_pct",
        size="bubble_size",
        color="composite_risk",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 1],
        hover_name="topic",
        hover_data={
            "grade":             True,
            "avg_trust":         ":.1f",
            "coverage_ratio_pct": ":.1f",
            "coverage_breadth":  True,
            "composite_risk":    ":.4f",
            "bubble_size":       False,
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
        height=max(320, len(df) * 44),
        coloraxis_colorbar=dict(title="Risk"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ---------------------------------------------------------------------------
# Grade scale legend
# ---------------------------------------------------------------------------

st.divider()
with st.expander("Grade scale reference"):
    leg_cols = st.columns(5)
    for i, (grade, label) in enumerate([
        ("A", "Reliability ≥ 80%"),
        ("B", "Reliability 60–79%"),
        ("C", "Reliability 40–59%"),
        ("D", "Reliability 20–39%"),
        ("F", "Reliability < 20%"),
    ]):
        leg_cols[i].markdown(
            f"<div style='background:{_GRADE_COLOUR[grade]};padding:8px;"
            f"border-radius:6px;text-align:center'>"
            f"<b>{grade}</b><br><small>{label}</small></div>",
            unsafe_allow_html=True,
        )
