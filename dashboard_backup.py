"""
dashboard.py
------------
Social Media Sentiment Analysis Dashboard
Industry-grade Streamlit application with 5 analytical pages.

Run: streamlit run app/dashboard.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express     as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure src/ modules are available
# Robustly locate the project root (directory that contains the `data` folder)
def _find_project_root(max_up=4):
    p = os.path.abspath(os.path.dirname(__file__))
    for _ in range(max_up):
        if os.path.exists(os.path.join(p, "data")):
            return p
        p = os.path.dirname(p)
    # Fallback: use the file's directory
    return os.path.abspath(os.path.dirname(__file__))

ROOT = _find_project_root()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "SentimentIQ Dashboard",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
    menu_items  = {
        "Get help":    "https://github.com",
        "Report a bug":"https://github.com",
        "About":       "Social Media Sentiment Analysis Dashboard — Portfolio Project",
    }
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography & Fonts ──────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { 
    padding: 8px 12px; border-radius: 8px;
    transition: background 0.2s;
}

/* ── Main Area ───────────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── KPI Cards ───────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 22px 28px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.2);
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 4px;
}
.kpi-delta {
    font-size: 0.82rem;
    margin-top: 6px;
    font-weight: 500;
}
.kpi-positive { color: #22c55e; }
.kpi-negative { color: #ef4444; }
.kpi-neutral  { color: #f59e0b; }

/* ── Section Headers ─────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,102,241,0.3);
}

/* ── Prediction Box ──────────────────────────────────── */
.prediction-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    border: 1px solid rgba(99,102,241,0.3);
}
.prediction-label {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.prediction-confidence {
    font-size: 1rem;
    color: #94a3b8;
}

/* ── Insight Card ────────────────────────────────────── */
.insight-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-left: 4px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #e2e8f0;
    font-size: 0.9rem;
}

/* ── Page Title ──────────────────────────────────────── */
.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* ── Plotly charts transparent bg ───────────────────── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
COLOR_MAP = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#f59e0b",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "font":          {"color": "#e2e8f0", "family": "Space Grotesk"},
        "xaxis":         {"gridcolor": "rgba(255,255,255,0.07)", "linecolor": "rgba(255,255,255,0.1)"},
        "yaxis":         {"gridcolor": "rgba(255,255,255,0.07)", "linecolor": "rgba(255,255,255,0.1)"},
        "legend":        {"bgcolor": "rgba(0,0,0,0)"},
        "margin":        {"t": 40, "b": 40, "l": 50, "r": 20},
    }
}


def _apply_plotly_layout(fig, **overrides):
    """Apply PLOTLY_TEMPLATE layout to a figure, merging any overrides.

    This avoids passing duplicate keyword args (e.g., 'legend') to
    plotly's update_layout by merging template and override dicts.
    """
    layout = dict(PLOTLY_TEMPLATE.get("layout", {}))
    # Merge legend dicts specially to avoid duplicate keyword errors
    if "legend" in overrides:
        tpl_leg = dict(layout.get("legend", {}))
        over_leg = dict(overrides.pop("legend"))
        tpl_leg.update(over_leg)
        layout["legend"] = tpl_leg
    # Remaining overrides (height, showlegend, xaxis_title, etc.)
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


# ── Data Loading ──────────────────────────────────────────────────────────────

def _dataset_paths():
    base = ROOT
    return [
        os.path.join(base, "data/processed/preprocessed_posts.csv"),
        os.path.join(base, "data/processed/cleaned_posts.csv"),
        os.path.join(base, "data/raw/social_media_posts.csv"),
    ]


def _get_paths_mtime(paths):
    mtimes = []
    for p in paths:
        try:
            mtimes.append(os.path.getmtime(p))
        except Exception:
            mtimes.append(0)
    return tuple(mtimes)


@st.cache_data(show_spinner=False)
def _load_data_cached(paths_mtime: tuple):
    # Read files in preferred order; cache key includes file mtimes
    for path in _dataset_paths():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["month"]     = df["timestamp"].dt.to_period("M").astype(str)
                df["date"]      = df["timestamp"].dt.date
            return df
    return pd.DataFrame()


def load_data():
    paths = _dataset_paths()
    mtimes = _get_paths_mtime(paths)
    return _load_data_cached(mtimes)

@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        from predict import SentimentPredictor
        model_dir = os.path.join(ROOT, "models")
        return SentimentPredictor(model_dir)
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(ROOT, "models/metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ── Sidebar Navigation ────────────────────────────────────────────────────────

def render_sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 12px 0 20px 0;">
            <div style="font-size:2rem;">🧠</div>
            <div style="font-size:1.3rem; font-weight:700; color:#f1f5f9; 
                        letter-spacing:-0.5px;">SentimentIQ</div>
            <div style="font-size:0.72rem; color:#6366f1; font-weight:600;
                        text-transform:uppercase; letter-spacing:2px;">
                Analytics Dashboard
            </div>
        </div>
        <hr style="border-color:rgba(99,102,241,0.25); margin-bottom:20px">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["📊  Overview",
             "🏢  Brand Analytics",
             "📡  Platform Insights",
             "📈  Trend Analysis",
             "🔮  Live Predictor"],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color:rgba(99,102,241,0.2);margin:20px 0'>",
                    unsafe_allow_html=True)
        st.markdown("**Filters**")

        brands    = sorted(df["brand"].unique()) if "brand" in df.columns else []
        platforms = sorted(df["platform"].unique()) if "platform" in df.columns else []

        sel_brands = st.multiselect("Brands", brands,
                                    placeholder="All brands",
                                    key="filter_brands")
        sel_plats  = st.multiselect("Platforms", platforms,
                                    placeholder="All platforms",
                                    key="filter_platforms")

        st.markdown("<hr style='border-color:rgba(99,102,241,0.2);margin:20px 0'>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.72rem;color:#475569;text-align:center'>"
            "Portfolio Project · Built with Python<br>"
            "5,000 synthetic posts · 3 ML models"
            "</div>",
            unsafe_allow_html=True
        )

    return page, sel_brands, sel_plats


def apply_filters(df, brands, platforms):
    if brands:
        df = df[df["brand"].isin(brands)]
    if platforms:
        df = df[df["platform"].isin(platforms)]
    return df


# ── Page 1: Overview ──────────────────────────────────────────────────────────

def page_overview(df, metrics):
    st.markdown("""
    <div class="page-title">📊 Sentiment Overview</div>
    <div class="page-subtitle">
        Real-time sentiment analytics across brands, platforms, and topics
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available. Run main.py first.")
        return

    total    = len(df)
    counts   = df["sentiment"].value_counts()
    pos_pct  = round(counts.get("positive", 0) / total * 100, 1)
    neg_pct  = round(counts.get("negative", 0) / total * 100, 1)
    neu_pct  = round(counts.get("neutral",  0) / total * 100, 1)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, total, "Total Posts", "#6366f1", ""),
        (c2, f"{pos_pct}%", "Positive", "#22c55e", "positive"),
        (c3, f"{neg_pct}%", "Negative", "#ef4444", "negative"),
        (c4, f"{neu_pct}%", "Neutral",  "#f59e0b", "neutral"),
        (c5, f"{metrics.get('accuracy', 0)*100:.1f}%" if metrics else "—",
             "Model Accuracy", "#818cf8", ""),
    ]
    for col, val, label, color, cls in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{color}">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Donut + Bar
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="section-header">Sentiment Split</div>',
                    unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            values  = [counts.get("positive",0), counts.get("negative",0),
                       counts.get("neutral",0)],
            labels  = ["Positive", "Negative", "Neutral"],
            marker_colors=[COLOR_MAP["positive"], COLOR_MAP["negative"],
                           COLOR_MAP["neutral"]],
            hole    = 0.62,
            textinfo= "percent",
            textfont= {"size": 13, "color": "white"},
            hovertemplate="<b>%{label}</b><br>%{value} posts<br>%{percent}<extra></extra>"
        ))
        _apply_plotly_layout(fig_donut, height=300,
                     showlegend=True,
                     legend={"orientation": "h", "y": -0.2,
                         "x": 0.5, "xanchor": "center"})
        fig_donut.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:10'>posts</span>",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 16, "color": "#f1f5f9"}
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Brand Sentiment Overview</div>',
                    unsafe_allow_html=True)
        if "brand" in df.columns:
            brand_pivot = (df.groupby(["brand", "sentiment"])
                             .size().unstack(fill_value=0)
                             .reindex(columns=["positive","negative","neutral"],
                                      fill_value=0))
            brand_pct   = brand_pivot.div(brand_pivot.sum(axis=1), axis=0) * 100
            brand_pct   = brand_pct.sort_values("positive", ascending=True)

            fig_bar = go.Figure()
            for sent in ["positive","negative","neutral"]:
                fig_bar.add_trace(go.Bar(
                    name      = sent.capitalize(),
                    y         = brand_pct.index,
                    x         = brand_pct[sent],
                    orientation = "h",
                    marker_color= COLOR_MAP[sent],
                    opacity   = 0.88,
                    hovertemplate=(
                        f"<b>%{{y}}</b><br>{sent.capitalize()}: %{{x:.1f}}%<extra></extra>"
                    )
                ))
            _apply_plotly_layout(
                fig_bar,
                barmode="stack", height=300,
                xaxis_title="Percentage (%)",
                legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Row 3: Platform breakdown
    if "platform" in df.columns:
        st.markdown('<div class="section-header">Platform Distribution</div>',
                    unsafe_allow_html=True)
        plat_data  = df.groupby(["platform","sentiment"]).size().reset_index(name="count")

        fig_plat = px.bar(
            plat_data, x="platform", y="count", color="sentiment",
            color_discrete_map=COLOR_MAP, barmode="group",
            labels={"count": "Posts", "platform": "Platform"},
            template="plotly_dark",
        )
        _apply_plotly_layout(fig_plat, height=280)
        st.plotly_chart(fig_plat, use_container_width=True)

    # Insights
    st.markdown('<div class="section-header">💡 Automated Insights</div>',
                unsafe_allow_html=True)
    if "brand" in df.columns:
        brand_pos = brand_pct["positive"].idxmax()
        brand_neg = (df[df["sentiment"]=="negative"]["brand"]
                       .value_counts().idxmax())
        top_topic = (df[df["sentiment"]=="negative"]["topic"]
                       .value_counts().idxmax()
                     if "topic" in df.columns else "N/A")
        most_act  = (df["platform"].value_counts().idxmax()
                     if "platform" in df.columns else "N/A")

        insights = [
            f"🏆 <b>{brand_pos}</b> has the strongest positive sentiment score across all brands.",
            f"⚠️ <b>{brand_neg}</b> receives the most negative feedback — consider reputation management.",
            f"🔥 <b>{top_topic.capitalize()}</b> is the most complained-about topic across all brands.",
            f"📡 <b>{most_act}</b> is the most active platform in your dataset.",
            f"📊 Positive posts dominate at <b>{pos_pct}%</b> — overall brand health looks strong."
              if pos_pct > 50 else
            f"📊 Negative sentiment at <b>{neg_pct}%</b> warrants urgent attention.",
        ]
        for ins in insights:
            st.markdown(f'<div class="insight-card">{ins}</div>',
                        unsafe_allow_html=True)


# ── Page 2: Brand Analytics ───────────────────────────────────────────────────

def page_brand_analytics(df):
    st.markdown('<div class="page-title">🏢 Brand Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Deep-dive reputation analysis per brand</div>',
                unsafe_allow_html=True)

    if "brand" not in df.columns:
        st.warning("Brand column not found.")
        return

    # Brand selector
    brands   = sorted(df["brand"].unique())
    sel_brand = st.selectbox("Select Brand to Analyse", brands)
    bdf       = df[df["brand"] == sel_brand]

    c1, c2, c3, c4 = st.columns(4)
    counts = bdf["sentiment"].value_counts()
    total  = len(bdf)
    for col, sent, color in zip(
        [c1, c2, c3, c4],
        ["positive","negative","neutral"],
        ["#22c55e","#ef4444","#f59e0b"]
    ):
        n   = counts.get(sent, 0)
        pct = round(n / total * 100, 1)
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{color}">{pct}%</div>
                <div class="kpi-label">{sent.capitalize()}</div>
                <div class="kpi-delta">{n} posts</div>
            </div>
            """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:#6366f1">{total:,}</div>
            <div class="kpi-label">Total Mentions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Topic Breakdown</div>',
                    unsafe_allow_html=True)
        if "topic" in df.columns:
            topic_data = bdf.groupby(["topic","sentiment"]).size().reset_index(name="count")
            fig = px.bar(
                topic_data, x="topic", y="count", color="sentiment",
                color_discrete_map=COLOR_MAP, barmode="stack",
                template="plotly_dark"
            )
            _apply_plotly_layout(fig, height=300, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Sentiment Over Time</div>',
                    unsafe_allow_html=True)
        if "month" in df.columns:
            ts = (bdf.groupby(["month","sentiment"])
                    .size().unstack(fill_value=0)
                    .reindex(columns=["positive","negative","neutral"], fill_value=0))
            ts.index = pd.to_datetime(ts.index)
            ts = ts.reset_index().melt(id_vars="month",
                                       var_name="sentiment",
                                       value_name="count")
            fig2 = px.line(
                ts, x="month", y="count", color="sentiment",
                color_discrete_map=COLOR_MAP,
                markers=True, template="plotly_dark"
            )
            _apply_plotly_layout(fig2, height=300)
            st.plotly_chart(fig2, use_container_width=True)

    # Brand-vs-brand comparison
    st.markdown('<div class="section-header">Brand Comparison Radar</div>',
                unsafe_allow_html=True)
    brand_pivot = (df.groupby(["brand","sentiment"])
                     .size().unstack(fill_value=0)
                     .reindex(columns=["positive","negative","neutral"], fill_value=0))
    brand_pct   = brand_pivot.div(brand_pivot.sum(axis=1), axis=0) * 100

    fig_heat = px.imshow(
        brand_pct.T, color_continuous_scale="RdYlGn",
        aspect="auto", template="plotly_dark",
        labels={"color": "% Posts"},
        zmin=0, zmax=100
    )
    _apply_plotly_layout(fig_heat, height=220)
    st.plotly_chart(fig_heat, use_container_width=True)


# ── Page 3: Platform Insights ─────────────────────────────────────────────────

def page_platform_insights(df):
    st.markdown('<div class="page-title">📡 Platform Insights</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Cross-platform sentiment comparison and volume analysis</div>',
                unsafe_allow_html=True)

    if "platform" not in df.columns:
        st.warning("Platform column not found.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Platform Volume</div>',
                    unsafe_allow_html=True)
        plat_count = df["platform"].value_counts().reset_index()
        plat_count.columns = ["platform", "count"]
        fig = px.bar(
            plat_count.sort_values("count"),
            x="count", y="platform", orientation="h",
            color="count", color_continuous_scale="Blues",
            template="plotly_dark"
        )
        _apply_plotly_layout(fig, height=280, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Sentiment Mix per Platform</div>',
                    unsafe_allow_html=True)
        pdata = df.groupby(["platform","sentiment"]).size().reset_index(name="count")
        fig2  = px.bar(
            pdata, x="platform", y="count", color="sentiment",
            color_discrete_map=COLOR_MAP, barmode="stack",
            template="plotly_dark"
        )
        _apply_plotly_layout(fig2, height=280)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: likes vs sentiment
    if "likes" in df.columns:
        st.markdown('<div class="section-header">Engagement (Likes) by Platform & Sentiment</div>',
                    unsafe_allow_html=True)
        sample_df = df.sample(min(2000, len(df)), random_state=42)
        fig3 = px.box(
            sample_df, x="platform", y="likes", color="sentiment",
            color_discrete_map=COLOR_MAP, template="plotly_dark",
            points=False
        )
        _apply_plotly_layout(fig3, height=300)
        st.plotly_chart(fig3, use_container_width=True)


# ── Page 4: Trend Analysis ────────────────────────────────────────────────────

def page_trend_analysis(df):
    st.markdown('<div class="page-title">📈 Trend Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Temporal sentiment patterns and volume trends</div>',
                unsafe_allow_html=True)

    if "timestamp" not in df.columns:
        st.warning("Timestamp column not found.")
        return

    df2 = df.copy()
    df2["date"]  = pd.to_datetime(df2["timestamp"]).dt.date
    df2["month"] = pd.to_datetime(df2["timestamp"]).dt.to_period("M").astype(str)

    # Monthly trend
    st.markdown('<div class="section-header">Monthly Sentiment Trends</div>',
                unsafe_allow_html=True)
    monthly = (df2.groupby(["month","sentiment"])
                 .size().unstack(fill_value=0)
                 .reindex(columns=["positive","negative","neutral"], fill_value=0))
    monthly_idx = pd.to_datetime(monthly.index)
    fig_trend   = go.Figure()
    for sent in ["positive","negative","neutral"]:
        fig_trend.add_trace(go.Scatter(
            x=monthly_idx, y=monthly[sent], mode="lines+markers",
            name=sent.capitalize(), line={"color": COLOR_MAP[sent], "width": 2.5},
            fill="tozeroy", fillcolor=COLOR_MAP[sent].replace(")", ",0.08)").replace("rgb","rgba")
                            if "rgb" in COLOR_MAP[sent] else COLOR_MAP[sent],
            marker={"size": 5},
        ))
    _apply_plotly_layout(fig_trend, height=320, hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Sentiment Share Over Time (%)</div>',
                    unsafe_allow_html=True)
        monthly_norm = monthly.div(monthly.sum(axis=1), axis=0) * 100
        fig_area     = go.Figure()
        for sent in ["positive","negative","neutral"]:
            fig_area.add_trace(go.Scatter(
                x=monthly_idx, y=monthly_norm[sent],
                name=sent.capitalize(),
                mode="lines",
                stackgroup="one",
                line={"color": COLOR_MAP[sent], "width": 1},
                fillcolor=COLOR_MAP[sent],
            ))
        _apply_plotly_layout(fig_area, height=280, yaxis_title="Percentage (%)")
        st.plotly_chart(fig_area, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Monthly Post Volume</div>',
                    unsafe_allow_html=True)
        vol = df2.groupby("month").size().reset_index(name="count")
        vol["month_dt"] = pd.to_datetime(vol["month"])
        fig_vol = go.Figure(go.Bar(
            x=vol["month_dt"], y=vol["count"],
            marker_color="#6366f1", opacity=0.8
        ))
        _apply_plotly_layout(fig_vol, height=280, yaxis_title="Posts")
        st.plotly_chart(fig_vol, use_container_width=True)


# ── Page 5: Live Predictor ────────────────────────────────────────────────────

def page_live_predictor(predictor):
    st.markdown('<div class="page-title">🔮 Live Sentiment Predictor</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Enter any social media post to get an instant '
        'sentiment classification</div>',
        unsafe_allow_html=True
    )

    if predictor is None:
        st.error("⚠️ Model not loaded. Run `python main.py` first to train the model.")
        return

    # Quick examples
    st.markdown("**Try an example:**")
    ex_cols = st.columns(3)
    examples = [
        ("😊 Positive", "Absolutely love Zomato! Delivery was super fast and food was perfect!"),
        ("😠 Negative", "Terrible experience with Amazon. Support never resolved my refund."),
        ("😐 Neutral",  "Just comparing Ola and Uber pricing before making my decision today."),
    ]
    user_text = ""
    for col, (label, ex) in zip(ex_cols, examples):
        with col:
            if st.button(label, use_container_width=True, key=f"ex_{label}"):
                user_text = ex

    st.markdown("<br>", unsafe_allow_html=True)
    input_text = st.text_area(
        "Enter your social media post:",
        value=user_text,
        height=120,
        placeholder="Type or paste a social media post, comment, or review here...",
        max_chars=500,
    )

    if st.button("🧠 Analyse Sentiment", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text to analyse.")
        else:
            with st.spinner("Analysing..."):
                time.sleep(0.3)
                result = predictor.predict_single(input_text)

            sent   = result["sentiment"]
            color  = COLOR_MAP[sent]
            emoji  = {"positive": "😊", "negative": "😠", "neutral": "😐"}[sent]
            conf   = result["confidence"]

            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size:3rem;">{emoji}</div>
                <div class="prediction-label" style="color:{color}">
                    {sent.upper()}
                </div>
                <div class="prediction-confidence">
                    Model Confidence: <b>{conf}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = conf,
                title = {"text": "Confidence Score", "font": {"color": "#e2e8f0"}},
                gauge = {
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar":  {"color": color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 40],  "color": "rgba(239,68,68,0.15)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                        {"range": [70, 100],"color": "rgba(34,197,94,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": conf
                    }
                },
                number={"suffix": "%", "font": {"color": color, "size": 36}}
            ))
            _apply_plotly_layout(fig_gauge, height=220)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Batch upload
    st.markdown("---")
    st.markdown('<div class="section-header">Batch File Prediction</div>',
                unsafe_allow_html=True)
    st.markdown("Upload a CSV with a `text` column to classify all posts at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        if "text" not in batch_df.columns:
            st.error("CSV must have a column named `text`.")
        else:
            with st.spinner(f"Predicting {len(batch_df)} posts..."):
                results = predictor.predict_batch(batch_df["text"].fillna("").tolist())
            batch_df["predicted_sentiment"] = results["sentiment"].values
            batch_df["confidence"]          = results["confidence"].values

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(batch_df[["text","predicted_sentiment","confidence"]].head(20),
                             use_container_width=True)
            with col2:
                vc  = batch_df["predicted_sentiment"].value_counts()
                fig = px.pie(values=vc.values, names=vc.index,
                             color=vc.index, color_discrete_map=COLOR_MAP,
                             hole=0.55, template="plotly_dark")
                _apply_plotly_layout(fig, height=300)
                st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "⬇️ Download Results CSV",
                data=batch_df.to_csv(index=False),
                file_name="sentiment_predictions.csv",
                mime="text/csv",
            )


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Loading data..."):
        df        = load_data()
        predictor = load_predictor()
        metrics   = load_metrics()

    page, sel_brands, sel_plats = render_sidebar(df)
    filtered_df = apply_filters(df, sel_brands, sel_plats)

    if "Overview" in page:
        page_overview(filtered_df, metrics)
    elif "Brand" in page:
        page_brand_analytics(filtered_df)
    elif "Platform" in page:
        page_platform_insights(filtered_df)
    elif "Trend" in page:
        page_trend_analysis(filtered_df)
    elif "Predictor" in page:
        page_live_predictor(predictor)


if __name__ == "__main__":
    main()
