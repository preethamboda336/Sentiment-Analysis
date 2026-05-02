"""
visualizations.py
-----------------
Module 7: Analytics Visualizations

Generates all project charts and saves to outputs/plots/:
  1. Sentiment distribution (donut chart)
  2. Platform-wise sentiment breakdown (grouped bar)
  3. Brand reputation heatmap
  4. Sentiment over time (line chart)
  5. Top positive / negative keywords (bar charts)
  6. Post engagement (likes) by sentiment (violin plot)

Run: python src/visualizations.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from collections import Counter

# ── Style Configuration ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        120,
})

COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#f59e0b",
    "bg":       "#0f172a",
    "surface":  "#1e293b",
    "text":     "#f1f5f9",
    "accent":   "#6366f1",
}

PALETTE = [COLORS["positive"], COLORS["negative"], COLORS["neutral"]]
OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Sentiment Distribution Donut ──────────────────────────────────────────

def plot_sentiment_distribution(df):
    counts = df["sentiment"].value_counts()
    labels = [f"{s.capitalize()}\n{c} posts\n({c/len(df)*100:.1f}%)"
              for s, c in counts.items()]
    colors = [COLORS[s] for s in counts.index]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts = ax.pie(
        counts.values, labels=None, colors=colors,
        startangle=90, pctdistance=0.85,
        wedgeprops={"linewidth": 3, "edgecolor": "white"},
        counterclock=False
    )
    # Inner circle for donut effect
    centre = plt.Circle((0, 0), 0.6, fc="white")
    ax.add_patch(centre)
    ax.text(0, 0.07, f"{len(df):,}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#1e293b")
    ax.text(0, -0.15, "Total Posts", ha="center", va="center",
            fontsize=10, color="#64748b")

    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=9)
    ax.set_title("Overall Sentiment Distribution", fontsize=14,
                 fontweight="bold", pad=20, color="#1e293b")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_sentiment_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")


# ── 2. Platform-Wise Sentiment ────────────────────────────────────────────────

def plot_platform_sentiment(df):
    pivot = (df.groupby(["platform", "sentiment"])
               .size().unstack(fill_value=0)
               .reindex(columns=["positive", "negative", "neutral"], fill_value=0))

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Platform", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Posts", fontsize=11, fontweight="bold")
    ax.set_title("Sentiment Breakdown by Platform", fontsize=14, fontweight="bold", pad=12)
    ax.legend(title="Sentiment", title_fontsize=10, fontsize=9)
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_platform_sentiment.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")


# ── 3. Brand Reputation Heatmap ───────────────────────────────────────────────

def plot_brand_heatmap(df):
    pivot = (df.groupby(["brand", "sentiment"])
               .size().unstack(fill_value=0)
               .reindex(columns=["positive", "negative", "neutral"], fill_value=0))

    # Normalize to percentages per brand
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        pct, annot=True, fmt=".1f", cmap="RdYlGn",
        ax=ax, linewidths=0.5, linecolor="#e2e8f0",
        vmin=0, vmax=100,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"label": "% of Posts", "shrink": 0.8}
    )
    ax.set_title("Brand Reputation Heatmap (% Sentiment)", fontsize=14,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment", fontsize=11, fontweight="bold")
    ax.set_ylabel("Brand", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_brand_reputation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")


# ── 4. Sentiment Over Time ────────────────────────────────────────────────────

def plot_sentiment_over_time(df):
    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    df2["month"]     = df2["timestamp"].dt.to_period("M")

    monthly = (df2.groupby(["month", "sentiment"])
                  .size().unstack(fill_value=0)
                  .reindex(columns=["positive", "negative", "neutral"], fill_value=0))
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    for col, color in zip(["positive", "negative", "neutral"], PALETTE):
        ax.plot(monthly.index, monthly[col], color=color,
                linewidth=2.5, label=col.capitalize(), marker="o",
                markersize=3, alpha=0.9)
        ax.fill_between(monthly.index, monthly[col], alpha=0.08, color=color)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    ax.set_title("Sentiment Trends Over Time", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Post Count", fontsize=11)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_sentiment_over_time.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")


# ── 5. Top Keywords by Sentiment ─────────────────────────────────────────────

def plot_top_keywords(df):
    try:
        from preprocessor import STOPWORDS
    except ModuleNotFoundError:
        from src.preprocessor import STOPWORDS

    def top_words(texts, n=15):
        words = []
        for t in texts:
            if isinstance(t, str):
                words.extend(t.split())
        filtered = [w for w in words if w not in STOPWORDS and len(w) > 3]
        return Counter(filtered).most_common(n)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sentiments = ["positive", "negative", "neutral"]

    for ax, sentiment in zip(axes, sentiments):
        texts   = df[df["sentiment"] == sentiment]["processed_text"].tolist() \
                  if "processed_text" in df.columns \
                  else df[df["sentiment"] == sentiment]["cleaned_text"].tolist()
        top     = top_words(texts, 12)
        words_  = [w for w, _ in top]
        counts_ = [c for _, c in top]

        bars = ax.barh(words_[::-1], counts_[::-1],
                       color=COLORS[sentiment], alpha=0.85, edgecolor="white")
        ax.set_title(f"Top Keywords — {sentiment.capitalize()}",
                     fontsize=12, fontweight="bold", color=COLORS[sentiment])
        ax.set_xlabel("Frequency", fontsize=9)
        for bar in bars:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(int(bar.get_width())), va="center", fontsize=7)

    plt.suptitle("Most Frequent Words by Sentiment Class",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_top_keywords.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")


# ── 6. Engagement by Sentiment ────────────────────────────────────────────────

def plot_engagement(df):
    if "likes" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Box plot of likes
    order  = ["positive", "negative", "neutral"]
    colors = [COLORS[s] for s in order]
    data   = [df[df["sentiment"] == s]["likes"].clip(0, 3000) for s in order]

    bp = axes[0].boxplot(data, labels=order, patch_artist=True,
                         medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Post Likes by Sentiment", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Likes")

    # Topic sentiment breakdown
    if "topic" in df.columns:
        topic_pct = (df.groupby(["topic", "sentiment"])
                       .size().unstack(fill_value=0)
                       .reindex(columns=order, fill_value=0))
        topic_pct_norm = topic_pct.div(topic_pct.sum(axis=1), axis=0) * 100
        topic_pct_norm[["positive", "negative"]].plot(
            kind="barh", ax=axes[1],
            color=[COLORS["positive"], COLORS["negative"]],
            alpha=0.85, edgecolor="white"
        )
        axes[1].set_title("Positive vs Negative by Topic", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Percentage (%)")
        axes[1].legend(fontsize=9)
        axes[1].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/06_engagement_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")


# ── Run All ───────────────────────────────────────────────────────────────────

def generate_all_visualizations():
    print("\nGenerating Analytics Visualizations...")

    # Load best available data
    if os.path.exists("data/processed/preprocessed_posts.csv"):
        df = pd.read_csv("data/processed/preprocessed_posts.csv")
    elif os.path.exists("data/processed/cleaned_posts.csv"):
        df = pd.read_csv("data/processed/cleaned_posts.csv")
    else:
        df = pd.read_csv("data/raw/social_media_posts.csv")

    plot_sentiment_distribution(df)
    plot_platform_sentiment(df)
    plot_brand_heatmap(df)
    plot_sentiment_over_time(df)
    plot_top_keywords(df)
    plot_engagement(df)

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    generate_all_visualizations()
