"""
train_model.py
--------------
Module 5: Model Training & Evaluation

Trains THREE classifiers and selects the best one:
  1. Logistic Regression  – fast, interpretable, strong baseline
  2. Multinomial Naive Bayes  – perfect for text, very fast
  3. Random Forest  – ensemble, handles non-linear patterns

Evaluation includes:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - Cross-validation scores
  - Per-class classification report
  - Best model selection and saving

Run: python src/train_model.py
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model       import LogisticRegression
from sklearn.naive_bayes        import MultinomialNB
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (accuracy_score, classification_report,
                                        confusion_matrix, f1_score)

# Local modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import FeatureExtractor


# ── Model Definitions ─────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        random_state=42, n_jobs=-1
    ),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
}


# ── Training Pipeline ─────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test,
                       label_encoder, output_dir="outputs"):
    """
    Train all models, evaluate, produce visualizations, save best model.
    Returns dict of results and the best model name.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs("models", exist_ok=True)

    class_names = list(label_encoder.classes_)
    results     = {}

    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    for name, model in MODELS.items():
        print(f"\n▶ Training: {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")

        print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  Time     : {train_time:.2f}s")

        results[name] = {
            "model":      model,
            "accuracy":   round(acc, 4),
            "f1_score":   round(f1, 4),
            "train_time": round(train_time, 2),
            "y_pred":     y_pred,
        }

    # ── Cross-Validation on Best Model ────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["f1_score"])
    best_model = results[best_name]["model"]

    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Running 5-Fold Cross-Validation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_train, y_train,
                                cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"   CV Scores: {[round(s,4) for s in cv_scores]}")
    print(f"   CV Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Full Classification Report ─────────────────────────────────────────────
    y_pred_best = results[best_name]["y_pred"]
    report      = classification_report(y_test, y_pred_best,
                                        target_names=class_names, output_dict=True)
    report_text = classification_report(y_test, y_pred_best,
                                        target_names=class_names)
    print(f"\n📋 Classification Report ({best_name}):\n{report_text}")

    # ── Confusion Matrix Plot ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred_best)
    _plot_confusion_matrix(cm, class_names, best_name, output_dir)

    # ── Accuracy Comparison Bar Chart ─────────────────────────────────────────
    _plot_model_comparison(results, output_dir)

    # ── Save Best Model & Metrics ──────────────────────────────────────────────
    joblib.dump(best_model, "models/sentiment_model.pkl")
    print(f"\n✅ Best model saved → models/sentiment_model.pkl")

    metrics_summary = {
        "best_model":    best_name,
        "accuracy":      results[best_name]["accuracy"],
        "f1_score":      results[best_name]["f1_score"],
        "cv_mean":       round(cv_scores.mean(), 4),
        "cv_std":        round(cv_scores.std(), 4),
        "class_names":   class_names,
        "all_models": {
            k: {"accuracy": v["accuracy"], "f1_score": v["f1_score"]}
            for k, v in results.items()
        },
        "classification_report": report,
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print("✅ Metrics saved → models/metrics.json")

    return results, best_name


# ── Plot Helpers ──────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm, class_names, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved → {path}")


def _plot_model_comparison(results, output_dir):
    names     = list(results.keys())
    accs      = [results[n]["accuracy"] * 100 for n in names]
    f1s       = [results[n]["f1_score"]  * 100 for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy (%)",  color="#4A90E2", alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1 Score (%)", color="#50C878", alpha=0.85)

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Model Comparison – Accuracy vs F1 Score",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Model comparison saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_path = "data/processed/preprocessed_posts.csv"

    if not os.path.exists(data_path):
        print("⚠ Run src/preprocessor.py first.")
    else:
        df = pd.read_csv(data_path)
        df = df.dropna(subset=["processed_text", "sentiment"])

        # Train/test split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["processed_text"], df["sentiment"],
            test_size=0.2, random_state=42, stratify=df["sentiment"]
        )

        # Feature extraction
        fe = FeatureExtractor()
        X_train, y_train_enc = fe.fit_transform(X_train_text, y_train)
        X_test               = fe.transform(X_test_text)
        y_test_enc           = fe.label_encoder.transform(y_test)
        fe.save()

        # Train and evaluate
        results, best = train_and_evaluate(
            X_train, X_test, y_train_enc, y_test_enc,
            fe.label_encoder
        )

        print(f"\n🎯 Training complete. Best model: {best}")
        print(f"   Accuracy: {results[best]['accuracy']*100:.2f}%")
        print(f"   F1 Score: {results[best]['f1_score']*100:.2f}%")
