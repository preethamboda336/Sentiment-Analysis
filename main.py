"""
main.py
-------
Master Pipeline Orchestrator

Runs the complete end-to-end pipeline:
  Step 1: Generate synthetic dataset
  Step 2: Clean text
  Step 3: NLP Preprocessing
  Step 4: Feature Extraction
  Step 5: Model Training & Evaluation
  Step 6: Demo Predictions

Run: python main.py
"""

import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_dataset  import generate_dataset
from text_cleaner      import clean_dataframe
from preprocessor      import preprocess_dataframe
from feature_extractor import FeatureExtractor
from train_model       import train_and_evaluate

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   Social Media Sentiment Analysis Dashboard                  ║
║   Full Pipeline — Step by Step                               ║
╚══════════════════════════════════════════════════════════════╝
"""

def run_pipeline():
    print(BANNER)
    total_start = time.time()

    # ── Step 1: Generate Dataset ──────────────────────────────────────────────
    print("STEP 1/5 │ Generating Synthetic Social Media Dataset")
    print("-" * 55)
    if os.path.exists("data/raw/social_media_posts.csv"):
        print("  ⏭ Dataset already exists. Skipping generation.")
        df_raw = pd.read_csv("data/raw/social_media_posts.csv")
    else:
        df_raw = generate_dataset(5000)
    print(f"  ✓ Dataset: {len(df_raw)} posts\n")

    # ── Step 2: Text Cleaning ─────────────────────────────────────────────────
    print("STEP 2/5 │ Text Cleaning")
    print("-" * 55)
    if os.path.exists("data/processed/cleaned_posts.csv"):
        print("  ⏭ Cleaned data exists. Loading...")
        df_cleaned = pd.read_csv("data/processed/cleaned_posts.csv")
    else:
        df_cleaned = clean_dataframe(df_raw)
        os.makedirs("data/processed", exist_ok=True)
        df_cleaned.to_csv("data/processed/cleaned_posts.csv", index=False)
    print(f"  ✓ Cleaned: {len(df_cleaned)} posts\n")

    # ── Step 3: NLP Preprocessing ─────────────────────────────────────────────
    print("STEP 3/5 │ NLP Preprocessing (Stopwords + Lemmatization)")
    print("-" * 55)
    if os.path.exists("data/processed/preprocessed_posts.csv"):
        print("  ⏭ Preprocessed data exists. Loading...")
        df_proc = pd.read_csv("data/processed/preprocessed_posts.csv")
    else:
        df_proc = preprocess_dataframe(df_cleaned)
        df_proc.to_csv("data/processed/preprocessed_posts.csv", index=False)
    df_proc = df_proc.dropna(subset=["processed_text", "sentiment"])
    print(f"  ✓ Preprocessed: {len(df_proc)} posts\n")

    # ── Step 4: Feature Extraction ────────────────────────────────────────────
    print("STEP 4/5 │ TF-IDF Feature Extraction + Train/Test Split")
    print("-" * 55)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df_proc["processed_text"], df_proc["sentiment"],
        test_size=0.2, random_state=42, stratify=df_proc["sentiment"]
    )

    fe = FeatureExtractor()
    X_train, y_train_enc = fe.fit_transform(X_train_text, y_train)
    X_test               = fe.transform(X_test_text)
    y_test_enc           = fe.label_encoder.transform(y_test)
    fe.save()
    print(f"  ✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

    # ── Step 5: Model Training ────────────────────────────────────────────────
    print("STEP 5/5 │ Model Training, Evaluation & Saving")
    print("-" * 55)
    results, best = train_and_evaluate(
        X_train, X_test, y_train_enc, y_test_enc, fe.label_encoder
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print("\n" + "="*55)
    print("  PIPELINE COMPLETE")
    print("="*55)
    print(f"  Total Time     : {elapsed:.1f} seconds")
    print(f"  Best Model     : {best}")
    print(f"  Test Accuracy  : {results[best]['accuracy']*100:.2f}%")
    print(f"  Test F1 Score  : {results[best]['f1_score']*100:.2f}%")
    print("\n  Artifacts:")
    print("    data/raw/social_media_posts.csv")
    print("    data/processed/cleaned_posts.csv")
    print("    data/processed/preprocessed_posts.csv")
    print("    models/sentiment_model.pkl")
    print("    models/tfidf_vectorizer.pkl")
    print("    models/label_encoder.pkl")
    print("    models/metrics.json")
    print("    outputs/plots/confusion_matrix.png")
    print("    outputs/plots/model_comparison.png")
    print("\n  Next step → Launch the dashboard:")
    print("    streamlit run app/dashboard.py")
    print("="*55)


if __name__ == "__main__":
    run_pipeline()
