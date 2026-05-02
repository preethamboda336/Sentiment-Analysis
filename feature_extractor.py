"""
feature_extractor.py
---------------------
Module 4: Feature Extraction using TF-IDF

Converts text strings into numerical feature vectors that ML models
can understand.

TF-IDF (Term Frequency – Inverse Document Frequency):
  TF  = How often a word appears in THIS post
  IDF = How rare the word is ACROSS ALL posts
  Score = TF × IDF → high score = unique, important word

We use TF-IDF (not Bag of Words) because:
  - It down-weights common words like "the", "I", "my"
  - It up-weights rare, meaningful words
  - It produces a sparse matrix — efficient for large text

Run standalone: python src/feature_extractor.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# ── Configuration ─────────────────────────────────────────────────────────────
TFIDF_CONFIG = {
    "max_features": 8000,   # Top 8000 most important terms
    "ngram_range":  (1, 2), # Unigrams + bigrams (e.g., "not good", "very fast")
    "min_df":       3,      # Ignore terms in < 3 documents (noise)
    "max_df":       0.90,   # Ignore terms in > 90% docs (too common)
    "sublinear_tf": True,   # Apply log normalization: 1 + log(TF)
    "analyzer":     "word",
    "strip_accents": "unicode",
}

# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Wraps TF-IDF vectorizer + LabelEncoder.
    Provides fit_transform() for training and transform() for inference.
    """

    def __init__(self):
        self.vectorizer     = TfidfVectorizer(**TFIDF_CONFIG)
        self.label_encoder  = LabelEncoder()
        self.is_fitted      = False

    def fit_transform(self, texts: pd.Series, labels: pd.Series):
        """
        Fit vectorizer on training text and encode labels.
        Returns: (X_sparse_matrix, y_encoded_array)
        """
        print(f"📊 Fitting TF-IDF on {len(texts)} posts...")

        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)

        self.is_fitted = True

        print(f"   Vocabulary size:  {len(self.vectorizer.vocabulary_):,} terms")
        print(f"   Feature matrix:   {X.shape[0]} posts × {X.shape[1]} features")
        print(f"   Classes:          {list(self.label_encoder.classes_)}")

        return X, y

    def transform(self, texts: pd.Series):
        """
        Transform new text using the already-fitted vectorizer.
        Returns: X_sparse_matrix
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted first.")
        return self.vectorizer.transform(texts)

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Convert numeric labels back to strings."""
        return self.label_encoder.inverse_transform(y_encoded)

    def save(self, model_dir: str = "models"):
        """Persist vectorizer and encoder to disk."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.vectorizer,    os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        joblib.dump(self.label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
        print(f"✅ Feature extractor saved → {model_dir}/")

    @classmethod
    def load(cls, model_dir: str = "models"):
        """Load fitted vectorizer and encoder from disk."""
        fe = cls()
        fe.vectorizer    = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        fe.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        fe.is_fitted     = True
        return fe

    def get_top_features(self, n: int = 20) -> list:
        """Return top-N features by IDF score (most informative terms)."""
        if not self.is_fitted:
            return []
        idf_scores = self.vectorizer.idf_
        vocab      = {v: k for k, v in self.vectorizer.vocabulary_.items()}
        top_idx    = np.argsort(idf_scores)[-n:][::-1]
        return [(vocab[i], round(idf_scores[i], 4)) for i in top_idx]


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    data_path = "data/processed/preprocessed_posts.csv"

    if not os.path.exists(data_path):
        print("⚠ Preprocessed data not found. Run src/preprocessor.py first.")
    else:
        df = pd.read_csv(data_path)

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["processed_text"], df["sentiment"],
            test_size=0.2, random_state=42, stratify=df["sentiment"]
        )

        fe = FeatureExtractor()
        X_train, y_train_enc = fe.fit_transform(X_train_text, y_train)
        X_test               = fe.transform(X_test_text)

        fe.save()

        print(f"\nTrain shape: {X_train.shape}")
        print(f"Test shape:  {X_test.shape}")

        print("\nTop 10 most informative terms:")
        for term, score in fe.get_top_features(10):
            print(f"  {term:30s}  IDF={score}")
