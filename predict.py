"""
predict.py
----------
Module 6: Sentiment Prediction

Loads the trained model + feature extractor to classify any text.
"""

import os, sys, json, joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from text_cleaner      import clean_text
from preprocessor      import preprocess_text
from feature_extractor import FeatureExtractor


class SentimentPredictor:
    LABEL_EMOJI = {
        "positive": "😊 Positive",
        "negative": "😠 Negative",
        "neutral":  "😐 Neutral",
    }

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._load_assets()

    def _load_assets(self):
        required = ["sentiment_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl"]
        for f in required:
            path = os.path.join(self.model_dir, f)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Run main.py first to train the model. Missing: {path}")
        self.model = joblib.load(os.path.join(self.model_dir, "sentiment_model.pkl"))
        self.fe    = FeatureExtractor.load(self.model_dir)
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        self.metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
        print(f"Model loaded from {self.model_dir}/")

    def predict_single(self, raw_text: str) -> dict:
        cleaned   = clean_text(raw_text)
        processed = preprocess_text(cleaned)
        if not processed.strip():
            processed = cleaned if cleaned.strip() else raw_text

        X      = self.fe.transform(pd.Series([processed]))
        y_enc  = self.model.predict(X)
        label  = self.fe.decode_labels(y_enc)[0]
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probs     = self.model.predict_proba(X)
            classes   = list(self.fe.label_encoder.classes_)
            label_idx = classes.index(label)
            confidence = round(float(probs[0][label_idx]) * 100, 1)

        return {
            "text":       raw_text[:200],
            "sentiment":  label,
            "display":    self.LABEL_EMOJI[label],
            "confidence": confidence,
        }

    def predict_batch(self, texts: list) -> pd.DataFrame:
        return pd.DataFrame([self.predict_single(t) for t in texts])


DEMO_TEXTS = [
    "Absolutely love Zomato! Delivery was super fast and food was amazing",
    "Terrible experience. Amazon support never responded to my refund request",
    "Netflix just released a new series. Will check it later.",
    "The app keeps crashing every time I open it. So frustrating!",
    "Best app update ever! The new features are exactly what I needed.",
]

if __name__ == "__main__":
    predictor = SentimentPredictor()
    print("\n=== SENTIMENT PREDICTIONS ===")
    for text in DEMO_TEXTS:
        result = predictor.predict_single(text)
        print(f"\nText: {text[:60]}...")
        print(f"  → {result['display']} (Confidence: {result['confidence']}%)")
