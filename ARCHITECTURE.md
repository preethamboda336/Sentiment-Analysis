# System Architecture

## Data Flow Diagram

```
INPUT
  └─ 5000 synthetic social media posts
       ├── Brands: Amazon, Flipkart, Zomato, Netflix, Swiggy,
       │           Samsung, Apple, Ola, Uber, BookMyShow
       ├── Platforms: Twitter, Instagram, Facebook, Reddit, Google Reviews
       ├── Topics: delivery, pricing, app, refund, quality, etc.
       └── Labels: positive (45%), negative (35%), neutral (20%)

MODULE 1 — Text Cleaner (src/text_cleaner.py)
  └─ Input:  Raw post text with noise
  └─ Output: Clean ASCII text
  └─ Steps:
       ├── Remove URLs (http/https/www)
       ├── Remove @mentions
       ├── Strip # from hashtags (keep word)
       ├── Remove HTML entities
       ├── Remove emojis (Unicode ranges)
       ├── Remove punctuation
       ├── Fix repeated characters (loooove → love)
       └── Normalize to lowercase + single spaces

MODULE 2 — NLP Preprocessor (src/preprocessor.py)
  └─ Input:  Cleaned text
  └─ Output: Space-separated token string
  └─ Steps:
       ├── Regex tokenization (word boundaries)
       ├── Custom stopword removal
       │    └─ KEEPS negation words: not, never, no, etc.
       └── Rule-based suffix lemmatization
            └─ running→run, boxes→box, loved→love

MODULE 3 — Feature Extractor (src/feature_extractor.py)
  └─ Input:  Preprocessed token strings
  └─ Output: Sparse TF-IDF feature matrix (N × 8000)
  └─ Config:
       ├── max_features: 8000
       ├── ngram_range: (1, 2)  ← captures "not good", "very fast"
       ├── min_df: 3
       ├── max_df: 0.90
       └── sublinear_tf: True   ← log normalization

MODULE 4 — Model Trainer (src/train_model.py)
  └─ Input:  TF-IDF matrix + encoded labels
  └─ Output: Trained classifier + evaluation reports
  └─ Models:
       ├── Logistic Regression (C=1.0, lbfgs solver)
       ├── Multinomial Naive Bayes (alpha=0.1)
       └── Random Forest (200 trees, max_depth=30)
  └─ Evaluation:
       ├── Accuracy, Precision, Recall, F1
       ├── Confusion matrix
       ├── 5-fold stratified cross-validation
       └── Per-class classification report

MODULE 5 — Predictor (src/predict.py)
  └─ Input:  Raw text string
  └─ Output: {sentiment, confidence, display}
  └─ Pipeline: clean → preprocess → vectorize → classify → decode

MODULE 6 — Dashboard (app/dashboard.py)
  └─ Framework: Streamlit + Plotly
  └─ Pages:
       ├── Overview: KPIs, donut, brand bar, insights
       ├── Brand Analytics: per-brand drill-down + heatmap
       ├── Platform Insights: cross-platform comparison
       ├── Trend Analysis: temporal charts
       └── Live Predictor: real-time classification + batch upload
```

## Design Decisions

### Why TF-IDF over BERT?
- No GPU required → accessible to all students
- Fast training (< 1 second per model)
- Interpretable features
- Strong baseline for tabular/structured social media text
- BERT is recommended as a future enhancement for production

### Why Logistic Regression wins?
- Text classification is largely linearly separable in TF-IDF space
- L2 regularization prevents overfitting
- Probabilistic output (predict_proba) enables confidence scores
- Fast inference — critical for real-time dashboards

### Why synthetic data?
- Twitter/Meta API access requires developer accounts and approval
- Synthetic data lets us control class distribution
- Templates based on real-world brand complaint patterns
- Can be easily replaced with real Kaggle datasets (Twitter Sentiment, Yelp, IMDB)
