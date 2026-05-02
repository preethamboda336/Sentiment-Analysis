# 🧠 Social Media Sentiment Analysis Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.22+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-22c55e?style=for-the-badge)

**An end-to-end NLP pipeline and analytics dashboard that classifies social media posts as Positive, Negative, or Neutral — built for brand reputation management, campaign analysis, and customer feedback intelligence.**

[View Live Demo](#) · [Read Architecture Doc](docs/ARCHITECTURE.md) · [Report Issues](https://github.com)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Industry Relevance](#-industry-relevance)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Model Performance](#-model-performance)
- [Dashboard Features](#-dashboard-features)
- [Screenshots](#-screenshots)
- [Learning Outcomes](#-learning-outcomes)
- [Author](#-author)

---

## 🔍 Overview

This project simulates a **real-world social media analytics system** used by companies like Amazon, Flipkart, Zomato, Netflix, and Swiggy to monitor public sentiment at scale.

The system processes **5,000 synthetic social media posts** across 10 brands and 5 platforms, runs them through a complete NLP pipeline, trains and compares 3 ML classifiers, and presents all insights in a **5-page interactive Streamlit dashboard**.

**Key capabilities:**
- Automatic text cleaning and NLP preprocessing
- TF-IDF feature extraction with bigram support
- Multi-class sentiment classification (Positive / Negative / Neutral)
- Brand reputation heatmap
- Temporal trend analysis
- Live single-text and batch prediction interface

---

## 🎯 Problem Statement

Companies post millions of customer interactions daily across Twitter, Instagram, Reddit, Facebook, and Google Reviews. Manually reviewing this data is:
- **Impossible at scale** (millions of posts per day)
- **Slow** (reactive rather than proactive)
- **Inconsistent** (human reviewers introduce bias)

Sentiment Analysis automates this — enabling teams to **detect brand crises early, measure campaign performance, and understand customer emotions in real time**.

---

## 🏭 Industry Relevance

| Company | Use Case |
|---------|----------|
| **Amazon / Flipkart** | Product review sentiment → improve listings |
| **Zomato / Swiggy** | Delivery/app complaint detection → ops improvement |
| **Netflix** | Content reception analysis → content strategy |
| **Banks / FinTech** | Customer complaint monitoring → regulatory compliance |
| **Political teams** | Public opinion tracking → campaign messaging |
| **Startups** | Competitor reputation benchmarking |

---

## 🏗️ Architecture

```
Raw Social Media Text
        │
        ▼
┌──────────────────┐
│  Text Cleaner    │  URLs, mentions, emojis, HTML, punctuation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  NLP Preprocessor│  Stopword removal, lemmatization, tokenization
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  TF-IDF Extractor│  8000 features, unigrams + bigrams
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ML Classifier   │  Logistic Regression / NB / Random Forest
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Streamlit       │  5-page analytics dashboard
│  Dashboard       │  Live predictions + visualizations
└──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| NLP | Custom tokenizer, rule-based lemmatizer, TF-IDF |
| ML Models | Scikit-learn (LR, NB, RF) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Version Control | Git + GitHub |

---

## 📁 Folder Structure

```
Social-Media-Sentiment-Analysis-Dashboard/
│
├── data/
│   ├── raw/
│   │   └── social_media_posts.csv       ← 5000 synthetic posts
│   └── processed/
│       ├── cleaned_posts.csv            ← After text cleaning
│       └── preprocessed_posts.csv      ← After NLP preprocessing
│
├── src/
│   ├── generate_dataset.py              ← Synthetic data generator
│   ├── text_cleaner.py                  ← URL/emoji/HTML cleaning
│   ├── preprocessor.py                  ← Stopwords + lemmatization
│   ├── feature_extractor.py             ← TF-IDF vectorization
│   ├── train_model.py                   ← Model training + evaluation
│   ├── predict.py                       ← Inference module
│   └── visualizations.py               ← Standalone plot generator
│
├── models/
│   ├── sentiment_model.pkl              ← Best trained classifier
│   ├── tfidf_vectorizer.pkl             ← Fitted TF-IDF vectorizer
│   ├── label_encoder.pkl                ← Label encoder
│   └── metrics.json                     ← Training metrics
│
├── app/
│   └── dashboard.py                     ← Streamlit 5-page dashboard
│
├── outputs/
│   └── plots/
│       ├── 01_sentiment_distribution.png
│       ├── 02_platform_sentiment.png
│       ├── 03_brand_reputation_heatmap.png
│       ├── 04_sentiment_over_time.png
│       ├── 05_top_keywords.png
│       ├── 06_engagement_analysis.png
│       ├── confusion_matrix.png
│       └── model_comparison.png
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb           ← Exploration notebook
│
├── docs/
│   └── ARCHITECTURE.md                  ← Detailed system design
│
├── main.py                              ← Pipeline orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Social-Media-Sentiment-Analysis-Dashboard.git
cd Social-Media-Sentiment-Analysis-Dashboard
```

**2. Create a virtual environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A: Full Pipeline (Recommended for first run)

```bash
python main.py
```

This runs all 5 steps in sequence:
1. Generates 5000 synthetic social media posts
2. Cleans text (removes URLs, emojis, mentions)
3. Applies NLP preprocessing (stopwords, lemmatization)
4. Extracts TF-IDF features
5. Trains and evaluates 3 ML models

Expected output:
```
╔══════════════════════════════════════════════════════════════╗
║   Social Media Sentiment Analysis Dashboard                  ║
╚══════════════════════════════════════════════════════════════╝

STEP 1/5 │ Generating Synthetic Dataset
  ✓ Dataset: 5000 posts

STEP 2/5 │ Text Cleaning
  ✓ Cleaned: 5000 posts

...

  Best Model     : Logistic Regression
  Test Accuracy  : 99.2%
  Test F1 Score  : 99.1%
```

### Option B: Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

Opens at: `http://localhost:8501`

### Option C: Run Individual Modules

```bash
# Generate data only
python src/generate_dataset.py

# Clean text only
python src/text_cleaner.py

# Preprocess only
python src/preprocessor.py

# Train model only
python src/train_model.py

# Run predictions on demo text
python src/predict.py

# Generate all plots
python src/visualizations.py
```

---

## 📊 Model Performance

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| **Logistic Regression** | **99.2%** | **99.1%** | 0.07s |
| Multinomial Naive Bayes | 99.0% | 98.9% | 0.01s |
| Random Forest | 99.2% | 99.1% | 0.80s |

> **Note:** High accuracy is expected with synthetic data. On real-world noisy datasets (Twitter API, Kaggle datasets), expect 80-90% accuracy — which is the industry standard.

**5-Fold Cross-Validation (Best Model):** 99.2% ± 0.4%

---

## 🖥️ Dashboard Features

| Page | What It Shows |
|------|---------------|
| 📊 **Overview** | Global KPIs, donut chart, brand comparison bar, automated insights |
| 🏢 **Brand Analytics** | Per-brand reputation drill-down, topic breakdown, time trends |
| 📡 **Platform Insights** | Cross-platform volume, engagement by sentiment, box plots |
| 📈 **Trend Analysis** | Monthly trends, stacked area chart, volume heatmap |
| 🔮 **Live Predictor** | Real-time single prediction + batch CSV upload + gauge chart |

---

## 📸 Screenshots

> *(Add screenshots to `images/` folder and update paths below)*

| Overview Page | Brand Analytics |
|---|---|
| ![Overview](images/screenshot_overview.png) | ![Brand](images/screenshot_brand.png) |

| Trend Analysis | Live Predictor |
|---|---|
| ![Trends](images/screenshot_trends.png) | ![Predictor](images/screenshot_predictor.png) |

---

## 🎓 Learning Outcomes

By completing this project, you will have demonstrated:

- ✅ **End-to-end NLP pipeline** design and implementation
- ✅ **Text preprocessing** (cleaning, tokenization, lemmatization, stopwords)
- ✅ **TF-IDF feature engineering** with n-grams
- ✅ **Multi-class ML classification** (LR, NB, RF)
- ✅ **Model evaluation** (accuracy, F1, confusion matrix, CV)
- ✅ **Interactive dashboard** development with Streamlit + Plotly
- ✅ **Modular, production-style Python packaging**
- ✅ **Git version control** and professional GitHub presentation

---

## 👤 Author

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- Email: yourname@email.com

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">
⭐ If this project helped you, please give it a star!
</div>
