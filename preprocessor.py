"""
preprocessor.py
---------------
Module 3: NLP Preprocessing (zero external network dependencies)

After cleaning, text still contains stopwords. This module:
  1. Tokenizes text into individual words
  2. Removes stopwords (common words with no sentiment value)
     - Keeps negation words: not, never, no, hardly, etc.
  3. Applies simple suffix-based lemmatization
  4. Rebuilds processed text string for TF-IDF
"""

import os
import re
import pandas as pd

# ── Built-in Stopword Set ─────────────────────────────────────────────────────
NEGATION_WORDS = {
    "not","no","never","none","nobody","nothing","neither","nor","nowhere",
    "hardly","barely","scarcely","without","cannot","cant","wont","dont",
    "doesnt","didnt","wasnt","werent","havent","hasnt","hadnt","shouldnt",
    "wouldnt","couldnt","isnt","arent",
}

_BASE_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","than","too","very","s","t","just","now","ll","re",
    "ve","d","m","will","would","could","should","may","might","shall","same",
    "own","so","also","however","although","though","even","still","yet",
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "well","back","us","via","per","etc","ie","eg","vs","mr","mrs","dr",
}

STOPWORDS = _BASE_STOPWORDS - NEGATION_WORDS


# ── Lightweight Lemmatizer ────────────────────────────────────────────────────

class SimpleLemmatizer:
    """Rule-based suffix lemmatizer. No external dependencies."""
    RULES = [
        (r"ying$", "y"), (r"ied$", "y"), (r"ies$", "y"),
        (r"ning$", "n"), (r"ting$", "t"), (r"ring$", "r"),
        (r"sing$", "s"), (r"ing$", ""),  (r"ed$", ""),
        (r"er$", ""),   (r"est$", ""),
        (r"ches$", "ch"), (r"shes$", "sh"), (r"xes$", "x"),
        (r"ses$", "s"), (r"oes$", "o"),  (r"ves$", "fe"),
        (r"s$", ""),
    ]

    def lemmatize(self, word: str) -> str:
        if len(word) < 5:
            return word
        for pattern, replacement in self.RULES:
            new = re.sub(pattern, replacement, word)
            if new != word and len(new) >= 3:
                return new
        return word

LEMMATIZER = SimpleLemmatizer()


# ── Core NLP Functions ────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    return re.findall(r"\b[a-z']+\b", text.lower())

def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def lemmatize_tokens(tokens: list) -> list:
    return [LEMMATIZER.lemmatize(t) for t in tokens]

def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


# ── DataFrame Preprocessor ────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    print(f"🔤 Preprocessing {len(df)} posts...")
    df = df.copy()
    df["processed_text"] = df[text_col].apply(preprocess_text)
    before = len(df)
    df = df[df["processed_text"].str.strip().str.len() > 3].reset_index(drop=True)
    after = len(df)
    print(f"   Dropped {before - after} empty posts → {after} posts ready.")
    return df


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cleaned_path = "data/processed/cleaned_posts.csv"
    out_path     = "data/processed/preprocessed_posts.csv"

    if not os.path.exists(cleaned_path):
        print("Run src/text_cleaner.py first.")
    else:
        df = pd.read_csv(cleaned_path)
        result = preprocess_dataframe(df)
        result.to_csv(out_path, index=False)
        print(f"\nSaved → {out_path}")
        for i in range(3):
            print(f"\n  Cleaned:   {df['cleaned_text'].iloc[i][:80]}")
            print(f"  Processed: {result['processed_text'].iloc[i][:80]}")
