"""
text_cleaner.py
---------------
Module 2: Cleans raw social media text.

Social media posts contain noise: URLs, mentions, hashtags, emojis,
punctuation, HTML entities, and inconsistent casing. This module
removes/normalizes all of that so the NLP model gets clean input.

Run standalone: python src/text_cleaner.py
"""

import re
import string
import pandas as pd
import os


# ── Core Cleaning Functions ───────────────────────────────────────────────────

def remove_urls(text: str) -> str:
    """Remove http/https links."""
    return re.sub(r"http\S+|www\.\S+", "", text)

def remove_mentions(text: str) -> str:
    """Remove @username mentions."""
    return re.sub(r"@\w+", "", text)

def remove_hashtags(text: str) -> str:
    """Remove # symbol but KEEP the word (it carries sentiment signal)."""
    return re.sub(r"#(\w+)", r"\1", text)

def remove_html_entities(text: str) -> str:
    """Remove HTML tags and entities like &amp; &lt; etc."""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    return text

def remove_emojis(text: str) -> str:
    """Remove all emoji characters (Unicode ranges)."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"   # emoticons
        u"\U0001F300-\U0001F5FF"   # symbols & pictographs
        u"\U0001F680-\U0001F6FF"   # transport & map
        u"\U0001F1E0-\U0001F1FF"   # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(" ", text)

def remove_punctuation(text: str) -> str:
    """Remove punctuation except apostrophes (for contractions)."""
    cleaned = re.sub(r"[^\w\s']", " ", text)
    return cleaned

def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces into one."""
    return re.sub(r"\s+", " ", text).strip()

def normalize_case(text: str) -> str:
    """Convert to lowercase."""
    return text.lower()

def remove_numbers(text: str) -> str:
    """Remove standalone numbers (optional — kept as False by default)."""
    return re.sub(r"\b\d+\b", "", text)

def fix_repeated_chars(text: str) -> str:
    """Normalize repeated characters: 'loooove' → 'love'."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


# ── Master Pipeline ───────────────────────────────────────────────────────────

def clean_text(text: str, remove_nums: bool = False) -> str:
    """
    Full cleaning pipeline applied in order:
    raw text → urls removed → mentions removed → hashtag # removed
    → html stripped → emojis removed → punctuation removed
    → repeated chars normalized → lowercase → whitespace normalized
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_html_entities(text)
    text = remove_emojis(text)
    text = remove_punctuation(text)
    text = fix_repeated_chars(text)
    text = normalize_case(text)
    if remove_nums:
        text = remove_numbers(text)
    text = normalize_whitespace(text)

    return text


# ── DataFrame Cleaner ─────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame,
                    text_col: str = "text",
                    label_col: str = "sentiment") -> pd.DataFrame:
    """
    Apply clean_text() to entire DataFrame.
    - Adds a 'cleaned_text' column
    - Removes rows where cleaned text is empty
    - Preserves all original columns
    """
    print(f"🧹 Cleaning {len(df)} posts...")

    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(clean_text)

    # Drop empty
    before = len(df)
    df = df[df["cleaned_text"].str.strip().str.len() > 5].reset_index(drop=True)
    after = len(df)
    print(f"   Dropped {before - after} empty/very short posts after cleaning.")
    print(f"   Remaining: {after} posts.")

    return df


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_path = "data/raw/social_media_posts.csv"
    out_path = "data/processed/cleaned_posts.csv"

    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(raw_path):
        print("⚠ Raw dataset not found. Run src/generate_dataset.py first.")
    else:
        df = pd.read_csv(raw_path)
        cleaned = clean_dataframe(df)
        cleaned.to_csv(out_path, index=False)

        print(f"\n✅ Cleaned dataset saved → {out_path}")
        print("\nBefore / After sample:")
        for i in range(3):
            print(f"\n  Raw:     {df['text'].iloc[i][:100]}")
            print(f"  Cleaned: {cleaned['cleaned_text'].iloc[i][:100]}")
