"""
generate_dataset.py
--------------------
Generates a realistic synthetic social media dataset with 5000 posts
simulating Twitter/Instagram/Facebook-style comments across multiple brands.

Run: python src/generate_dataset.py
Output: data/raw/social_media_posts.csv
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Brand & Platform Configuration ───────────────────────────────────────────
BRANDS    = ["Amazon", "Flipkart", "Zomato", "Netflix", "Swiggy",
             "Samsung", "Apple", "Ola", "Uber", "BookMyShow"]

PLATFORMS = ["Twitter", "Instagram", "Facebook", "Reddit", "Google Reviews"]

TOPICS    = ["delivery", "customer service", "app experience", "pricing",
             "product quality", "refund", "new feature", "offer/sale",
             "packaging", "subscription"]

# ── Sentiment Templates ───────────────────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "Absolutely love {brand}! {topic} was smooth and flawless 😍",
    "Just got my order from {brand} — {topic} exceeded expectations! 🎉",
    "{brand} has the best {topic} I've ever experienced! 10/10 would recommend.",
    "Wow, {brand}'s {topic} is top notch. Totally worth every rupee! ✅",
    "Shoutout to {brand} for amazing {topic}. Super fast and reliable! 🙌",
    "I'm genuinely impressed with {brand}. Their {topic} is flawless 💯",
    "Thank you {brand} for the seamless {topic} experience. Will definitely use again!",
    "{brand} never disappoints! {topic} is always on point 🔥",
    "Finally a brand that cares! {brand}'s {topic} is the gold standard.",
    "My {topic} with {brand} was incredible. Super happy customer here! 😊",
    "Best {topic} in the market hands down — {brand} you're amazing 🌟",
    "Ordered yesterday, received today. {brand}'s {topic} speed is unreal!",
    "5 stars for {brand}! Their {topic} team went above and beyond 🤩",
    "{brand} just won me over completely. {topic} was handled perfectly 👏",
    "No complaints at all! {brand}'s {topic} is genuinely world-class.",
    "Happy to be a loyal {brand} customer. {topic} keeps getting better!",
    "Fast, reliable, and affordable. {brand}'s {topic} is simply the best.",
    "Just renewed my {brand} subscription — {topic} alone is worth it!",
    "The {topic} update from {brand} is 🔥 — exactly what users needed!",
    "{brand} handled my {topic} issue instantly. Best support ever! 💪",
]

NEGATIVE_TEMPLATES = [
    "Terrible experience with {brand}. {topic} was a complete disaster! 😤",
    "Never ordering from {brand} again. {topic} was the worst I've seen 🤬",
    "{brand}'s {topic} is broken. Waited 3 days and still nothing resolved.",
    "Pathetic {topic} from {brand}. Feels like they don't care about customers.",
    "Avoid {brand} at all costs. {topic} ruined my entire experience 😠",
    "I'm done with {brand}. Their {topic} is an absolute joke.",
    "Charged me twice and {brand}'s {topic} team is unreachable. Disgusting!",
    "The worst {topic} I've ever dealt with — {brand} needs to do better.",
    "0 stars for {brand}. {topic} caused me so much stress 😭",
    "Unacceptable! {brand}'s {topic} team gave me zero solutions. #Disappointed",
    "{brand} please fix your {topic}. This is getting ridiculous 😡",
    "My {topic} complaint to {brand} has been ignored for a week now. Horrible.",
    "Extremely disappointed with {brand}'s {topic}. Expected way better.",
    "Warning: {brand}'s {topic} is completely broken right now. Save yourself!",
    "{brand} took my money and ghosted me on {topic}. Furious! 🤯",
    "Still waiting for {brand} to resolve my {topic} from last month. Useless.",
    "Overpriced and underdelivered. {brand}'s {topic} is a scam.",
    "Spent 2 hours on hold with {brand} about {topic}. Total waste of time.",
    "Cancelled my {brand} subscription because {topic} is just too unreliable.",
    "{brand} you have lost a loyal customer. {topic} was the last straw. 💔",
]

NEUTRAL_TEMPLATES = [
    "Just tried {brand}'s {topic} for the first time. Let's see how it goes.",
    "Comparing {brand} with competitors on {topic}. Interesting differences.",
    "Anyone else notice {brand} changed their {topic} policy recently?",
    "{brand} announced updates to their {topic}. Details still unclear.",
    "Using {brand} for {topic} today. Will update with thoughts later.",
    "Heard mixed reviews about {brand}'s {topic}. Trying it out myself.",
    "Can someone explain {brand}'s {topic} terms? A bit confusing.",
    "{brand} seems to be testing a new {topic} feature. Interesting.",
    "Not sure how I feel about {brand}'s latest {topic} changes yet.",
    "Researching options for {topic} — {brand} is one of my choices.",
    "{brand} updated their {topic} page today. Just noticed.",
    "Reading reviews about {brand}'s {topic} before making a decision.",
    "Anyone have experience with {brand}'s {topic}? Need honest feedback.",
    "The {topic} numbers from {brand} this quarter are quite average.",
    "Neutral on {brand}'s {topic} update. Neither good nor bad for now.",
    "{brand} released a statement on {topic}. I'll read it carefully.",
    "Checked {brand}'s {topic} status — shows operational with no issues.",
    "Does {brand} offer {topic} for international users too?",
    "Just created an account on {brand} to explore their {topic} options.",
    "So {brand} changed {topic} again. Still evaluating what it means.",
]

# ── Noise / Realistic Variations ──────────────────────────────────────────────
HASHTAG_POOL = [
    "#CustomerExperience", "#BrandReview", "#TechTalk", "#UserFeedback",
    "#OnlineShopping", "#AppReview", "#StartupIndia", "#DigitalIndia",
    "#ConsumerRights", "#ProductReview", "#ServiceAlert", "#MustTry",
    "#Disappointed", "#Impressed", "#FoodDelivery", "#TechNews",
]

USER_HANDLES = [f"@user_{i:04d}" for i in range(1, 500)]

def add_noise(text):
    """Randomly add hashtags, mentions, typos, emojis to make text realistic."""
    if random.random() < 0.35:
        text += " " + random.choice(HASHTAG_POOL)
    if random.random() < 0.2:
        text += " " + random.choice(USER_HANDLES)
    if random.random() < 0.15:
        text = text.replace("the", "teh").replace("is", "iz")
    return text

def generate_post(sentiment_label):
    brand    = random.choice(BRANDS)
    topic    = random.choice(TOPICS)
    platform = random.choice(PLATFORMS)

    if sentiment_label == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
    elif sentiment_label == "negative":
        template = random.choice(NEGATIVE_TEMPLATES)
    else:
        template = random.choice(NEUTRAL_TEMPLATES)

    text = template.format(brand=brand, topic=topic)
    text = add_noise(text)

    # Realistic timestamp: last 2 years
    days_back     = random.randint(1, 730)
    post_datetime = datetime.now() - timedelta(days=days_back,
                                               hours=random.randint(0, 23),
                                               minutes=random.randint(0, 59))

    return {
        "post_id":   f"POST_{random.randint(100000, 999999)}",
        "timestamp": post_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "platform":  platform,
        "brand":     brand,
        "topic":     topic,
        "text":      text,
        "likes":     random.randint(0, 5000),
        "retweets":  random.randint(0, 2000),
        "replies":   random.randint(0, 500),
        "sentiment": sentiment_label,
    }

def generate_dataset(n=5000):
    # Class distribution: 45% positive, 35% negative, 20% neutral
    labels = (
        ["positive"] * int(n * 0.45) +
        ["negative"] * int(n * 0.35) +
        ["neutral"]  * int(n - int(n * 0.45) - int(n * 0.35))
    )
    random.shuffle(labels)

    records = [generate_post(label) for label in labels]
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/social_media_posts.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Dataset generated: {len(df)} posts → {out_path}")
    print(f"\nSentiment Distribution:\n{df['sentiment'].value_counts()}")
    print(f"\nPlatform Distribution:\n{df['platform'].value_counts()}")
    print(f"\nBrand Distribution:\n{df['brand'].value_counts()}")
    print(f"\nSample Posts:\n{df[['text','sentiment']].head(3).to_string()}")
    return df

if __name__ == "__main__":
    generate_dataset(5000)
