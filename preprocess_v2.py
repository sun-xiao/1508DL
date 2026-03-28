"""
Preprocessing pipeline v2 — optimised for deep-learning models (LSTM / CNN).

Key differences from v1:
  - Keeps stopwords (words like "not", "no", "but" carry sentiment signal)
  - No lemmatization (sequence models benefit from natural word forms)
  - Handles contractions so negations survive cleaning
"""

import os
import re

import emoji
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()

RAW_PATH = "./data/raw/training.1600000.processed.noemoticon.csv"
OUT_DIR  = "./data/processed_v2"

CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}
_CONTRACTION_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in CONTRACTIONS) + ")",
    flags=re.IGNORECASE,
)


def load_sentiment140(path):
    df = pd.read_csv(path, encoding="latin-1", header=None)
    df.columns = ["target", "id", "date", "flag", "user", "text"]
    df = df[["target", "text"]].copy()
    df["target"] = df["target"].map({0: 0, 4: 1})
    return df


def basic_clean(text):
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return text


def expand_contractions(text):
    return _CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)


def convert_emojis(text):
    text = emoji.demojize(text)
    text = text.replace(":", " ")
    return text


def remove_special_chars_and_numbers(text):
    text = re.sub(r"[^a-z\s!?']", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_v2(text):
    text = basic_clean(text)
    text = expand_contractions(text)
    text = convert_emojis(text)
    text = remove_special_chars_and_numbers(text)
    return text


def split_and_save(df: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=2026, stratify=df["target"],
    )

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print("Saved:")
    print(f" - {OUT_DIR}/train.csv  ({len(train_df)} rows)")
    print(f" - {OUT_DIR}/test.csv   ({len(test_df)} rows)")


if __name__ == "__main__":
    print("Loading raw data...")
    df = load_sentiment140(RAW_PATH)
    print(df["target"].value_counts())

    print("Preprocessing text (v2 — keep stopwords, no lemmatization)...")
    df["text"] = df["text"].progress_apply(preprocess_v2)

    num_empty = (df["text"].str.len() == 0).sum()
    print(f"Number of empty texts: {num_empty}")
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    print("Splitting and saving...")
    split_and_save(df)

    print(df.head(10))
