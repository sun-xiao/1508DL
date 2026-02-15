import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import re
from sklearn.model_selection import train_test_split
import emoji

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

RAW_PATH = "./data/raw/training.1600000.processed.noemoticon.csv"
OUT_DIR  = "./data/processed"

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# -------------------------
# Step 0: Load raw data
# -------------------------

def load_sentiment140(path):
    df = pd.read_csv(path, encoding="latin-1", header=None)
    df.columns = ["target", "id", "date", "flag", "user", "text"]
    df = df[["target", "text"]].copy()

    # 0 -> 0 (neg), 4 -> 1 (pos)
    df["target"] = df["target"].map({0: 0, 4: 1})
    return df


# -------------------------
# Step 1: Basic cleaning
# Remove HTML, URL, @mention, #hashtag marker
# -------------------------

def basic_clean(text):
    # remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # convert hashtags: "#happy" -> "happy"
    text = re.sub(r"#(\w+)", r"\1", text)

    return text

# -------------------------
# Step 2: Handle emojis
# -------------------------
def convert_emojis(text):
    text = emoji.demojize(text)
    text = text.replace(":", " ")  # ":smile:" -> " smile "
    return text

# -------------------------
# Step 3: Remove special chars & numbers
# Keep letters, spaces, and ! ?
# -------------------------
def remove_special_chars_and_numbers(text):
    # keep a-z, space, !, ?
    text = re.sub(r"[^a-z\s!?']", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------
# Step 4: Tokenize
# -------------------------
def tokenize(text):
    return text.split()


# -------------------------
# Step 5: Stop-word removal
# -------------------------
def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS]

# -------------------------
# Step 6: Lemmatization
# -------------------------
def lemmatize(tokens):
    return [LEMMATIZER.lemmatize(t) for t in tokens]

def preprocess(text):
    text = basic_clean(text)
    text = convert_emojis(text)
    text = remove_special_chars_and_numbers(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)


def split_and_save(df: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=2026, stratify=df["target"])

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print("Saved:")
    print(" - data/processed/train.csv")
    print(" - data/processed/test.csv")

if __name__ == "__main__":
    # Update nltk datasete
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    ## ************************************************************************* ##

    #Step0: load and clear raw data
    print("Loading raw data...")
    df = load_sentiment140(RAW_PATH)
    # Note that this is a balanced dataset set with 800k label 0 and 800k label 1
    print(df["target"].value_counts())

    ## ************************************************************************* ##
    print("Preprocessing text...")
    df["text"] = df["text"].progress_apply(preprocess)

    num_empty = (df["text"].str.len() == 0).sum()
    print("Number of empty texts: {}".format(num_empty))
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    print("Splitting and saving...")
    split_and_save(df)

    print(df.head(10))


