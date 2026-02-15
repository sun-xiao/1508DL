import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -------------------------
# Step 1: Load data
# -------------------------

TRAIN_PATH = "./data/processed/train.csv"
TEST_PATH  = "./data/processed/test.csv"

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = train_df["text"]
y_train = train_df["target"]

X_test = test_df["text"]
y_test = test_df["target"]

# -------------------------
# Step 2: TF-IDF feature extraction
# -------------------------

print("Vectorizing text using TF-IDF...")

vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=5, max_df=0.9 )

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)

# -------------------------
# Step 3: Train Logistic Regression
# -------------------------

print("Training Logistic Regression...")

model = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)

model.fit(X_train_tfidf, y_train)

# -------------------------
# Step 4: Evaluation
# -------------------------

print("Evaluating...")

y_pred = model.predict(X_test_tfidf)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nDetailed report:")
print(classification_report(y_test, y_pred))
