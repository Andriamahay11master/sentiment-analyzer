import re
import pandas as pd
import joblib
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# Load Kaggle dataset
# -------------------------------

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mdismielhossenabir/sentiment-analysis",
    "sentiment_analysis.csv"
)

# Rename columns
df = df.rename(columns={
    "text": "text",
    "sentiment": "label"
})

# Normalize labels
df["label"] = df["label"].str.lower()

# Keep only positive & negative
df = df[df["label"].isin(["positive", "negative", "neutral"])]

df["label"] = df["label"].map({
    "negative": 0,
    "positive": 1,
    "neutral": 2
})

print("Label distribution:")
print(df["label"].value_counts())

# -------------------------------
# Clean text
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]

# -------------------------------
# Train / test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# TF-IDF Vectorization
# -------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# Train Linear SVM (balanced)
# -------------------------------

model = LinearSVC(class_weight="balanced", C=0.5, random_state=42)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print("\n=== Linear SVM Results ===")
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# -------------------------------
# Save model & vectorizer
# -------------------------------

joblib.dump(model, "model/sentiment_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")

print("\nModel and vectorizer saved successfully.")
