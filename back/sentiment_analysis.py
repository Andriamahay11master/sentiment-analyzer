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

