import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# -------------------------------
# Load data
# -------------------------------

url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
data = pd.read_csv(url, encoding="latin-1")
data = data[["tweet", "label"]]

# Optional: sample for faster runs
max_samples = 50000
if len(data) > max_samples:
    data = data.sample(max_samples, random_state=42)

# -------------------------------
# Clean text
# -------------------------------

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["clean_text"] = data["tweet"].apply(clean_tweet)

X = data["clean_text"]
y = data["label"]

# -------------------------------
# Train / test split
# -------------------------------

X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Vectorization
# -------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X_train_vec = vectorizer.fit_transform(X_train)

# -------------------------------
# Train best model (Linear SVM)
# -------------------------------

model = LinearSVC()
model.fit(X_train_vec, y_train)

# -------------------------------
# Save model & vectorizer
# -------------------------------

joblib.dump(model, "model/sentiment_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")

print("✅ Model and vectorizer saved successfully!")
