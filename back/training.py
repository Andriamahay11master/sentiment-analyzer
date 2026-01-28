import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load data
# -------------------------------

url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
data = pd.read_csv(url, encoding="latin-1")

# Keep only relevant columns
data = data[["tweet", "label"]]

# -------------------------------
# Normalize labels
# 0 -> negative
# 4 -> positive
# -------------------------------

data["label"] = data["label"].map({
    0: 0,
    4: 1
})

# 🔴 IMPORTANT: remove invalid labels created by mapping
data = data.dropna(subset=["label"])

# Ensure labels are integers (required by sklearn)
data["label"] = data["label"].astype(int)

# -------------------------------
# Optional: sample for faster runs
# -------------------------------

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
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# Model 1: Logistic Regression
# -------------------------------

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_vec, y_train)
logreg_preds = logreg.predict(X_test_vec)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, logreg_preds))
print(classification_report(y_test, logreg_preds))

# -------------------------------
# Model 2: Linear SVM
# -------------------------------

svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_preds = svm.predict(X_test_vec)

print("\n=== Linear SVM ===")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))
