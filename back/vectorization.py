import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Load data (same as Step 1)
# -------------------------------

url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
data = pd.read_csv(url, encoding="latin-1")

data = data[["tweet", "label"]]

# Optional: sample for faster experimentation
max_samples = 50000
if len(data) > max_samples:
    data = data.sample(max_samples, random_state=42)


# -------------------------------
# Text cleaning function
# -------------------------------

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)                # remove mentions
    text = re.sub(r"#", "", text)                   # remove hashtag symbol
    text = re.sub(r"[^a-z\s]", "", text)            # keep letters only
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text

# Apply cleaning
data["clean_text"] = data["tweet"].apply(clean_tweet)

print("Example before cleaning:")
print(data["tweet"].iloc[0])

print("\nExample after cleaning:")
print(data["clean_text"].iloc[0])

# -------------------------------
# TF-IDF Vectorization
# -------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

print("\nTF-IDF shape:", X.shape)
print("Number of features:", len(vectorizer.get_feature_names_out()))
