import pandas as pd

# Twitter sentiment dataset
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"

# Load data
data = pd.read_csv(url, encoding="latin-1")

# Keep only relevant columns
data = data[["tweet", "label"]]

# Inspect data
print("Columns:", data.columns.tolist())
print("Dataset shape:", data.shape)

print("\nFirst rows:")
print(data.head())

print("\nLabel distribution:")
print(data["label"].value_counts())
