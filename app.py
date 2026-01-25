import pandas as pd

# Public sentiment dataset (movie reviews)
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"

# Load data
data = pd.read_csv(url)

# Keep only relevant columns
data = data[["text", "sentiment"]]

# Inspect data
print("Dataset shape:", data.shape)
print("\nFirst rows:")
print(data.head())

print("\nLabel distribution:")
print(data["sentiment"].value_counts())
