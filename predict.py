import re
import joblib
import numpy as np

# -------------------------------
# Load model & vectorizer
# -------------------------------

model = joblib.load("model/sentiment_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

# -------------------------------
# Text cleaning (same as training)
# -------------------------------

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# Confidence helper (sigmoid)
# -------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------------
# Prediction function
# -------------------------------

def predict_sentiment(text):
    clean_text = clean_tweet(text)
    vector = vectorizer.transform([clean_text])

    # SVM decision score (scalar)
    score = model.decision_function(vector)

    # Convert to confidence
    prob_positive = sigmoid(score)
    prob_negative = 1 - prob_positive

    prediction = model.predict(vector)[0]

    if prediction == 1:
        return {
            "sentiment": "positive",
            "confidence": float(prob_positive)
        }
    else:
        return {
            "sentiment": "negative",
            "confidence": float(prob_negative)
        }

# -------------------------------
# Test predictions
# -------------------------------

if __name__ == "__main__":
    examples = [
        "I absolutely love this product!",
        "This is the worst experience ever",
        "Not bad, but could be better",
        "I hate how slow this is"
    ]

    for text in examples:
        result = predict_sentiment(text)
        print(f"Text: {text}")
        print(f"Prediction: {result['sentiment']} ({result['confidence']:.2f})")
        print("-" * 40)
