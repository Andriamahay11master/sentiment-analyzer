from flask import Flask, render_template, request
import re
import joblib
import numpy as np

app = Flask(
    __name__,
    static_folder="assets",
    static_url_path="/static"
)

# -------------------------------
# Load model & vectorizer
# -------------------------------

model = joblib.load("model/sentiment_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

# -------------------------------
# Text cleaning
# -------------------------------

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------------
# Explainability
# -------------------------------
def explain_prediction(text, top_n=5):
    clean_text = clean_tweet(text)
    vector = vectorizer.transform([clean_text])

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    feature_indices = vector.nonzero()[1]

    contributions = []
    for idx in feature_indices:
        word = feature_names[idx]
        score = coefficients[idx] * vector[0, idx]
        contributions.append((word, score))

    contributions = sorted(
        contributions,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return contributions[:top_n]

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    text = ""
    explanation = []

    if request.method == "POST":
        text = request.form["text"]
        cleaned = clean_tweet(text)
        vector = vectorizer.transform([cleaned])

        score = model.decision_function(vector)
        prob_positive = sigmoid(score)
        prob_negative = 1 - prob_positive

        label = model.predict(vector)[0]

        if label == 1:
            prediction = "Positive"
            confidence = float(prob_positive)
        else:
            prediction = "Negative"
            confidence = float(prob_negative)

        explanation = explain_prediction(text)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        text=text,
        explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)
