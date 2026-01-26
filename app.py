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
# Routes
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    text = ""

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

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        text=text
    )

if __name__ == "__main__":
    app.run(debug=True)
