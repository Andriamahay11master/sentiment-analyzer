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

LABEL_MAP = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# -------------------------------
# Text cleaning
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# -------------------------------
# Explainability (per predicted class)
# -------------------------------

def explain_prediction(text, predicted_class, top_n=5):
    clean = clean_text(text)
    vector = vectorizer.transform([clean])

    feature_names = vectorizer.get_feature_names_out()
    class_coef = model.coef_[predicted_class]

    feature_indices = vector.nonzero()[1]

    contributions = []
    for idx in feature_indices:
        word = feature_names[idx]
        score = class_coef[idx] * vector[0, idx]
        contributions.append((word, score))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
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
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])

        # Prediction
        label = model.predict(vector)[0]
        prediction = LABEL_MAP[label]

        # Confidence (softmax over decision scores)
        scores = model.decision_function(vector)[0]
        probs = softmax(scores)
        confidence = float(probs[label])

        # Explainability
        explanation = explain_prediction(text, label)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        text=text,
        explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)
