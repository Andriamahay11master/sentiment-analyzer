# Sentiment Analyzer

A **machine learning–powered sentiment analysis web application** that classifies the emotional tone of text as **positive or negative**, using **Natural Language Processing (NLP)** and a **Linear Support Vector Machine (SVM)**. The application also provides **confidence scores and explainability**, highlighting which words most influenced each prediction.

---

## 🚀 Overview

This project implements a **full end-to-end AI pipeline**, from data preprocessing and model training to deployment-ready inference and a web interface.

The system:

- Cleans and preprocesses noisy social media text (tweets)
- Converts text into numerical features using **TF-IDF**
- Trains and evaluates multiple models (Logistic Regression & Linear SVM)
- Selects the best-performing model (Linear SVM)
- Saves and reloads the trained model for production use
- Provides **interpretable predictions** by showing influential words
- Exposes predictions through a **Flask web application** styled with **Sass (SCSS)**

---

## ✨ Features

- 🎯 **Sentiment Classification** — Positive or Negative sentiment
- 📊 **Confidence Scoring** — Confidence derived from SVM decision scores (sigmoid)
- 🔍 **Explainability** — Displays top words contributing to each prediction
- 🧹 **Text Preprocessing** — URL, mention, hashtag, and noise removal
- 💾 **Persistent Models** — Model and vectorizer saved with `joblib`
- 🌐 **Web Application** — Flask-based UI with Sass-powered styling
- ⚡ **Production-Oriented Design** — Train once, predict many times

---

## 📁 Project Structure

```
sentiment-analyzer/
├── app.py                     # Flask web application
├── predict.py                 # Prediction & explainability logic
├── back/
│   ├── training.py            # Model training & evaluation
│   └── vectorization.py       # Text cleaning & TF-IDF pipeline
├── model/
│   ├── sentiment_model.joblib # Trained Linear SVM model
│   └── vectorizer.joblib      # Fitted TF-IDF vectorizer
├── assets/
│   ├── scss/                  # Sass (SCSS) source files
│   └── css/                   # Compiled CSS served by Flask
├── templates/
│   └── index.html             # Web UI template
└── README.md
```

---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- pip
- Node.js (only if you want to compile Sass locally)

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd sentiment-analyzer
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

Main dependencies:

- pandas
- scikit-learn
- numpy
- joblib
- flask

3. _(Optional)_ **Install Sass**

```bash
npm install -g sass
```

---

## ▶️ Usage

### Run the Web Application

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

Enter a sentence and receive:

- Sentiment prediction
- Confidence score
- Explanation of influential words

---

### Making Predictions in Code

```python
from predict import predict_sentiment

result = predict_sentiment("I love my job")
print(result)
# {'sentiment': 'Positive', 'confidence': 0.96}
```

---

## 🧠 Model Details

- **Algorithm**: Linear Support Vector Machine (LinearSVC)
- **Vectorization**: TF-IDF (unigrams + bigrams)
- **Dataset**: Twitter Sentiment Analysis dataset
- **Labels**:
  - 0 → Negative
  - 1 → Positive

- **Preprocessing**:
  - Lowercasing
  - URL & mention removal
  - Hashtag normalization
  - Non-alphabetic filtering

---

## 🔍 Explainability

For each prediction, the system identifies the **most influential words** by:

- Using SVM feature weights (`model.coef_`)
- Computing per-word contribution scores
- Displaying words that pushed the prediction toward positive or negative

This makes the model **transparent and interpretable**, rather than a black box.

---

## 📈 Evaluation

During training, models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

Linear SVM achieved the **best overall performance**, outperforming Logistic Regression on the same feature set.

---

## 🧪 Retraining the Model

To retrain from scratch:

```bash
python back/training.py
```

This will:

1. Load and preprocess the dataset
2. Train and evaluate models
3. Save the best model and vectorizer to `model/`

---

## 🧩 Future Improvements

- Neutral sentiment support
- Probability calibration
- Dark mode UI
- REST API endpoint
- Dockerized deployment
- Model monitoring and logging

---

## 📜 License

This project uses publicly available Twitter sentiment data for educational and demonstration purposes.

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the project, open issues, or submit pull requests.
