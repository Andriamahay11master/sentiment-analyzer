# Sentiment Analyzer

A small Flask web application and Python toolkit for performing sentiment analysis on short text (tweets, reviews, etc.). The project uses TF‑IDF vectorization and a trained Linear SVM to produce a sentiment label and a confidence score.

**Quick summary:** run the web app with `python app.py` and use `predict.py` to integrate predictions in scripts.

## Features

- Sentiment classification (positive / negative)
- Confidence scores from the SVM decision function (sigmoid conversion)
- Simple text preprocessing (URLs, mentions, hashtags, non-letter characters)
- Web interface with explainability (word-level contribution)
- Saved model and vectorizer in the `model/` directory for fast inference

## Repository layout

```
sentiment-analyzer/
├── app.py                  # Flask web UI
├── predict.py              # `predict_sentiment(text)` helper for scripts
├── back/                   # Training and explainability utilities
│   ├── training.py
│   ├── vectorization.py
│   └── explainability.py
├── model/                  # Pretrained artifacts (joblib)
│   ├── sentiment_model.joblib
│   └── vectorizer.joblib
├── templates/              # Flask HTML templates (index.html)
├── assets/                 # Static assets (css)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` already lists the runtime packages used by the app (Flask, scikit-learn, joblib, numpy, pandas, etc.).

## Run the web app (local)

1. Ensure the `model/` directory contains `sentiment_model.joblib` and `vectorizer.joblib`.
2. Start the Flask server:

```bash
python app.py
```

3. Open your browser at `http://127.0.0.1:5000/` and enter text to get a sentiment prediction and a simple explainability view.

## Use programmatically via `predict.py`

Example:

```python
from predict import predict_sentiment

print(predict_sentiment("I absolutely love this product!"))
# {'sentiment': 'positive', 'confidence': 0.98}
```

`predict.py` exposes `predict_sentiment(text)` which loads the model and vectorizer from `model/` and returns a dictionary with `sentiment` and `confidence`.

## Retrain the model

If you want to retrain the model with your own labeled data, see `back/training.py`:

```bash
python back/training.py
```

This script will preprocess text, train a Linear SVM on the training data, evaluate on a holdout set, and save the trained model and vectorizer to `model/`.

## Explainability

The web UI (and `app.py`) include a word-level contribution view computed from the model coefficients and the TF‑IDF vector. See `back/explainability.py` for additional utilities.

## Notes & Troubleshooting

- If model loading fails, confirm the files `model/sentiment_model.joblib` and `model/vectorizer.joblib` exist and were created with compatible scikit-learn versions.
- To run without the web UI, use `predict.py` directly for batch or scriptable predictions.

## Contributing

Contributions welcome. Open an issue or submit a pull request for features, bug fixes, or documentation improvements.

## License

This repository is provided for educational/demo purposes. Data and models may be derived from public Twitter sentiment datasets.
