# Sentiment Analyzer

A machine learning-powered sentiment analysis tool that automatically classifies the emotional tone (positive, negative, or neutral) of text data using Natural Language Processing (NLP) and Support Vector Machines (SVM).

## Overview

This project implements an end-to-end sentiment analysis pipeline that:

- **Cleans and preprocesses** Twitter/social media text data
- **Vectorizes** text using TF-IDF (Term Frequency-Inverse Document Frequency)
- **Trains** a Linear SVM classifier on labeled sentiment data
- **Predicts** sentiment with confidence scores for new text inputs

## Features

- 🎯 **Sentiment Classification**: Classifies text into positive or negative sentiment
- 📊 **Confidence Scoring**: Returns probability scores for predictions using sigmoid conversion
- 🧹 **Text Preprocessing**: Removes URLs, mentions, hashtags, and special characters
- 💾 **Persistent Models**: Pre-trained model and vectorizer saved in `model/` directory
- ⚡ **Easy Integration**: Simple Python functions for making predictions on new text

## Project Structure

```
sentiment-analyzer/
├── app.py                    # Main application script
├── predict.py               # Prediction module with sentiment inference
├── back/
│   ├── training.py          # Model training script
│   └── vectorization.py     # TF-IDF vectorization utilities
├── model/
│   ├── sentiment_model.joblib    # Trained Linear SVM model
│   └── vectorizer.joblib         # Fitted TF-IDF vectorizer
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Setup

1. **Clone or download the repository**

   ```bash
   git clone <repository-url>
   cd sentiment-analyzer
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - pandas
   - scikit-learn
   - joblib
   - numpy

## Usage

### Making Predictions

Use the `predict_sentiment()` function from `predict.py`:

```python
from predict import predict_sentiment

# Analyze sentiment
result = predict_sentiment("I love this product! It's amazing!")
print(result)
# Output: {"sentiment": "Positive", "confidence": 0.95}
```

### Training a New Model

To retrain the model with your own data:

```bash
python back/training.py
```

The script will:

1. Load Twitter sentiment data from the source
2. Clean and preprocess text
3. Train a Linear SVM classifier
4. Evaluate on test data
5. Save the model and vectorizer to `model/`

## Model Details

- **Algorithm**: Linear SVM (Support Vector Machine)
- **Vectorization**: TF-IDF with sklearn
- **Data Source**: Twitter Sentiment Analysis dataset (50,000 samples)
- **Labels**: Binary classification (Positive: 1, Negative: 0)
- **Preprocessing**: Lowercasing, URL removal, mention removal, special character filtering

## Output Format

The `predict_sentiment()` function returns a dictionary:

```python
{
    "sentiment": "Positive",     # or "Negative"
    "confidence": 0.85           # Confidence score (0.0 to 1.0)
}
```

## Performance Metrics

Model evaluation metrics are generated during training and include:

- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)

## License

This project uses publicly available Twitter sentiment data for demonstration purposes.

## Contributing

Feel free to fork, improve, and submit pull requests!
