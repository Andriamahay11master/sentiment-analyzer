def explain_prediction(text, top_n=5):
    clean_text = clean_tweet(text)
    vector = vectorizer.transform([clean_text])

    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    # Get non-zero features in this text
    feature_indices = vector.nonzero()[1]

    contributions = []

    for idx in feature_indices:
        word = feature_names[idx]
        weight = coefficients[idx]
        value = vector[0, idx]
        contributions.append((word, weight * value))

    # Sort by absolute contribution
    contributions = sorted(
        contributions,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return contributions[:top_n]
