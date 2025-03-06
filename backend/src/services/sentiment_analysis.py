def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of the given text using a pre-trained sentiment analysis model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing the sentiment score and label.
    """
    from transformers import pipeline

    # Load the sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Perform sentiment analysis
    result = sentiment_pipeline(text)[0]

    return {
        "label": result["label"],
        "score": result["score"]
    }