class Sentiment:
    def __init__(self, sentiment_score: float, sentiment_label: str):
        self.sentiment_score = sentiment_score
        self.sentiment_label = sentiment_label

    def __repr__(self):
        return f"Sentiment(score={self.sentiment_score}, label='{self.sentiment_label}')"