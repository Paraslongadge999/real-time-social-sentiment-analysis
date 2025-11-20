from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Returns sentiment scores using VADER.
    Scores include: neg, neu, pos, compound
    """
    if not isinstance(text, str):
        return {"neg": 0, "neu": 0, "pos": 0, "compound": 0}

    scores = vader.polarity_scores(text)
    return scores
