import time
from preprocess import load_and_clean_dataset

def stream_comments(path, delay=1.0):
    """
    Simulates real-time comment streaming.
    Yields one cleaned comment at a time.
    """
    df = load_and_clean_dataset(path)

    for _, row in df.iterrows():
        yield {
            "id": row["id"],
            "topic": row["topic"],
            "sentiment": row["sentiment"],
            "text": row["text"],
            "clean_text": row["clean_text"]
        }
        time.sleep(delay)  # simulate real-time streaming delay


if __name__ == "__main__":
    # Test streaming
    for comment in stream_comments("../data/twitter_training.csv", delay=0.5):
        print(comment)
