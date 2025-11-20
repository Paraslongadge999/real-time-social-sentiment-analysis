import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Convert to lowercase
    text = text.lower()
    
    return text


def load_and_clean_dataset(path):
    # Load without header because our dataset has no header
    df = pd.read_csv(path, header=None)

    # Drop rows with missing text
    df = df.dropna(subset=[3])
    
    # Rename columns for clarity
    df = df.rename(columns={
        0: "id",
        1: "topic",
        2: "sentiment",
        3: "text"
    })
    
    # Clean text column
    df["clean_text"] = df["text"].apply(clean_text)
    
    return df
