import pandas as pd

train_path = "../data/twitter_training.csv"
valid_path = "../data/twitter_validation.csv"

print("=== TRAINING DATA ===")
train_df = pd.read_csv(train_path, header=None)
print("Shape:", train_df.shape)

print("\nColumn names:", train_df.columns.tolist())

print("\nSample rows:")
print(train_df.head())

print("\nMissing values:")
print(train_df.isnull().sum())

print("\nSentiment distribution:")
print(train_df[2].value_counts())  # column 2 contains sentiment labels


print("\n=== VALIDATION DATA ===")
valid_df = pd.read_csv(valid_path, header=None)
print("Shape:", valid_df.shape)

print("\nSentiment distribution:")
print(valid_df[2].value_counts())
