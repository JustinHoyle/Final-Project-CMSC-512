# Step 1: Data Cleaning (clean_csv.py)
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not done yet
# nltk.download('stopwords')

def clean_csv(input_file, output_file):
    """
    Reads a CSV file, removes unnecessary columns, cleans text data, and saves the processed CSV.
    """
    df = pd.read_csv(input_file, encoding='latin-1', header=None)
    df.drop(df.columns[[0, 2, 3]], axis=1, inplace=True)  # Remove unwanted columns
    df.columns = ['ids', 'user', 'text']  # Rename remaining columns

    # Drop empty entries, remove duplicate tweets, and clean text for further processing
    df = df.dropna()
    df = df.drop_duplicates(subset=['text'])
    df['clean_text'] = df['text'].apply(clean_text)  # Apply text cleaning function
    df.to_csv(output_file, index=False, encoding='utf-8', header=False)

def clean_text(text):
    """
    Cleans the text by removing URLs, mentions, hashtags, punctuation, and stopwords.
    """
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Cleanup function to remove mentions from PII-detected data
def remove_mentions(input_name, output_name):
    """
    Filters out rows where PII information includes mentions (PERSON tags with '@').
    """
    df = pd.read_csv(input_name)
    def has_person_at_symbol(pii_text):
        if pd.isna(pii_text):
            return False
        return any(tag.startswith("PERSON: @") for tag in pii_text.split(", "))
    df_filtered = df[~df['pii'].apply(has_person_at_symbol)]
    df_filtered.to_csv(output_name, index=False)

input_csv = 'pii_detected_tweets_unclean.csv'
output_csv = 'pii_detected_tweets_unclean_no_mentions.csv'
remove_mentions(input_csv, output_csv)