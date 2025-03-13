import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not done yet
# nltk.download('stopwords')

def clean_csv(input_file, output_file):
    df = pd.read_csv(input_file, encoding='latin-1', header=None)
    df.drop(df.columns[[0, 2, 3]], axis=1, inplace=True)
    df.columns = ['ids', 'user', 'text']

    # Drop empty sets, remove duplicate tweets, clean for processing
    df = df.dropna()
    df = df.drop_duplicates(subset=['text'])
    df['clean_text'] = df['text'].apply(clean_text)
    df.to_csv(output_file, index=False, encoding='utf-8', header=False)

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters & numbers
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text



input_csv = 'SentimentTwitterDataset.csv'
output_csv = 'cleaned_tweets.csv'
clean_csv(input_csv, output_csv)