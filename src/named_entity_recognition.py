import spacy
import pandas as pd
import re

# Make sure to download the model first
nlp = spacy.load("en_core_web_trf")

PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
}


def extract_pii(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""

    doc = nlp(text)
    
    detected_pii = set()

    # Extract PII using spacy's NER
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            detected_pii.add(f"{ent.label_}: {ent.text}")

    # Extract PII using regex
    for label, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        for match in matches:
            detected_pii.add(f"{label}: {match}")

    return ", ".join(detected_pii) if detected_pii else "No PII Detected"


df = pd.read_csv("SentimentTwitterDataset.csv", encoding='latin-1', header=None)
df.drop(df.columns[[0, 2, 3]], axis=1, inplace=True)
df.columns = ['ids', 'user', 'text']
pii_results = []
total_rows = len(df)
print(f"Processing {total_rows} rows...")
for iter, text in enumerate(df['text']):
    pii_results.append(extract_pii(text))

    # Print progress every 1000 rows
    if (iter + 1) % 1000 == 0:
        print(f"Processed {iter + 1}/{total_rows} rows...")

df['pii'] = pii_results
df.to_csv("pii_detected_tweets_unclean.csv", index=False)
