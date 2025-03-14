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

    print(", ".join(detected_pii) if detected_pii else "No PII Detected")
    return ", ".join(detected_pii) if detected_pii else "No PII Detected"


df = pd.read_csv("cleaned_tweets.csv", names=['ids', 'user', 'text', 'clean_text'])
df['pii'] = df['clean_text'].apply(extract_pii)
df.to_csv("pii_detected_tweets.csv", index=False)
