import joblib
import pandas as pd
import numpy as np
from mastodon import Mastodon
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model and embedder
model = joblib.load("random_forest_model_embed.pkl")
embedder = joblib.load("sentence_transformer_embedder.pkl")

# Load Jean-Baptiste NER pipeline
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ner_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, grouped_entities=True)

# Mastodon API setup
ACCESS_TOKEN = '5rBeVLWvRKhpDW4PXrcjNE0ze3vjnUKC6qGQ8p0x9Z4'
API_BASE_URL = 'https://mastodon.social'

mastodon = Mastodon(
    access_token=ACCESS_TOKEN,
    api_base_url=API_BASE_URL
)

def get_account_info(username):
    account = mastodon.account_search(username, limit=1)
    return account[0] if account else None

def get_recent_statuses(account_id, limit=10):
    return mastodon.account_statuses(account_id, limit=limit)

def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def extract_structured_features(text):
    if not isinstance(text, str):
        return {}

    entities = ner_pipeline(text)
    labels = [ent["entity_group"] for ent in entities]

    return {
        "has_per": int("PER" in labels),
        "has_org": int("ORG" in labels),
        "has_loc": int("LOC" in labels),
        "has_misc": int("MISC" in labels),
        "text_length": len(text),
        "word_count": len(text.split())
    }

if __name__ == "__main__":
    username = "PII_TEST_MODEL@mastodon.social"
    account = get_account_info(username)

    if account:
        statuses = get_recent_statuses(account['id'], limit=10)
        texts = [clean_html(status['content']) for status in statuses]

        # Embeddings
        X_text = embedder.encode(texts, show_progress_bar=False)

        # Structured features
        structured = [extract_structured_features(text) for text in texts]
        structured_df = pd.DataFrame(structured)
        X_structured = structured_df[['text_length', 'word_count', 'has_per', 'has_org', 'has_loc', 'has_misc']].values

        # Combine
        X_combined = np.concatenate([X_text, X_structured], axis=1)

        # Predict
        preds = model.predict(X_combined)
        probs = model.predict_proba(X_combined)

        for i, text in enumerate(texts):
            print(f"\nToot #{i+1}: {text}")
            print(f"PII Risk Prediction: {int(preds[i])}")
            print(f"Confidence: {max(probs[i]) * 100:.2f}%")
    else:
        print("User not found.")
