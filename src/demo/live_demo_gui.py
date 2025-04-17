import tkinter as tk
from tkinter import scrolledtext, messagebox
from mastodon import Mastodon
from bs4 import BeautifulSoup
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model and embedder
model = joblib.load("random_forest_model_embed.pkl")
embedder = joblib.load("sentence_transformer_embedder.pkl")

# Load NER model
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ner_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, grouped_entities=True)

# Mastodon API setup
ACCESS_TOKEN = '5rBeVLWvRKhpDW4PXrcjNE0ze3vjnUKC6qGQ8p0x9Z4'
API_BASE_URL = 'https://mastodon.social'

mastodon = Mastodon(access_token=ACCESS_TOKEN, api_base_url=API_BASE_URL)

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

def run_analysis(username, output_box):
    try:
        account = mastodon.account_search(username, limit=1)
        if not account:
            messagebox.showerror("Error", "User not found.")
            return

        account = account[0]
        statuses = mastodon.account_statuses(account['id'], limit=10)
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

        # Display results
        output_box.configure(state='normal')
        output_box.delete(1.0, tk.END)
        for i, text in enumerate(texts):
            output_box.insert(tk.END, f"\nToot #{i+1}:\n{text.strip()}\n")
            output_box.insert(tk.END, f"Prediction: Risk Level {int(preds[i])}\n")
            output_box.insert(tk.END, f"Confidence: {max(probs[i]) * 100:.2f}%\n\n")
        overall_score = np.round(np.mean(preds), 2)
        output_box.insert(tk.END, f"\nOverall Profile Risk Score: {overall_score}\n")
        output_box.configure(state='disabled')
    except Exception as e:
        messagebox.showerror("Error", str(e))

# --- GUI ---
root = tk.Tk()
root.title("Mastodon PII Risk Analyzer")

tk.Label(root, text="Enter Mastodon Username:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
username_entry = tk.Entry(root, width=40)
username_entry.grid(row=0, column=1, padx=10, pady=5)

output_box = scrolledtext.ScrolledText(root, width=100, height=30, wrap=tk.WORD, state='disabled')
output_box.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

def on_submit():
    user = username_entry.get().strip()
    if user:
        run_analysis(user, output_box)

def on_clear():
    username_entry.delete(0, tk.END)
    output_box.configure(state='normal')
    output_box.delete(1.0, tk.END)
    output_box.configure(state='disabled')

submit_btn = tk.Button(root, text="Analyze", command=on_submit)
submit_btn.grid(row=1, column=0, padx=10, pady=5, sticky='e')

clear_btn = tk.Button(root, text="Clear", command=on_clear)
clear_btn.grid(row=1, column=1, padx=10, pady=5, sticky='w')

# Make window resize-friendly
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()
