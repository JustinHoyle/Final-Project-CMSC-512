from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import multiprocessing as mp
import torch
from tqdm import tqdm

mp.set_start_method("spawn", force=True)

# Load model and tokenizer for Jean-Baptiste/roberta-large-ner-english
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

# Create NER pipeline
device = 0 if torch.cuda.is_available() else -1  # GPU if available, else CPU
# Grouped entities = clean output with `entity_group`
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    device=device,
    grouped_entities=True
)

#def parallel_process(df, func, text_column='text'):
#    with mp.Pool(mp.cpu_count()) as pool:
#        results = pool.map(func, df[text_column])
#    return results

def extract_pii_hf(text, idx=None, total=None):
    if pd.isna(text) or not isinstance(text, str):
        return "No PII Detected"

    detected_pii = set()
    results = ner_pipeline(text)

    for entity in results:
        if entity["entity_group"] in ["PER", "ORG", "LOC", "MISC"]:
            detected_pii.add(f"{entity['entity_group']}: {entity['word']}")

    if idx is not None and idx % 100 == 0:
        print(f"[{idx}/{total}] Processed...")

    return ", ".join(detected_pii) if detected_pii else "No PII Detected"


# Load dataset and apply PII extraction
print("Starting NER Python...")
df = pd.read_csv("SentimentTwitterDataset.csv", encoding='latin-1', header=None)
df.drop(df.columns[[0, 2, 3]], axis=1, inplace=True)
df.columns = ['ids', 'user', 'text']
total_rows = len(df)
df['pii'] = [
    extract_pii_hf(text, idx=i, total=total_rows)
    for i, text in enumerate(df['text'])
]
df.to_csv("pii_detected_tweets.csv", index=False)