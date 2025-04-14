import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

def prepare_data_for_transformer(input_file):
    df = pd.read_csv(input_file)
    pii_categories = ["PERSON", "ORG", "GPE"]
    
    for category in pii_categories:
        df[f'has_{category.lower()}'] = df['pii'].str.contains(f"{category}:", na=False).astype(int)
    df['pii_count'] = df[[f'has_{cat.lower()}' for cat in pii_categories]].sum(axis=1)
    df['risk_label'] = df['pii_count'].apply(lambda x: 2 if x >= 2 else (1 if x == 1 else 0))

    sampled_users = pd.Series(df['user'].unique()).sample(frac=0.1, random_state=42)
    df = df[df['user'].isin(sampled_users)]

    train_df, test_df = train_test_split(df[['text', 'risk_label']], test_size=0.2, random_state=42)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

train_ds, test_ds = prepare_data_for_transformer('csv/pii_detected_tweets.csv')

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'risk_label'])
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'risk_label'])

# Replace 'risk_label' with 'labels'
train_ds = train_ds.rename_column("risk_label", "labels")
test_ds = test_ds.rename_column("risk_label", "labels")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Evaluation metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Training configuration
training_args = TrainingArguments(
    output_dir="./transformer_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("models/transformer_pii_risk_model")