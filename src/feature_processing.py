import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib

def process_for_ml(input_file):
    df = pd.read_csv(input_file)
    unique_users = df['user'].unique()
    sampled_users = pd.Series(unique_users).sample(frac=0.1, random_state=42).tolist()
    df = df[df['user'].isin(sampled_users)]

    pii_categories = ["PER", "ORG", "LOC", "MISC"]
    for category in pii_categories:
        df[f'has_{category.lower()}'] = df['pii'].str.contains(f"{category}:", na=False).astype(int)
    df['pii_count'] = df[[f'has_{cat.lower()}' for cat in pii_categories]].sum(axis=1)

    df['text_length'] = df['text'].astype(str).apply(len)
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

    df['risk_label'] = df['pii_count'].apply(lambda x: 2 if x >= 2 else (1 if x == 1 else 0))

    print("Generating sentence embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X_text = embedder.encode(df['text'].astype(str).tolist(), show_progress_bar=True)

    X_structured = df[['text_length', 'word_count'] + [f'has_{cat.lower()}' for cat in pii_categories]].values
    X = np.concatenate([X_text, X_structured], axis=1)
    y = df['risk_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Training the Random Forest model...")
    model = RandomForestClassifier(
        max_depth=20, 
        max_features='sqrt', 
        min_samples_leaf=4, 
        min_samples_split=10, 
        n_estimators=200, 
        class_weight='balanced', 
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)

    print("Predicting on the test set...")
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, Path('random_forest_model_embed.pkl'))
    joblib.dump(embedder, Path('sentence_transformer_embedder.pkl'))

if __name__ == "__main__":
    process_for_ml(Path('../models/pii_detected_tweets_old.csv'))
