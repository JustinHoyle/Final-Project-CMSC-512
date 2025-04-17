import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def prepare_data_for_testing(new_data, vectorizer):
    new_df = pd.DataFrame(new_data)
    new_df['has_email'] = new_df['pii'].str.contains("EMAIL:", na=False).astype(int)
    new_df['has_phone'] = new_df['pii'].str.contains("PHONE:", na=False).astype(int)
    new_df['has_person'] = new_df['pii'].str.contains("PERSON:", na=False).astype(int)
    new_df['has_org'] = new_df['pii'].str.contains("ORG:", na=False).astype(int)
    new_df['has_gpe'] = new_df['pii'].str.contains("GPE:", na=False).astype(int)
    new_df['text_length'] = new_df['text'].astype(str).apply(len)
    new_df['word_count'] = new_df['text'].astype(str).apply(lambda x: len(x.split()))
    X_text = vectorizer.transform(new_df['text'])
    X_structured = new_df[['has_email', 'has_phone', 'has_person', 'has_org', 'has_gpe', 'text_length', 'word_count']].values
    X_new = pd.concat([pd.DataFrame(X_text.toarray()), pd.DataFrame(X_structured)], axis=1)
    return X_new

def make_predictions(model, X_new):
    y_pred = model.predict(X_new)
    return y_pred

def evaluate_predictions(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

def main():
    model_path = 'models/random_forest_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    new_data = [
        {"user": "user1", "text": "My email is john.doe@example.com", "pii": "EMAIL:john.doe@example.com"},
        {"user": "user2", "text": "I live in Virginia, my name is Jane", "pii": "GPE:Virginia,PER:Jane"},
    ]
    X_new = prepare_data_for_testing(new_data, vectorizer)
    y_pred = make_predictions(model, X_new)
    print("Predicted risk labels:", y_pred)
    y_true = [0, 1]
    evaluate_predictions(y_true, y_pred)

if __name__ == "__main__":
    main()