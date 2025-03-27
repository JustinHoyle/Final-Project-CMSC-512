# Step 3: Feature Engineering and Machine Learning (feature_processing.py)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib

def process_for_ml(input_file):
    """
    Prepares the dataset for machine learning by feature engineering, applying TF-IDF, and training a classifier.
    """
    df = pd.read_csv(input_file)
    unique_users = df['user'].unique()
    sampled_users = pd.Series(unique_users).sample(frac=0.1, random_state=42).tolist()
    df = df[df['user'].isin(sampled_users)]

    # One-hot encode PII types
    pii_categories = ["EMAIL", "PHONE", "PERSON", "ORG", "GPE"]
    for category in pii_categories:
        df[f'has_{category.lower()}'] = df['pii'].str.contains(f"{category}:", na=False).astype(int)
    df['pii_count'] = df[[f'has_{cat.lower()}' for cat in pii_categories]].sum(axis=1)

    # Feature extraction: Text length and word count
    df['text_length'] = df['text'].astype(str).apply(len)
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

    # Risk label: 0 (no PII), 1 (some PII), 2 (high PII)
    df['risk_label'] = df['pii_count'].apply(lambda x: 2 if x >= 2 else (1 if x == 1 else 0))

    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(df['text'])

    # Merge TF-IDF with structured features
    X_structured = df[['text_length', 'word_count'] + [f'has_{cat.lower()}' for cat in pii_categories]].values
    X = pd.concat([pd.DataFrame(X_text.toarray()), pd.DataFrame(X_structured)], axis=1)
    y = df['risk_label']

    # Train-test split and oversampling with SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    """ 
    The code commented out is what was used to find the best hyperparameters.
    It's no longer needed, but I left it here for documentation sake.
    
    print("Setting up GridSearchCV for hyperparameter tuning...")
    # Define the parameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    X_train_resampled_small, y_train_resampled_small = X_train_resampled[:10000], y_train_resampled[:10000]  # Take only the first 10,000 samples
    grid_search.fit(X_train_resampled_small, y_train_resampled_small)

    # Print the best parameters and score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score found: ", grid_search.best_score_)

    # Get the best model from the grid search
    best_rf_model = grid_search.best_estimator_

    print("Predicting on the test set using the best model...")
    y_pred = best_rf_model.predict(X_test) """

    # Train Random Forest model
    print("Training the Random Forest model with the best parameters...")
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

    # Evaluate and save model
    df.to_csv(Path('csv/processed_pii_risk_data_with_ml.csv'), index=False)
    joblib.dump(model, Path('models/random_forest_model.pkl'))
    joblib.dump(vectorizer, Path('models/tfidf_vectorizer.pkl'))


process_for_ml(Path('csv/pii_detected_tweets.csv'))