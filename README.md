# Final-Project-CMSC-512: Privacy Ranker
This project aims to develop a Privacy Ranker that evaluates and ranks a users’ social media profile based on their privacy risk. Many users unknowingly expose Personally Identifiable Information (PII) on platforms such as Twitter, LinkedIn, and Instagram, leading to privacy leaks, identity theft, and more. The privacy ranker will analyze user-generated content and determine the likelihood of sensitive information being exposed and suggest privacy settings and actions to take for increased security.

## Running the project with requirements
python -m venv venv  
source venv/bin/activate  # macOS/Linux  
venv\Scripts\activate  # Windows    
pip install -r requirements.txt

- This will download the english language model for the spacy NLP  
python -m spacy download en_core_web_trf

## Datasets
There are currently two datasets, SentimentTwitterDatase.csv, and cleaned_tweets.csv. 
- pii_detected_tweets.csv will be the set used for any training purposes. Contains the fields ['id', 'user', 'text', 'pii'].
- SentimentTwitterDatase.csv is the original set, but will be unused once processed.
- Download Sentiment dataset here: https://www.kaggle.com/datasets/kazanova/sentiment140

## What are stopwords?
Stopwords are common words (like "the," "is," "in," "and") that don't carry much meaningful information in text analysis. It'll help speed up any training step from this point forward.


## Project Pipeline

1.  **Data Cleaning & Preprocessing (`clean_csv.py`)**

    -   Loads raw tweet datasets.

    -   Removes unwanted characters, extra spaces, and stopwords.

    -   Prepares text data for Named Entity Recognition (NER).

    -   Outputs a cleaned dataset for further processing.

2.  **Named Entity Recognition (NER) for PII Detection (`named_entity_recognition.py`)**

    -   Uses spaCy NLP to detect and extract Personally Identifiable Information (PII).

    -   Identifies key entities:

        -   `EMAIL`, `PHONE`, `PERSON`, `ORG` (Organization), `GPE` (Geopolitical Entity).

    -   Outputs a dataset with PII annotations for feature extraction and model training.

3.  **Feature Engineering (`feature_processing.py`)**

    -   Converts textual and structured PII data into numerical features.

    -   Uses TF-IDF (Term Frequency-Inverse Document Frequency) to transform text.

    -   Generates additional structured features:

        -   Presence of different PII types (binary values).

        -   Text length and word count.

    -   Outputs a structured dataset for machine learning model training.

4.  **Training & Testing the Machine Learning Model (`test_model.py`)**

    -   Loads the trained Random Forest model and TF-IDF vectorizer.

    -   Processes new input text to extract PII-based features.

    -   Predicts the privacy risk level of a user's profile.

    -   Outputs risk labels that can be used for privacy recommendations.


# Project Timeline  

## Project Proposal (Due: **March 2nd**) – **10%**  
Answer the following questions:  
- What is the problem you are solving?  
- What data will you use? (How will you obtain it?)  
- How will you approach the project?  
- Which algorithms/techniques/models will you use or develop?  
- How will you evaluate and measure the quality of your approach?  
- What do you expect to submit/accomplish by the end of the quarter?  

---

## Project Milestone Report (Due: **March 31st**) – **15%**  
- **Length**: 4-6 pages in LaTeX (or equivalent in MS Word/Google Docs).  
- **File Format**: Single `.zip` file containing:  
  - PDF of the report  
  - LaTeX source files (if applicable)  
- **Submission**: Upload to Canvas.  

---

## Project Final Report (Due: **April 21st**) – **60%**  
- **Length**: 5-8 pages, single-spaced in LaTeX (or equivalent in MS Word/Google Docs).  
- **File Format**: Single `.zip` file containing:  
  - PDF of the final report  
  - LaTeX source files (if applicable)  
  - Data and experimental results  
- **Submission**: Upload to Canvas.  

---

## Project Presentation (April 28th) – **15%**  
- Present your work before class.  
