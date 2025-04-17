# Final-Project-CMSC-512: Privacy Ranker
This project aims to develop a Privacy Ranker that evaluates and ranks a users’ social media profile based on their privacy risk. Many users unknowingly expose Personally Identifiable Information (PII) on platforms such as Twitter, LinkedIn, and Instagram, leading to privacy leaks, identity theft, and more. The privacy ranker will analyze user-generated content and determine the likelihood of sensitive information being exposed and suggest privacy settings and actions to take for increased security.

## Running the project with requirements
python -m venv venv  
source venv/bin/activate  # macOS/Linux  
venv\Scripts\activate  # Windows    
pip install -r requirements.txt

## Datasets
This project primarily uses a modified version of the Sentiment140 Twitter Dataset, which contains 1.6 million tweets with basic metadata.
- SentimentTwitterDataset.csv
    The original dataset, containing raw tweets. This file is cleaned and processed using clean_csv.py and then passed through the PII extraction pipeline.
- pii_detected_tweets.csv
    The core dataset used for feature engineering and model training. It includes the fields:
    ['ids', 'user', 'text', 'pii']
    where pii contains extracted named entities (e.g., PER: John, ORG: Microsoft) from the Hugging Face NER model (Jean-Baptiste/roberta-large-ner-english).

## What are stopwords?
Stopwords are common words (like "the," "is," "in," "and") that don't carry much meaningful information in text analysis. It'll help speed up any training step from this point forward.

## NER Model
https://huggingface.co/Jean-Baptiste/roberta-large-ner-english

## SLURM
This project can be executed on a High-Performance Computing (HPC) cluster using [Slurm](https://slurm.schedmd.com/documentation.html), a workload manager for job scheduling. Slurm handles job submission and resource allocation (CPU, GPU, memory) efficiently across compute nodes.

## Machine Learning Models

- **Random Forest Classifier**: Used for classification of risk levels based on textual and structured features.
- **Jean-Baptiste/roberta-large-ner-english**: Hugging Face NER model used to extract PII entities.
- **SentenceTransformer ('all-MiniLM-L6-v2')**: Used as an alternative to TF-IDF for embedding text into semantic space (optional model).

The output of NER is transformed into structured binary features (e.g., `has_per`, `has_org`) for ML classification.

## Model Evaluation
The model’s performance is evaluated using the following metrics:

- **Accuracy**: Percentage of correctly predicted labels.
- **Precision/Recall/F1-Score**: Assesses balance between false positives and false negatives for each class.
- **Confusion Matrix**: Visual breakdown of predictions vs actual labels.

## Project Pipeline

1.  **Data Cleaning & Preprocessing (`clean_csv.py`)**

    -   Loads raw tweet datasets.

    -   Removes unwanted characters, extra spaces, and stopwords.

    -   Prepares text data for Named Entity Recognition (NER).

    -   Outputs a cleaned dataset for further processing.

2.  **Named Entity Recognition (NER) for PII Detection (`named_entity_recognition.py`)**

    -   Applies the Hugging Face model Jean-Baptiste/roberta-large-ner-english to detect PII.

    -   Identifies and labels key entities such as: PER (Person), ORG (Organization), LOC (Location), MISC.

    -   Outputs an annotated CSV (pii_detected_tweets.csv) containing extracted PII information for each text entry.

3.  **Feature Engineering (`feature_processing.py`)**

    -   Encodes presence of detected entity types as binary structured features (e.g., has_per, has_org).

    -   Adds linguistic features such as text length and word count.

    -   Transforms text data using Sentence embeddings with sentence-transformers (all-MiniLM-L6-v2).

    -   Merges structured and text-based features.

    -   Balances class distribution with SMOTE oversampling.

    -   Trains a machine learning classifier (e.g., Random Forest or LightGBM) to predict privacy risk labels:

        -   0 = Low (no PII)

        -   1 = Medium (some PII)

        -   2 = High (multiple types of PII)

    -   Saves the model and vectorizer for later use.

4.  **Training & Testing the Machine Learning Model (`test_model.py`)**

    -   Loads the trained model and embedder.

    -   Accepts new input text or a dataset (e.g., scraped Mastodon toots).

    -   Repeats feature extraction and inference pipeline.

    -   Outputs risk level predictions and classification confidence scores.

    -   Prints a classification report and confusion matrix for evaluation (if ground-truth labels exist).


# Project Timeline  

~~## Project Proposal (Due: **March 2nd**) – **10%**  ~~
Answer the following questions:  
- What is the problem you are solving?  
- What data will you use? (How will you obtain it?)  
- How will you approach the project?  
- Which algorithms/techniques/models will you use or develop?  
- How will you evaluate and measure the quality of your approach?  
- What do you expect to submit/accomplish by the end of the quarter?  

---

~~## Project Milestone Report (Due: **March 31st**) – **15%**  ~~
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
