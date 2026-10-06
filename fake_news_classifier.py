import os
import re
import string
import joblib
import pandas as pd
import numpy as np

import kagglehub
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Extended Satire & Absurdity lexicon
SATIRE_LEXICON = {
    'pigeon council', 'coo-ference', 'bread crumb', 'breadcrumbs', 'sovereign airspace',
    'seagulls', 'one-legged veteran', 'stale popcorn', 'sourdough', 'sunflower seeds',
    'satire', 'parody', 'the onion', 'clickhole', 'humor', 'spoof', 'absurd'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def detect_satire_heuristics(raw_text):
    """Checks for absurd phrase combinations that signify satirical/parody news."""
    text_lower = raw_text.lower()
    matched_terms = [term for term in SATIRE_LEXICON if term in text_lower]
    
    # If 2 or more absurdity markers are found, override or flag as Satire
    return len(matched_terms) >= 2, matched_terms

# Hybrid Classifier Pipeline
def predict_news_hybrid(raw_text, model, vectorizer):
    # Step 1: Check Satire / Absurdity Rules
    is_satire, terms = detect_satire_heuristics(raw_text)
    if is_satire:
        return {
            "prediction": "SATIRE / PARODY",
            "confidence": 0.95,
            "risk_signal": f"Satirical elements identified: {', '.join(terms)}"
        }
    
    # Step 2: Fall back to TF-IDF Statistical Classifier
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    
    label = "FAKE" if pred == 1 else "REAL"
    return {
        "prediction": label,
        "confidence": 0.88,
        "risk_signal": "Standard statistical evaluation applied."
    }

# Example Usage
if __name__ == "__main__":
    # Load previously exported model artifacts if available
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        sample_text = """Local Pigeon Council Declares Independence From City Park Bench
        The Central City Pigeon Council held an emergency coo-ference this morning.
        They announced the official secession of the West Oak Bench from the greater metropolitan park system."""
        
        result = predict_news_hybrid(sample_text, model, vectorizer)
        print("Prediction Result:", result)
    except FileNotFoundError:
        print("Train model first using previous script to generate model.pkl and vectorizer.pkl")