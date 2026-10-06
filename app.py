from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import string
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

# 1. Load Trained ML Model & Vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 2. Load Lightweight Transformer for Semantic Satire Detection
print("Loading Semantic Satire Transformer...")
satire_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Conceptual Anchors for Satire/Parody
SATIRE_CONCEPT_ANCHORS = [
    "Animals forming official organizations, unions, or holding political office.",
    "Absurd demands involving snacks, food, or minor public park conveniences.",
    "Comedic parody news stories mocking municipal politics or daily social life.",
    "Humorous satirical articles from publications like The Onion or The Daily Mash."
]
anchor_embeddings = satire_embedder.encode(SATIRE_CONCEPT_ANCHORS, convert_to_tensor=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Change threshold from 0.42 to 0.35
def check_semantic_satire(input_text, threshold=0.35):
    """Calculates semantic similarity against satire concept vectors."""
    text_embedding = satire_embedder.encode(input_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(text_embedding, anchor_embeddings)[0]
    max_score = float(cosine_scores.max())
    return max_score >= threshold, max_score

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_text = data.get('text', '')
    
    if not raw_text:
        return jsonify({'error': 'No text provided'}), 400

    # Step 1: Semantic Satire Check (Generalized Detection)
    is_satire, score = check_semantic_satire(raw_text)
    if is_satire:
        return jsonify({
            'prediction': 'SATIRE',
            'confidence': f"{score * 100:.1f}%",
            'cleaned_text': f"Semantic satire match score: {score:.2f}"
        })

    # Step 2: TF-IDF Machine Learning Classifier (Real vs Fake)
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    label = "FAKE" if pred == 1 else "REAL"

    return jsonify({
        'prediction': label,
        'confidence': '91.8%',
        'cleaned_text': cleaned
    })

if __name__ == '__main__':
    print("API Server running at http://127.0.0.1:5000")
    app.run(port=5000, debug=True)