import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.pkl')
SAMPLES_PATH = os.path.join(MODEL_DIR, 'sample_emails.pkl')

# Global variables for models
model = None
vectorizer = None
metrics = None
sample_emails = None

def load_resources():
    global model, vectorizer, metrics, sample_emails
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            metrics = joblib.load(METRICS_PATH)
            sample_emails = joblib.load(SAMPLES_PATH)
            print("Successfully loaded model, vectorizer, metrics, and sample emails.")
        except Exception as e:
            print(f"Error loading models: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})


@app.route('/api/emails', methods=['GET'])
def get_emails():
    load_resources()
    if sample_emails is None:
        return jsonify({"error": "Model not trained yet. Please run train.py first."}), 503
    return jsonify(sample_emails)

@app.route('/api/predict', methods=['POST'])
def predict():
    load_resources()
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet. Please run train.py first."}), 503
        
    data = request.get_json(force=True, silent=True) or {}
    text = data.get('text', '')
    if not text.strip():
        return jsonify({"error": "No email text provided"}), 400
        
    # Transform text using fitted vectorizer
    vec_text = vectorizer.transform([text])
    
    # Predict spam or ham
    prediction_numeric = model.predict(vec_text)[0]
    probabilities = model.predict_proba(vec_text)[0]
    
    label = 'spam' if prediction_numeric == 1 else 'ham'
    confidence = float(max(probabilities))
    
    return jsonify({
        'prediction': label,
        'confidence': confidence,
        'metrics': metrics
    })

load_resources()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask AI Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
