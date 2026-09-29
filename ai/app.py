import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
SAMPLES_PATH = os.path.join(MODEL_DIR, 'sample_emails.pkl')

# name -> (model, vectorizer, metrics)
_model_cache = {}
sample_emails = None


def list_available_models():
    if not os.path.isdir(MODEL_DIR):
        return []
    names = []
    for entry in sorted(os.scandir(MODEL_DIR), key=lambda e: e.name):
        if entry.is_dir() and os.path.exists(os.path.join(entry.path, 'model.pkl')):
            names.append(entry.name)
    return names


def load_model_artifacts(name):
    if name in _model_cache:
        return _model_cache[name]

    model_dir = os.path.join(MODEL_DIR, name)
    model_path = os.path.join(model_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    metrics_path = os.path.join(model_dir, 'metrics.pkl')

    if not os.path.exists(model_path):
        return None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    metrics = joblib.load(metrics_path)
    _model_cache[name] = (model, vectorizer, metrics)
    return _model_cache[name]


def load_sample_emails():
    global sample_emails
    if sample_emails is None and os.path.exists(SAMPLES_PATH):
        sample_emails = joblib.load(SAMPLES_PATH)
    return sample_emails


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})


@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({"models": list_available_models()})


@app.route('/api/emails', methods=['GET'])
def get_emails():
    emails = load_sample_emails()
    if emails is None:
        return jsonify({"error": "Model not trained yet. Please run train.py first."}), 503
    return jsonify(emails)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get('text', '')
    model_name = data.get('model', '')

    if not text.strip():
        return jsonify({"error": "No email text provided"}), 400
    if not model_name:
        return jsonify({"error": "No model specified", "available_models": list_available_models()}), 400

    artifacts = load_model_artifacts(model_name)
    if artifacts is None:
        return jsonify({"error": f"Unknown model '{model_name}'", "available_models": list_available_models()}), 404

    model, vectorizer, metrics = artifacts

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
        'metrics': metrics,
        'model': model_name
    })


load_sample_emails()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask AI Server on port {port}...")
    print(f"Available models: {list_available_models()}")
    app.run(host='0.0.0.0', port=port, debug=False)
