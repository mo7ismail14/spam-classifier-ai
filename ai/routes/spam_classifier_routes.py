"""Spam classifier API routes: model discovery, sample emails, and prediction."""

import os
import joblib
from flask import Blueprint, current_app, jsonify, request

spam_classifier_routes = Blueprint('spam_classifier', __name__)

# name -> (model, vectorizer, metrics)
_model_cache = {}
_sample_emails = None


def _model_dir() -> str:
    return current_app.config['MODEL_DIR']


def list_available_spam_models() -> list[str]:
    """A folder qualifies by having both model.pkl and vectorizer.pkl, which excludes
    the loan Decision Tree (it has preprocessor.pkl instead)."""
    model_dir = _model_dir()
    if not os.path.isdir(model_dir):
        return []
    names = []
    for entry in sorted(os.scandir(model_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        has_model = os.path.exists(os.path.join(entry.path, 'model.pkl'))
        has_vectorizer = os.path.exists(os.path.join(entry.path, 'vectorizer.pkl'))
        if has_model and has_vectorizer:
            names.append(entry.name)
    return names


def load_spam_model_artifacts(name: str):
    if name in _model_cache:
        return _model_cache[name]

    model_dir = os.path.join(_model_dir(), name)
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
    global _sample_emails
    if _sample_emails is None:
        samples_path = os.path.join(_model_dir(), 'sample_emails.pkl')
        if os.path.exists(samples_path):
            _sample_emails = joblib.load(samples_path)
    return _sample_emails


@spam_classifier_routes.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({"models": list_available_spam_models()})


@spam_classifier_routes.route('/api/emails', methods=['GET'])
def get_emails():
    emails = load_sample_emails()
    if emails is None:
        return jsonify({"error": "Model not trained yet. Please run training/train_spam_classifier.py first."}), 503
    return jsonify(emails)


@spam_classifier_routes.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get('text', '')
    model_name = data.get('model', '')

    if not text.strip():
        return jsonify({"error": "No email text provided"}), 400
    if not model_name:
        return jsonify({"error": "No model specified", "available_models": list_available_spam_models()}), 400

    artifacts = load_spam_model_artifacts(model_name)
    if artifacts is None:
        return jsonify({"error": f"Unknown model '{model_name}'", "available_models": list_available_spam_models()}), 404

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
