"""Loan approval API routes: form schema, sample applications, and prediction with explanation."""

import os
import joblib
import pandas as pd
from flask import Blueprint, current_app, jsonify, request

from functions.decisiontree.data_preprocessing import (
    FEATURE_DISPLAY_NAMES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    get_categorical_encoder
)
from functions.decisiontree.rule_extraction import explain_prediction

loan_approval_routes = Blueprint('loan_approval', __name__, url_prefix='/api/loan')

# Cached (model, preprocessor, metrics, feature_order, cv_results) for the loan Decision Tree.
_loan_artifacts = None
_loan_sample_applications = None


def _decision_tree_dir() -> str:
    return os.path.join(current_app.config['MODEL_DIR'], 'DecisionTree')


def _loan_samples_path() -> str:
    return os.path.join(current_app.config['MODEL_DIR'], 'loan_sample_applications.pkl')


def load_loan_artifacts():
    global _loan_artifacts
    if _loan_artifacts is None:
        model_dir = _decision_tree_dir()
        model_path = os.path.join(model_dir, 'model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
            metrics = joblib.load(os.path.join(model_dir, 'metrics.pkl'))
            feature_order = joblib.load(os.path.join(model_dir, 'feature_order.pkl'))
            cv_results_path = os.path.join(model_dir, 'cv_results.pkl')
            cv_results = joblib.load(cv_results_path) if os.path.exists(cv_results_path) else []
            _loan_artifacts = (model, preprocessor, metrics, feature_order, cv_results)
    return _loan_artifacts


def load_loan_sample_applications():
    global _loan_sample_applications
    if _loan_sample_applications is None:
        samples_path = _loan_samples_path()
        if os.path.exists(samples_path):
            _loan_sample_applications = joblib.load(samples_path)
    return _loan_sample_applications


@loan_approval_routes.route('/schema', methods=['GET'])
def get_loan_schema():
    artifacts = load_loan_artifacts()
    if artifacts is None:
        return jsonify({"error": "Loan model not trained yet. Please run training/train_decision_tree.py first."}), 503

    _, preprocessor, metrics, feature_order, cv_results = artifacts
    encoder = get_categorical_encoder(preprocessor)

    categorical_options = {
        feature_name: [str(category) for category in categories]
        for feature_name, categories in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }

    fields = [
        {
            'name': feature_name,
            'label': FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
            'type': 'select' if feature_name in categorical_options else 'number',
            'options': categorical_options.get(feature_name)
        }
        for feature_name in feature_order
    ]

    return jsonify({
        'fields': fields,
        'metrics': metrics,
        'model_type': 'Decision Tree',
        'cv_results': cv_results
    })


@loan_approval_routes.route('/samples', methods=['GET'])
def get_loan_samples():
    samples = load_loan_sample_applications()
    if samples is None:
        return jsonify({"error": "Loan model not trained yet. Please run training/train_decision_tree.py first."}), 503
    return jsonify(samples)


@loan_approval_routes.route('/predict', methods=['POST'])
def predict_loan():
    artifacts = load_loan_artifacts()
    if artifacts is None:
        return jsonify({"error": "Loan model not trained yet. Please run training/train_decision_tree.py first."}), 503

    model, preprocessor, metrics, feature_order, _ = artifacts
    applicant = request.get_json(force=True, silent=True) or {}

    missing_fields = [field for field in feature_order if applicant.get(field) in (None, '')]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    numeric_field_names = set(NUMERIC_FEATURES) | set(BINARY_FEATURES)
    try:
        applicant = {
            field: (float(applicant[field]) if field in numeric_field_names else applicant[field])
            for field in feature_order
        }
    except (TypeError, ValueError):
        return jsonify({"error": "Numeric fields must contain valid numbers"}), 400

    input_row = pd.DataFrame([{field: applicant[field] for field in feature_order}])
    encoded_input = preprocessor.transform(input_row)

    prediction_numeric = model.predict(encoded_input)[0]
    prediction_label = 'Y' if prediction_numeric == 1 else 'N'

    explanation = explain_prediction(model, preprocessor, feature_order, applicant, prediction_label)

    return jsonify({
        'prediction': prediction_label,
        'explanation': explanation,
        'metrics': metrics
    })
