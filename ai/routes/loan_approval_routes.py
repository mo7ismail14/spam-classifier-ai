"""Loan approval API routes: form schema, sample applications, and prediction with explanation."""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, request
from sklearn.tree import export_text

from functions.decisiontree.data_preprocessing import (
    FEATURE_DISPLAY_NAMES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    get_categorical_encoder
)
from functions.decisiontree.rule_extraction import explain_prediction

loan_approval_routes = Blueprint('loan_approval', __name__, url_prefix='/api/loan')

NUMERIC_FIELD_NAMES = set(NUMERIC_FEATURES) | set(BINARY_FEATURES)

# Cached (model, preprocessor, metrics, feature_order, cv_results, threshold) for the
# loan Decision Tree. Cleared via POST /api/loan/reload after retraining.
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
            threshold_path = os.path.join(model_dir, 'threshold.pkl')
            threshold = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.5
            _loan_artifacts = (model, preprocessor, metrics, feature_order, cv_results, threshold)
    return _loan_artifacts


def load_loan_sample_applications():
    global _loan_sample_applications
    if _loan_sample_applications is None:
        samples_path = _loan_samples_path()
        if os.path.exists(samples_path):
            _loan_sample_applications = joblib.load(samples_path)
    return _loan_sample_applications


def _validate_and_coerce_applicant(applicant: dict, feature_order: list[str]) -> tuple[dict, list[str]]:
    """Missing/blank fields are allowed (the preprocessor imputes them). A field that IS
    provided must still have the right shape: a real number for numeric fields, a string
    for categorical fields."""
    coerced: dict = {}
    errors: list[str] = []

    for field in feature_order:
        value = applicant.get(field)
        if value is None or value == '':
            coerced[field] = None
            continue

        if field in NUMERIC_FIELD_NAMES:
            try:
                coerced[field] = float(value)
            except (TypeError, ValueError):
                errors.append(f"{field} must be a valid number")
        else:
            if not isinstance(value, str):
                errors.append(f"{field} must be a string")
            else:
                coerced[field] = value

    return coerced, errors


@loan_approval_routes.route('/schema', methods=['GET'])
def get_loan_schema():
    artifacts = load_loan_artifacts()
    if artifacts is None:
        return jsonify({"error": "Loan model not trained yet. Please run training/train_decision_tree.py first."}), 503

    _, preprocessor, metrics, feature_order, cv_results, _ = artifacts
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

    model, preprocessor, metrics, feature_order, _, threshold = artifacts
    applicant_raw = request.get_json(force=True, silent=True) or {}

    applicant, validation_errors = _validate_and_coerce_applicant(applicant_raw, feature_order)
    if validation_errors:
        return jsonify({"error": '; '.join(validation_errors)}), 400

    input_row = pd.DataFrame([{field: applicant[field] for field in feature_order}])
    # Normalize both Python None and NaN to np.nan uniformly, numeric and categorical
    # columns alike, so the preprocessor's imputers reliably recognize missing values.
    input_row = input_row.where(pd.notnull(input_row), np.nan)
    encoded_input = preprocessor.transform(input_row)

    probability_approved = float(model.predict_proba(encoded_input)[0, 1])
    prediction_label = 'Y' if probability_approved >= threshold else 'N'

    explanation = explain_prediction(model, preprocessor, feature_order, applicant, prediction_label)

    return jsonify({
        'prediction': prediction_label,
        'probability': probability_approved,
        'explanation': explanation,
        'metrics': metrics
    })


@loan_approval_routes.route('/reload', methods=['POST'])
def reload_loan_model():
    """Clear cached artifacts so a freshly retrained model loads without restarting Flask."""
    global _loan_artifacts, _loan_sample_applications
    _loan_artifacts = None
    _loan_sample_applications = None

    artifacts = load_loan_artifacts()
    load_loan_sample_applications()

    if artifacts is None:
        return jsonify({"status": "no_model_found", "message": "No trained model artifacts were found to load."}), 503
    return jsonify({"status": "reloaded"})


@loan_approval_routes.route('/tree-text', methods=['GET'])
def get_tree_text():
    artifacts = load_loan_artifacts()
    if artifacts is None:
        return jsonify({"error": "Loan model not trained yet. Please run training/train_decision_tree.py first."}), 503

    model, _, _, feature_order, _, _ = artifacts
    tree_text = export_text(model, feature_names=feature_order)
    return jsonify({'tree_text': tree_text})
