"""Train a Decision Tree that predicts loan approval, with grid-searched hyperparameters,
a held-out confusion matrix, and saved artifacts for the Flask API to serve predictions
and plain-English explanations from.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ai/ is the project root for sibling packages (functions/); this script lives in ai/training/.
AI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)

from functions.decisiontree.data_preprocessing import (
    load_loan_dataset,
    clean_loan_data,
    split_features_and_target,
    build_preprocessor,
    ENCODED_FEATURE_ORDER
)
from functions.decisiontree.decision_tree_model import train_decision_tree

TRAIN_CSV_PATH = os.path.join(AI_DIR, 'dataset', 'train_u6lujuX_CVtuZ9i.csv')
MODEL_DIR = os.path.join(AI_DIR, 'model')
DECISION_TREE_DIR = os.path.join(MODEL_DIR, 'DecisionTree')
SAMPLE_APPLICATIONS_COUNT = 50


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """Stratified 75/25 train/test split, matching the spam classifier's convention."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test_encoded, y_test, best_params: dict) -> dict:
    """Compute a confusion matrix on the held-out test split."""
    print("Calculating Confusion Matrix on Test Set...")
    y_pred = model.predict(X_test_encoded)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'best_params': best_params
    }
    print(f"Metrics: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return metrics


def prepare_sample_applications(X_test: pd.DataFrame, y_test: pd.Series, n: int) -> list[dict]:
    """Pull real, held-out (never trained on) applicants for the frontend's 'load an example'
    dropdown, each tagged with its true Loan_Status so the UI can show Actual vs. Predicted."""
    sample_size = min(n, len(X_test))
    sample_indices = X_test.sample(sample_size, random_state=42).index

    samples = []
    for idx in sample_indices:
        row = {}
        for col in X_test.columns:
            value = X_test.loc[idx, col]
            if pd.isna(value):
                row[col] = None
            elif hasattr(value, 'item'):
                row[col] = value.item()  # numpy scalar (e.g. int64/float64) -> native Python type
            else:
                row[col] = value
        row['Loan_Status'] = 'Y' if y_test.loc[idx] == 1 else 'N'
        samples.append(row)
    return samples


def save_artifacts(model_dir: str, model, preprocessor, metrics: dict, feature_order: list[str], cv_results: list[dict]) -> None:
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    joblib.dump(metrics, os.path.join(model_dir, 'metrics.pkl'))
    joblib.dump(feature_order, os.path.join(model_dir, 'feature_order.pkl'))
    joblib.dump(cv_results, os.path.join(model_dir, 'cv_results.pkl'))
    print(f"Saved model, preprocessor, metrics, feature order, and grid-search results to {model_dir}")


def save_sample_applications(model_dir: str, samples: list[dict]) -> None:
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(samples, os.path.join(model_dir, 'loan_sample_applications.pkl'))
    print(f"Saved {len(samples)} sample applications to {model_dir}")


def main():
    print("Loading Loan Approval dataset...")
    df = load_loan_dataset(TRAIN_CSV_PATH)
    df = clean_loan_data(df)

    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("Fitting preprocessor (imputation + categorical encoding) on training data...")
    preprocessor = build_preprocessor()
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    model, best_params, cv_results = train_decision_tree(X_train_encoded, y_train)
    metrics = evaluate_model(model, X_test_encoded, y_test, best_params)

    save_artifacts(DECISION_TREE_DIR, model, preprocessor, metrics, ENCODED_FEATURE_ORDER, cv_results)

    sample_applications = prepare_sample_applications(X_test, y_test, SAMPLE_APPLICATIONS_COUNT)
    save_sample_applications(MODEL_DIR, sample_applications)

    print("Loan approval model training and serialization successfully completed!")


if __name__ == '__main__':
    main()
