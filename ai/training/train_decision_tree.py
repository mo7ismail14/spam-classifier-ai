"""Train a Decision Tree that predicts loan approval, with grid-searched hyperparameters,
a held-out confusion matrix, a more reliable repeated-CV estimate, a tuned decision
threshold, and saved artifacts for the Flask API to serve predictions and plain-English
explanations from.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_validate
)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
THRESHOLD_SEARCH_RANGE = np.arange(0.30, 0.80 + 1e-9, 0.02)


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """Stratified 75/25 train/test split, matching the spam classifier's convention."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")
    return X_train, X_test, y_train, y_test


def tune_decision_threshold(best_pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Pick the probability threshold that maximizes macro F1, using only out-of-fold
    predictions on the TRAINING split (never touches the test split)."""
    print("Tuning decision threshold via out-of-fold predictions on the training split only...")
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(
        clone(best_pipeline), X_train, y_train, cv=skfold, method='predict_proba', n_jobs=-1
    )[:, 1]

    best_threshold, best_score = 0.5, -1.0
    for threshold in THRESHOLD_SEARCH_RANGE:
        predictions = (oof_probs >= threshold).astype(int)
        score = f1_score(y_train, predictions, average='macro', zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), score

    print(f"Selected decision threshold: {best_threshold:.2f} (out-of-fold macro F1: {best_score:.4f})")
    return best_threshold


def compute_cv_summary(best_pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """5x5 RepeatedStratifiedKFold on the FULL cleaned dataset — a more reliable estimate
    than the ~154-row held-out test split alone."""
    print("Running RepeatedStratifiedKFold (5x5) for a more reliable performance estimate...")
    repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores = cross_validate(
        clone(best_pipeline), X, y, cv=repeated_cv,
        scoring=['accuracy', 'f1_macro', 'recall'], n_jobs=-1
    )
    summary = {
        'accuracy_mean': float(np.mean(scores['test_accuracy'])),
        'accuracy_std': float(np.std(scores['test_accuracy'])),
        'f1_macro_mean': float(np.mean(scores['test_f1_macro'])),
        'f1_macro_std': float(np.std(scores['test_f1_macro'])),
        'recall_mean': float(np.mean(scores['test_recall'])),
        'recall_std': float(np.std(scores['test_recall']))
    }
    print(
        f"CV Summary (5x5 RepeatedStratifiedKFold): "
        f"accuracy={summary['accuracy_mean']:.4f}+/-{summary['accuracy_std']:.4f}, "
        f"macro F1={summary['f1_macro_mean']:.4f}+/-{summary['f1_macro_std']:.4f}, "
        f"recall={summary['recall_mean']:.4f}+/-{summary['recall_std']:.4f}"
    )
    return summary


def _score_block(y_test, y_pred, prefix: str = '') -> dict:
    """Confusion matrix + accuracy/precision/recall/F1 for Approved, plus Denied-class
    precision/recall, plus macro F1. Keys are prefixed (e.g. 'tuned_') for a second block."""
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        f'{prefix}tn': int(tn), f'{prefix}fp': int(fp), f'{prefix}fn': int(fn), f'{prefix}tp': int(tp),
        f'{prefix}accuracy': float(accuracy_score(y_test, y_pred)),
        f'{prefix}precision': float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        f'{prefix}recall': float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        f'{prefix}f1': float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        f'{prefix}denied_precision': float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
        f'{prefix}denied_recall': float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        f'{prefix}macro_f1': float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    }


def evaluate_model(
    model, X_test_encoded, y_test, best_params: dict, feature_order: list[str], tuned_threshold: float
) -> dict:
    """Confusion matrix + full metrics on the held-out test split, at both the default
    0.5 threshold (tn/fp/fn/tp keys stay backward compatible) and the tuned threshold."""
    print("Calculating Confusion Matrix on Test Set...")
    y_pred_default = model.predict(X_test_encoded)
    probabilities = model.predict_proba(X_test_encoded)[:, 1]
    y_pred_tuned = (probabilities >= tuned_threshold).astype(int)

    metrics = {}
    metrics.update(_score_block(y_test, y_pred_default))
    metrics.update(_score_block(y_test, y_pred_tuned, prefix='tuned_'))
    metrics['best_params'] = best_params
    metrics['decision_threshold'] = float(tuned_threshold)
    metrics['feature_importances'] = {
        name: float(importance) for name, importance in zip(feature_order, model.feature_importances_)
    }

    tn, fp, fn, tp = metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']
    print(f"Metrics (default 0.5 threshold): TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    t_tn, t_fp, t_fn, t_tp = metrics['tuned_tn'], metrics['tuned_fp'], metrics['tuned_fn'], metrics['tuned_tp']
    print(f"Metrics (tuned {tuned_threshold:.2f} threshold): TN={t_tn}, FP={t_fp}, FN={t_fn}, TP={t_tp}")
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


def save_artifacts(
    model_dir: str, model, preprocessor, metrics: dict, feature_order: list[str],
    cv_results: list[dict], threshold: float
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    joblib.dump(metrics, os.path.join(model_dir, 'metrics.pkl'))
    joblib.dump(feature_order, os.path.join(model_dir, 'feature_order.pkl'))
    joblib.dump(cv_results, os.path.join(model_dir, 'cv_results.pkl'))
    joblib.dump(threshold, os.path.join(model_dir, 'threshold.pkl'))
    print(f"Saved model, preprocessor, metrics, feature order, grid-search results, and threshold to {model_dir}")


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

    # The preprocessor is UNFITTED here; train_decision_tree() fits it fresh inside each
    # CV fold via the Pipeline, so no fold ever leaks into another fold's imputation stats.
    preprocessor = build_preprocessor()
    best_pipeline, best_params, cv_results = train_decision_tree(preprocessor, X_train, y_train)

    fitted_preprocessor = best_pipeline.named_steps['preprocess']
    model = best_pipeline.named_steps['tree']
    X_test_encoded = fitted_preprocessor.transform(X_test)

    tuned_threshold = tune_decision_threshold(best_pipeline, X_train, y_train)

    metrics = evaluate_model(model, X_test_encoded, y_test, best_params, ENCODED_FEATURE_ORDER, tuned_threshold)
    metrics['cv_summary'] = compute_cv_summary(best_pipeline, X, y)

    save_artifacts(
        DECISION_TREE_DIR, model, fitted_preprocessor, metrics, ENCODED_FEATURE_ORDER, cv_results, tuned_threshold
    )

    sample_applications = prepare_sample_applications(X_test, y_test, SAMPLE_APPLICATIONS_COUNT)
    save_sample_applications(MODEL_DIR, sample_applications)

    print("Loan approval model training and serialization successfully completed!")


if __name__ == '__main__':
    main()
