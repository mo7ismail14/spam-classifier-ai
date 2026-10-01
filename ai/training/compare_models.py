"""Benchmark DecisionTree (tuned) against RandomForest and GradientBoosting, all using
the same preprocessing pipeline, to justify keeping the Decision Tree for explainability.

This script does NOT change the artifacts served by the Flask API — it only prints a
comparison table.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score

AI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)

from functions.decisiontree.data_preprocessing import (
    load_loan_dataset,
    clean_loan_data,
    split_features_and_target,
    build_preprocessor
)

TRAIN_CSV_PATH = os.path.join(AI_DIR, 'dataset', 'train_u6lujuX_CVtuZ9i.csv')
DECISION_TREE_DIR = os.path.join(AI_DIR, 'model', 'DecisionTree')

# The Decision Tree's own GridSearchCV-selected hyperparameters (ai/model/DecisionTree/metrics.pkl).
BEST_DECISION_TREE_PARAMS = {
    'criterion': 'gini',
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': None,
    'ccp_alpha': 0.01,
    'random_state': 42
}

denied_recall_scorer = make_scorer(recall_score, pos_label=0, zero_division=0)


def build_candidates() -> dict:
    return {
        'DecisionTree (tuned)': DecisionTreeClassifier(**BEST_DECISION_TREE_PARAMS),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }


def benchmark(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 'denied_recall': denied_recall_scorer}

    results = []
    for name, estimator in build_candidates().items():
        pipeline = Pipeline(steps=[('preprocess', build_preprocessor()), ('model', estimator)])
        scores = cross_validate(pipeline, X, y, cv=repeated_cv, scoring=scoring, n_jobs=-1)
        results.append({
            'name': name,
            'accuracy_mean': float(np.mean(scores['test_accuracy'])),
            'accuracy_std': float(np.std(scores['test_accuracy'])),
            'f1_macro_mean': float(np.mean(scores['test_f1_macro'])),
            'f1_macro_std': float(np.std(scores['test_f1_macro'])),
            'denied_recall_mean': float(np.mean(scores['test_denied_recall'])),
            'denied_recall_std': float(np.std(scores['test_denied_recall']))
        })
    return results


def print_table(results: list[dict]) -> None:
    header = f"{'Model':<22}{'Accuracy':>18}{'Macro F1':>18}{'Denied Recall':>18}"
    print(header)
    print('-' * len(header))
    for r in results:
        acc = f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f}"
        f1 = f"{r['f1_macro_mean']:.4f}+/-{r['f1_macro_std']:.4f}"
        rec = f"{r['denied_recall_mean']:.4f}+/-{r['denied_recall_std']:.4f}"
        print(f"{r['name']:<22}{acc:>18}{f1:>18}{rec:>18}")


def main():
    print("Loading Loan Approval dataset for model comparison...")
    df = load_loan_dataset(TRAIN_CSV_PATH)
    df = clean_loan_data(df)
    X, y = split_features_and_target(df)

    print("Running RepeatedStratifiedKFold (5x5) for each candidate model...")
    results = benchmark(X, y)

    print()
    print("Model comparison (5x5 RepeatedStratifiedKFold, mean +/- std):")
    print_table(results)


if __name__ == '__main__':
    main()
