"""Decision Tree training for loan approval, tuned via GridSearchCV + K-Fold CV."""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold


def train_decision_tree(X_train_encoded, y_train):
    """Grid-search a DecisionTreeClassifier's hyperparameters.

    Returns (best_model, best_params, cv_results) where cv_results is every
    parameter combination tried, with its mean CV F1 score, sorted best-first
    — used by the API/frontend to show the full search and mark the winner.
    """
    print("[DecisionTree] Configuring K-Fold Cross-Validation & Hyperparameter Grid...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 6, 8, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        # 'balanced' re-weights the minority (Denied) class, since this dataset
        # is skewed ~69% Approved / 31% Denied.
        'class_weight': [None, 'balanced']
    }

    tree = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=tree,
        param_grid=param_grid,
        cv=kfold,
        # Macro-averaged F1 weighs the Denied class as heavily as the Approved
        # class, instead of optimizing only for the majority class.
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    print("[DecisionTree] Fitting GridSearchCV (finding best criterion, depth, and split settings)...")
    grid_search.fit(X_train_encoded, y_train)

    best_model = grid_search.best_estimator_
    print(f"[DecisionTree] Best Hyperparameters: {grid_search.best_params_}")
    print(f"[DecisionTree] Best CV Score (macro F1): {grid_search.best_score_:.4f}")

    cv_results = [
        {
            'params': params,
            'mean_test_score': float(mean_score),
            'std_test_score': float(std_score),
            'rank': int(rank)
        }
        for params, mean_score, std_score, rank in zip(
            grid_search.cv_results_['params'],
            grid_search.cv_results_['mean_test_score'],
            grid_search.cv_results_['std_test_score'],
            grid_search.cv_results_['rank_test_score']
        )
    ]
    cv_results.sort(key=lambda result: result['rank'])

    return best_model, grid_search.best_params_, cv_results
