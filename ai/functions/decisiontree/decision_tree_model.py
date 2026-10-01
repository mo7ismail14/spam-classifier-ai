"""Decision Tree training for loan approval, tuned via GridSearchCV + Stratified K-Fold CV.

The preprocessor is fit INSIDE the pipeline that GridSearchCV cross-validates, so
imputation medians, outlier bounds, and categorical encodings are learned fresh on
each training fold and never see the held-out validation fold's data.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def _strip_tree_prefix(params: dict) -> dict:
    """Pipeline step params are named 'tree__max_depth' etc.; strip that for external callers."""
    return {key.removeprefix('tree__'): value for key, value in params.items()}


def train_decision_tree(preprocessor, X_train_raw, y_train):
    """Grid-search a DecisionTreeClassifier's hyperparameters inside a
    Pipeline([preprocess, tree]).

    Returns (best_pipeline, best_params, cv_results). best_params and every
    cv_results entry's params have the 'tree__' prefix stripped, so the API/frontend
    format is unchanged from before this pipeline refactor.
    """
    print("[DecisionTree] Configuring Stratified K-Fold Cross-Validation & Hyperparameter Grid...")
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('tree', DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        'tree__criterion': ['gini', 'entropy'],
        'tree__max_depth': [3, 4, 5, 6, 8, None],
        'tree__min_samples_split': [2, 5, 10],
        'tree__min_samples_leaf': [1, 2, 4],
        # 'balanced' re-weights the minority (Denied) class, since this dataset
        # is skewed ~69% Approved / 31% Denied.
        'tree__class_weight': [None, 'balanced'],
        # Cost-complexity pruning: larger alpha prunes more aggressively, which can
        # reduce overfitting to the majority class and cut false positives.
        'tree__ccp_alpha': [0.0, 0.001, 0.005, 0.01]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skfold,
        # Macro-averaged F1 weighs the Denied class as heavily as the Approved
        # class, instead of optimizing only for the majority class.
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    print("[DecisionTree] Fitting GridSearchCV (criterion, depth, split settings, pruning)...")
    grid_search.fit(X_train_raw, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = _strip_tree_prefix(grid_search.best_params_)
    print(f"[DecisionTree] Best Hyperparameters: {best_params}")
    print(f"[DecisionTree] Best CV Score (macro F1): {grid_search.best_score_:.4f}")

    cv_results = [
        {
            'params': _strip_tree_prefix(params),
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

    return best_pipeline, best_params, cv_results
