from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold


def train_logistic_regression(X_train_vec, y_train):
    print("[LogisticRegression] Configuring K-Fold Cross-Validation & Hyperparameter Grid...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear']
    }

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        cv=kfold,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    print("[LogisticRegression] Fitting GridSearchCV (finding best C and solver)...")
    grid_search.fit(X_train_vec, y_train)

    best_model = grid_search.best_estimator_
    print(f"[LogisticRegression] Best Hyperparameters: {grid_search.best_params_}")
    print(f"[LogisticRegression] Best CV Score (F1): {grid_search.best_score_:.4f}")
    return best_model, grid_search.best_params_
