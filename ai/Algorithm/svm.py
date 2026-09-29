from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold


def train_svm(X_train_vec, y_train):
    print("[SVM] Configuring K-Fold Cross-Validation & Hyperparameter Grid...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=kfold,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    print("[SVM] Fitting GridSearchCV (finding best C, kernel, and gamma)...")
    grid_search.fit(X_train_vec, y_train)

    best_model = grid_search.best_estimator_
    print(f"[SVM] Best Hyperparameters: {grid_search.best_params_}")
    print(f"[SVM] Best CV Score (F1): {grid_search.best_score_:.4f}")
    return best_model, grid_search.best_params_
