from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold


def train_random_forest(X_train_vec, y_train):
    print("[RandomForest] Configuring K-Fold Cross-Validation & Hyperparameter Grid...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=kfold,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("[RandomForest] Fitting GridSearchCV (finding best n_estimators, max_depth, min_samples_split)...")
    grid_search.fit(X_train_vec, y_train)

    print("[RandomForest] Best Parameters: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    print(f"[RandomForest] Best Hyperparameters: {grid_search.best_params_}")
    print(f"[RandomForest] Best CV Score (F1): {grid_search.best_score_:.4f}")

    return best_model , grid_search.best_params_