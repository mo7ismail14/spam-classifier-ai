from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def train_logistic_regression(X_train, y_train):
    print(
        "[LogisticRegression] Configuring Stratified K-Fold "
        "Cross-Validation & Pipeline..."
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=4000,
                ngram_range=(1, 2),
            ),
        ),
        (
            "logreg",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "logreg__C": [0.01, 0.1, 1.0, 10.0],
        "logreg__solver": ["lbfgs", "liblinear"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    print(
        "[LogisticRegression] Fitting GridSearchCV "
        "(finding best C and solver)..."
    )

    grid_search.fit(X_train, y_train)

    print(
        f"[LogisticRegression] Best Hyperparameters: "
        f"{grid_search.best_params_}"
    )

    print(
        f"[LogisticRegression] Best CV Score (F1): "
        f"{grid_search.best_score_:.4f}"
    )

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
    )
