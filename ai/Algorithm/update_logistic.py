from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def train_logistic_regression(X_train, y_train):
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=4000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__solver": ["lbfgs", "liblinear"],
    }

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    print("[LogisticRegression] Training...")
    search.fit(X_train, y_train)

    print(
        f"[LogisticRegression] "
        f"Best parameters: {search.best_params_}"
    )

    print(
        f"[LogisticRegression] "
        f"Best CV F1: {search.best_score_:.4f}"
    )

    return search.best_estimator_, search.best_params_
