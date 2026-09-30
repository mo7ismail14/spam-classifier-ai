from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


RANDOM_STATE = 42


def train_svm(X_train, y_train):
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
                SVC(
                    probability=True,
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

    param_grid = [
        {
            "model__kernel": ["linear"],
            "model__C": [0.1, 1.0, 10.0],
        },
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto"],
        },
    ]

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    print("[SVM] Training...")
    search.fit(X_train, y_train)

    print(f"[SVM] Best parameters: {search.best_params_}")
    print(f"[SVM] Best CV F1: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_
