from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


RANDOM_STATE = 42


def train_svm(X_train, y_train):
    print(
        "[SVM] Configuring Stratified K-Fold "
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
            "svm",
            SVC(
                probability=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    param_grid = [
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1.0, 10.0],
        },
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1.0, 10.0],
            "svm__gamma": ["scale", "auto"],
        },
    ]

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    print(
        "[SVM] Fitting GridSearchCV "
        "(finding best hyperparameters)..."
    )

    grid_search.fit(X_train, y_train)

    print(
        f"[SVM] Best Hyperparameters: "
        f"{grid_search.best_params_}"
    )

    print(
        f"[SVM] Best CV Score (F1): "
        f"{grid_search.best_score_:.4f}"
    )

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
    )
