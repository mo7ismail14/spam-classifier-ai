from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from Algorithm.svm import train_svm
from Algorithm.logistic_regression import train_logistic_regression


RANDOM_STATE = 42
SAMPLE_N_PER_CLASS = 1200

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "spam_Emails_data.csv"
MODEL_DIR = BASE_DIR / "model"


def load_data(path):
    print(f"Reading dataset from: {path}")
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df.dropna(subset=["label"])

    return df[df["text"].str.len() > 10]


def deduplicate(df):
    before = len(df)

    df = (
        df.drop_duplicates(subset="text")
        .reset_index(drop=True)
    )

    print(
        f"Dropped {before - len(df)} duplicate rows "
        f"({before} → {len(df)})."
    )

    return df


def balanced_sample(df, n_per_class):
    sample_n = min(
        n_per_class,
        df["label"].value_counts().min(),
    )

    return (
        df.groupby("label", group_keys=False)
        .sample(
            n=sample_n,
            random_state=RANDOM_STATE,
        )
        .sample(
            frac=1,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )


def split_data(df):
    X = df["text"]
    y = df["label"].eq("spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Dataset prepared: "
        f"{len(X_train)} training samples, "
        f"{len(X_test)} testing samples."
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, best_params):
    print("Calculating Confusion Matrix on Test Set...")

    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
    ).ravel()

    metrics = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "best_params": best_params,
    }

    print(
        f"Metrics: TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )

    return metrics


def prepare_dropdown_samples(
    X_test,
    y_test,
    n_per_class=100,
):
    test_df = pd.DataFrame(
        {
            "text": X_test,
            "label": y_test,
        }
    )

    samples = (
        test_df
        .groupby("label", group_keys=False)
        .sample(
            n=min(
                n_per_class,
                test_df["label"].value_counts().min(),
            ),
            random_state=RANDOM_STATE,
        )
        .sample(
            frac=1,
            random_state=RANDOM_STATE,
        )
    )

    return [
        {
            "Subject": "",
            "Message": row.text,
            "Spam/Ham": "spam" if row.label else "ham",
        }
        for row in samples.itertuples()
    ]


def save_model_artifacts(model_dir, model, metrics):
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_dir / "model.pkl",
    )

    joblib.dump(
        metrics,
        model_dir / "metrics.pkl",
    )

    print(f"Saved model and metrics to {model_dir}")


def save_shared_samples(model_dir, samples):
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        samples,
        model_dir / "sample_emails.pkl",
    )

    print(f"Saved sample emails to {model_dir}")


def main():
    print("Loading Spam/Ham email dataset...")

    df = load_data(CSV_PATH)
    df = clean_data(df)
    df = deduplicate(df)

    df_sample = balanced_sample(
        df,
        SAMPLE_N_PER_CLASS,
    )

    X_train, X_test, y_train, y_test = split_data(
        df_sample
    )

    # SVM
    print("\n--- SVM Training ---")

    svm_model, svm_params = train_svm(
        X_train,
        y_train,
    )

    svm_metrics = evaluate_model(
        svm_model,
        X_test,
        y_test,
        svm_params,
    )

    save_model_artifacts(
        MODEL_DIR / "SVM",
        svm_model,
        svm_metrics,
    )

    # Logistic Regression
    print("\n--- Logistic Regression Training ---")

    logreg_model, logreg_params = train_logistic_regression(
        X_train,
        y_train,
    )

    logreg_metrics = evaluate_model(
        logreg_model,
        X_test,
        y_test,
        logreg_params,
    )

    save_model_artifacts(
        MODEL_DIR / "LogisticRegression",
        logreg_model,
        logreg_metrics,
    )

    # Sample emails
    samples = prepare_dropdown_samples(
        X_test,
        y_test,
    )

    save_shared_samples(
        MODEL_DIR,
        samples,
    )

    print(
        "\nTraining and model serialization "
        "completed successfully!"
    )


if __name__ == "__main__":
    main()
