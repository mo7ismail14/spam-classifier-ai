import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib

# ai/ is the project root for sibling packages (Algorithm/); this script lives in ai/training/.
AI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)

from Algorithm.svm import train_svm
from Algorithm.logistic_regression import train_logistic_regression

CSV_PATH = os.path.join(AI_DIR, 'dataset', 'spam_Emails_data.csv')
MODEL_DIR = os.path.join(AI_DIR, 'model')
SAMPLE_N_PER_CLASS = 1200


def load_data(csv_path):
    print(f"Reading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def clean_data(df):
    df['text'] = df['text'].fillna('').astype(str).str.strip()
    df = df.dropna(subset=['label'])
    df = df[df['text'].str.len() > 10]
    return df


def deduplicate(df):
    before = len(df)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows ({before} -> {len(df)}).")
    return df


def balanced_sample(df, sample_n_per_class):
    hams = df[df['label'].astype(str).str.lower().str.strip() == 'ham']
    spams = df[df['label'].astype(str).str.lower().str.strip() == 'spam']
    sample_n = min(sample_n_per_class, len(hams), len(spams))
    df_sample = pd.concat([
        hams.sample(sample_n, random_state=42),
        spams.sample(sample_n, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_sample


def split_data(df_sample):
    X = df_sample['text']
    y = df_sample['label'].apply(lambda x: 1 if str(x).lower().strip() == 'spam' else 0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    print("Converting text using TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=4000,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def evaluate_model(best_model, X_test_vec, y_test, best_params):
    print("Calculating Confusion Matrix on Test Set...")
    y_pred = best_model.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'best_params': best_params
    }
    print(f"Metrics: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return metrics


def prepare_dropdown_samples(X_test, y_test, n_per_class=100):
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})
    hams = test_df[test_df['label'] == 0]
    spams = test_df[test_df['label'] == 1]

    n = min(n_per_class, len(hams), len(spams))
    hams = hams.sample(n, random_state=42)
    spams = spams.sample(n, random_state=42)

    dropdown_samples = []
    for _, row in pd.concat([hams, spams]).sample(frac=1, random_state=42).iterrows():
        dropdown_samples.append({
            'Subject': '',
            'Message': str(row['text']),
            'Spam/Ham': 'spam' if row['label'] == 1 else 'ham'
        })
    return dropdown_samples


def save_model_artifacts(model_dir, model, vectorizer, metrics):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
    joblib.dump(metrics, os.path.join(model_dir, 'metrics.pkl'))
    print(f"Saved model, vectorizer, and metrics to {model_dir}")


def save_shared_samples(model_dir, dropdown_samples):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(dropdown_samples, os.path.join(model_dir, 'sample_emails.pkl'))
    print(f"Saved shared sample emails to {model_dir}")


def main():
    print("Loading Spam/Ham email dataset...")
    df = load_data(CSV_PATH)
    df = clean_data(df)
    df = deduplicate(df)

    df_sample = balanced_sample(df, SAMPLE_N_PER_CLASS)
    X_train, X_test, y_train, y_test = split_data(df_sample)

    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    # SVM Training
    svm_model, svm_params = train_svm(X_train_vec, y_train)
    svm_metrics = evaluate_model(svm_model, X_test_vec, y_test, svm_params)
    save_model_artifacts(os.path.join(MODEL_DIR, 'SVM'), svm_model, vectorizer, svm_metrics)

    # Logistic Regression Training
    logreg_model, logreg_params = train_logistic_regression(X_train_vec, y_train)
    logreg_metrics = evaluate_model(logreg_model, X_test_vec, y_test, logreg_params)
    save_model_artifacts(os.path.join(MODEL_DIR, 'LogisticRegression'), logreg_model, vectorizer, logreg_metrics)

    # Save shared sample emails
    dropdown_samples = prepare_dropdown_samples(X_test, y_test)
    save_shared_samples(MODEL_DIR, dropdown_samples)

    print("Training and model serialization successfully completed!")


if __name__ == '__main__':
    main()
