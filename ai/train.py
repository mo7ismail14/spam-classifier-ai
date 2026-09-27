import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import confusion_matrix
import joblib

CSV_PATH = os.path.join(os.path.dirname(__file__), 'spam_Emails_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
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


def train_model(X_train_vec, y_train):
    print("Configuring K-Fold Cross-Validation & Hyperparameter Grid...")
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

    print("Fitting GridSearchCV (finding best C, kernel, and gamma)...")
    grid_search.fit(X_train_vec, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best CV Score (F1): {grid_search.best_score_:.4f}")
    return best_model, grid_search.best_params_


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


def save_artifacts(model_dir, best_model, vectorizer, metrics, dropdown_samples):
    os.makedirs(model_dir, exist_ok=True)
    print("Saving best model, vectorizer, metrics, and email samples...")
    joblib.dump(best_model, os.path.join(model_dir, 'svm_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
    joblib.dump(metrics, os.path.join(model_dir, 'metrics.pkl'))
    joblib.dump(dropdown_samples, os.path.join(model_dir, 'sample_emails.pkl'))


def main():
    print("Loading Spam/Ham email dataset...")
    df = load_data(CSV_PATH)
    df = clean_data(df)
    df = deduplicate(df)

    df_sample = balanced_sample(df, SAMPLE_N_PER_CLASS)
    X_train, X_test, y_train, y_test = split_data(df_sample)

    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    best_model, best_params = train_model(X_train_vec, y_train)
    metrics = evaluate_model(best_model, X_test_vec, y_test, best_params)

    dropdown_samples = prepare_dropdown_samples(X_test, y_test)
    save_artifacts(MODEL_DIR, best_model, vectorizer, metrics, dropdown_samples)

    print("Training and model serialization successfully completed!")


if __name__ == '__main__':
    main()
