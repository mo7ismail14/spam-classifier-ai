"""Data loading, cleaning, and feature encoding for the loan approval dataset."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Continuous numeric features (median imputation; Decision Trees don't need scaling).
NUMERIC_FEATURES: list[str] = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Binary numeric feature stored as 0/1 but semantically categorical -> most-frequent imputation.
BINARY_FEATURES: list[str] = ['Credit_History']

# Nominal categorical features -> most-frequent imputation + ordinal encoding.
CATEGORICAL_FEATURES: list[str] = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# Column order after the ColumnTransformer runs (must match the transformer list below).
ENCODED_FEATURE_ORDER: list[str] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

ALL_INPUT_FEATURES: list[str] = ENCODED_FEATURE_ORDER

TARGET_COLUMN: str = 'Loan_Status'

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    'ApplicantIncome': 'Applicant Income',
    'CoapplicantIncome': 'Co-applicant Income',
    'LoanAmount': 'Loan Amount',
    'Loan_Amount_Term': 'Loan Term (months)',
    'Credit_History': 'Credit History',
    'Gender': 'Gender',
    'Married': 'Marital Status',
    'Dependents': 'Number of Dependents',
    'Education': 'Education',
    'Self_Employed': 'Self-Employment Status',
    'Property_Area': 'Property Area',
}


class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """Clips numeric values to [Q1 - k*IQR, Q3 + k*IQR], learned from the fitted (training) data.

    Capping instead of dropping rows preserves this already-small (~600 row) dataset.
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.iqr_multiplier * iqr
        self.upper_bounds_ = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def load_loan_dataset(csv_path: str) -> pd.DataFrame:
    """Read the raw loan application CSV into a DataFrame."""
    print(f"Reading dataset from: {csv_path}")
    return pd.read_csv(csv_path)


def clean_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns, exact duplicate rows, and rows with a missing target label.

    Outlier handling happens later, inside build_preprocessor() (IQROutlierCapper), so it
    is fit only on the training split and applied consistently at inference time.
    """
    df = df.copy()
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    before = len(df)
    df = df.drop_duplicates()
    duplicates_dropped = before - len(df)
    if duplicates_dropped:
        print(f"Dropped {duplicates_dropped} duplicate rows ({before} -> {len(df)}).")

    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])
    return df.reset_index(drop=True)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate raw input features from the encoded (0/1) target column."""
    X = df[ALL_INPUT_FEATURES].copy()
    y = df[TARGET_COLUMN].apply(lambda status: 1 if str(status).strip().upper() == 'Y' else 0)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer that imputes missing values and encodes categoricals.

    Output column order matches ENCODED_FEATURE_ORDER, which the rule-extraction
    module relies on to map a Decision Tree split back to a named feature.
    """
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_capper', IQROutlierCapper())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    return ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, NUMERIC_FEATURES),
        ('binary', SimpleImputer(strategy='most_frequent'), BINARY_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def get_categorical_encoder(preprocessor: ColumnTransformer) -> OrdinalEncoder:
    """Fetch the fitted OrdinalEncoder from within the preprocessor (used to decode categories)."""
    return preprocessor.named_transformers_['categorical'].named_steps['encoder']
