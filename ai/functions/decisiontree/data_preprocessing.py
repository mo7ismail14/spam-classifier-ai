"""Data loading, cleaning, and feature encoding for the loan approval dataset."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

# Continuous numeric features (median imputation; Decision Trees don't need scaling).
NUMERIC_FEATURES: list[str] = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Binary numeric feature stored as 0/1 but semantically categorical -> most-frequent imputation.
BINARY_FEATURES: list[str] = ['Credit_History']

# Nominal categorical features -> most-frequent imputation + ordinal encoding.
CATEGORICAL_FEATURES: list[str] = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# The 11 fields an applicant/the frontend form actually supplies. This never changes,
# regardless of USE_ENGINEERED_FEATURES, since engineered columns are derived, not input.
ALL_INPUT_FEATURES: list[str] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Derived numeric features, computed from the raw inputs above (see add_engineered_features).
ENGINEERED_FEATURES: list[str] = ['TotalIncome', 'EMI', 'IncomeToLoan', 'BalanceIncome']

# Toggle: when True, build_preprocessor() prepends an engineering step and the tree is
# trained on NUMERIC_FEATURES + ENGINEERED_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES.
# Flip only after comparing CV macro F1 with/without (see training/train_decision_tree.py).
USE_ENGINEERED_FEATURES: bool = False

# Column order after the preprocessor runs (must match build_preprocessor()'s transformer
# list below). This is what rule_extraction.py and feature_importances are indexed by —
# NOT what the API/frontend form exposes (that's always ALL_INPUT_FEATURES).
if USE_ENGINEERED_FEATURES:
    ENCODED_FEATURE_ORDER: list[str] = NUMERIC_FEATURES + ENGINEERED_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
else:
    ENCODED_FEATURE_ORDER: list[str] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

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
    'TotalIncome': 'Total Household Income',
    'EMI': 'Estimated Monthly Installment',
    'IncomeToLoan': 'Income-to-Loan Ratio',
    'BalanceIncome': 'Income Remaining After EMI',
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
    """Separate raw input features (always just the 11 user-supplied fields, regardless
    of USE_ENGINEERED_FEATURES) from the encoded (0/1) target column."""
    X = df[ALL_INPUT_FEATURES].copy()
    y = df[TARGET_COLUMN].apply(lambda status: 1 if str(status).strip().upper() == 'Y' else 0)
    return X, y


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure function: derive numeric features from the raw loan fields.

    TotalIncome = ApplicantIncome + CoapplicantIncome
    EMI = LoanAmount / Loan_Amount_Term
    IncomeToLoan = TotalIncome / LoanAmount
    BalanceIncome = TotalIncome - EMI * 1000

    Division by zero produces inf, which is converted to NaN so the downstream
    imputer fills it like any other missing value.
    """
    df = df.copy()
    applicant_income = pd.to_numeric(df['ApplicantIncome'], errors='coerce')
    coapplicant_income = pd.to_numeric(df['CoapplicantIncome'], errors='coerce')
    loan_amount = pd.to_numeric(df['LoanAmount'], errors='coerce')
    loan_term = pd.to_numeric(df['Loan_Amount_Term'], errors='coerce')

    total_income = applicant_income + coapplicant_income
    emi = (loan_amount / loan_term).replace([np.inf, -np.inf], np.nan)
    income_to_loan = (total_income / loan_amount).replace([np.inf, -np.inf], np.nan)
    balance_income = total_income - emi * 1000

    df['TotalIncome'] = total_income
    df['EMI'] = emi
    df['IncomeToLoan'] = income_to_loan
    df['BalanceIncome'] = balance_income
    return df


def build_preprocessor():
    """Build the preprocessor that imputes missing values, caps outliers, and encodes
    categoricals. When USE_ENGINEERED_FEATURES is True, a FunctionTransformer step runs
    first to derive TotalIncome/EMI/IncomeToLoan/BalanceIncome from the raw inputs.

    Output column order matches ENCODED_FEATURE_ORDER, which the rule-extraction
    module and feature_importances rely on to map a Decision Tree split back to a name.
    """
    numeric_columns = NUMERIC_FEATURES + ENGINEERED_FEATURES if USE_ENGINEERED_FEATURES else NUMERIC_FEATURES

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_capper', IQROutlierCapper())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, numeric_columns),
        ('binary', SimpleImputer(strategy='most_frequent'), BINARY_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    if not USE_ENGINEERED_FEATURES:
        return column_transformer

    return Pipeline(steps=[
        ('engineer', FunctionTransformer(add_engineered_features, validate=False)),
        ('columns', column_transformer)
    ])


def get_categorical_encoder(preprocessor) -> OrdinalEncoder:
    """Fetch the fitted OrdinalEncoder from within the preprocessor (used to decode
    categories). Handles both the bare ColumnTransformer and the engineered-features
    Pipeline wrapping it."""
    column_transformer = preprocessor.named_steps['columns'] if isinstance(preprocessor, Pipeline) else preprocessor
    return column_transformer.named_transformers_['categorical'].named_steps['encoder']
