from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns expected in IBM Telco churn dataset
CATEGORICAL_COLS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_COLS: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def validate_telco_columns(df: pd.DataFrame, target_col: str) -> None:
    required = set(CATEGORICAL_COLS + NUMERIC_COLS + [target_col, "customerID"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def clean_telco_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Coerce types, handle blanks, and normalize labels."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[target_col] = df[target_col].str.strip().map({"Yes": 1, "No": 0})
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Fill missing values conservatively
    for col in NUMERIC_COLS:
        if col in df:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df:
            df[col] = df[col].fillna("Unknown")
    return df


def build_preprocessor(
    categorical_cols: List[str] = CATEGORICAL_COLS,
    numeric_cols: List[str] = NUMERIC_COLS,
) -> ColumnTransformer:
    """Create a sklearn ColumnTransformer for mixed-type preprocessing."""
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", cat_pipeline, categorical_cols),
            ("numerical", num_pipeline, numeric_cols),
        ]
    )


def fit_preprocessor(
    preprocessor: ColumnTransformer, df: pd.DataFrame, target_col: str, id_col: str
) -> ColumnTransformer:
    feature_df = df.drop(columns=[target_col, id_col])
    preprocessor.fit(feature_df)
    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply fitted preprocessor and return features + labels."""
    feature_df = df.drop(columns=[target_col, id_col])
    X = preprocessor.transform(feature_df)
    y = df[target_col].values.astype(np.float32)
    return X, y

