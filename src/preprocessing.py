"""Preprocessing: cleaning, label encoding and the single shared train/test split.

Leakage rules followed here
  * The split happens BEFORE any scaler is fitted.
  * StandardScaler lives inside each sklearn Pipeline, so it is fitted on the
    training fold only (and on training folds only during cross-validation).
  * Exact duplicate rows are removed BEFORE splitting so an identical bean can
    never appear in both train and test.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Handle missing values and duplicates. Returns (clean_df, log).

    Nothing is imputed or invented: missing values -> rows dropped (there are
    none in Dry Bean); exact duplicate rows -> dropped, keeping the first copy.
    """
    log = {"rows_before": len(df)}
    log["missing_cells"] = int(df.isna().sum().sum())
    if log["missing_cells"]:
        df = df.dropna()
    log["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log["rows_after"] = len(df)
    return df, log


def encode_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """Map class names to integers 0..K-1 (alphabetical order)."""
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index, name=config.TARGET_COL)
    return y_enc, le


def split_data(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
               random_state: int = config.RANDOM_STATE):
    """80:20 stratified split with a fixed seed. Returns a dict of everything
    the other modules need. The same call always gives the same split."""
    X = df[config.FEATURE_COLS]
    y_enc, le = encode_target(df[config.TARGET_COL])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, stratify=y_enc, random_state=random_state)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train,
            "y_test": y_test, "label_encoder": le}


def save_splits(split: dict) -> None:
    """Write train/test CSVs (with the human-readable Class) for transparency.
    data/test_data.csv is the held-out test set that the Streamlit app can use."""
    le = split["label_encoder"]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, X, y in (("train", split["X_train"], split["y_train"]),
                       ("test", split["X_test"], split["y_test"])):
        out = X.copy()
        out[config.TARGET_COL] = le.inverse_transform(y)
        out.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
        if name == "test":
            out.to_csv(config.SAMPLE_TEST_CSV, index=False)
