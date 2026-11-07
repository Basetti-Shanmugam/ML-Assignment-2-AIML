"""Dataset loading, schema standardisation and validation.

Two interchangeable sources produce the SAME standardised DataFrame:
  * Option A - UCI repository via ucimlrepo (needs internet)
  * Option B - Kaggle / Koklu download (Excel or CSV on disk)
Everything downstream only sees the standardised schema.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config

# Different sources spell some column names differently (e.g. the Koklu Excel
# file has 'AspectRation' and 'roundness'; the UCI listing has 'AspectRatio'
# and 'Roundness'). Matching is case-insensitive via this alias map.
_ALIASES = {
    "aspectration": "AspectRatio",   # typo present in the original Excel/ARFF
    "aspectratio": "AspectRatio",
    "roundness": "Roundness",
}
_CANON_LOOKUP = {c.lower(): c for c in config.FEATURE_COLS + [config.TARGET_COL]}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema. Only names change - never values."""
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _ALIASES:
            rename[col] = _ALIASES[key]
        elif key in _CANON_LOOKUP:
            rename[col] = _CANON_LOOKUP[key]
    return df.rename(columns=rename)


# ----------------------------------------------------------------- Option A
def load_from_uci(dataset_id: int = config.UCI_DATASET_ID) -> pd.DataFrame:
    """Download Dry Bean from the UCI repository using ucimlrepo (needs internet)."""
    from ucimlrepo import fetch_ucirepo  # imported lazily: not needed for Option B

    dataset = fetch_ucirepo(id=dataset_id)
    features = dataset.data.features
    targets = dataset.data.targets
    df = pd.concat([features, targets], axis=1)
    # ucimlrepo names the target column 'Class'; standardise anyway.
    return standardize_columns(df)


# ----------------------------------------------------------------- Option B
def load_from_file(path: str | Path = config.RAW_EXCEL,
                   target_col: str = config.TARGET_COL) -> pd.DataFrame:
    """Load the Kaggle/Koklu Excel or CSV file and standardise it.

    `target_col` lets you name the target if your file uses a different one;
    it is renamed to 'Class'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset from {config.KAGGLE_URL} "
            f"(or {config.UCI_URL}) and place the file in {config.RAW_DIR}.")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    df = standardize_columns(df)
    if target_col != config.TARGET_COL:
        df = df.rename(columns={target_col: config.TARGET_COL})
    return df


# --------------------------------------------------------------- validation
def validate_dataset(df: pd.DataFrame) -> dict:
    """Return a validation report. Raises ValueError if the schema is wrong."""
    missing_cols = [c for c in config.FEATURE_COLS + [config.TARGET_COL] if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in config.FEATURE_COLS + [config.TARGET_COL]]
    if missing_cols or extra_cols:
        raise ValueError(f"Schema mismatch. Missing: {missing_cols}; unexpected: {extra_cols}")
    return {
        "shape": df.shape,
        "n_features": len(config.FEATURE_COLS),
        "target": config.TARGET_COL,
        "n_classes": int(df[config.TARGET_COL].nunique()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": df[config.TARGET_COL].value_counts().to_dict(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


def schemas_match(df_a: pd.DataFrame, df_b: pd.DataFrame) -> bool:
    """True if two sources give the same columns (same order) and the same
    numeric/non-numeric kind for every column, after standardisation."""
    cols = config.FEATURE_COLS + [config.TARGET_COL]
    a, b = standardize_columns(df_a), standardize_columns(df_b)
    if not (set(cols) <= set(a.columns) and set(cols) <= set(b.columns)):
        return False
    return all(
        pd.api.types.is_numeric_dtype(a[c]) == pd.api.types.is_numeric_dtype(b[c])
        for c in cols)


def save_raw_csv(df: pd.DataFrame, path: str | Path = config.RAW_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------- uploaded-CSV validation (app)
def validate_uploaded_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate a user-uploaded test CSV. Returns (clean_df, errors).

    Never modifies values: only standardises column names, and checks that the
    features are numeric, non-missing, and the target labels are known classes.
    """
    errors: list[str] = []
    if df is None or df.empty:
        return df, ["The uploaded file is empty."]
    df = standardize_columns(df)
    expected = config.FEATURE_COLS + [config.TARGET_COL]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
    if extra:
        errors.append(f"Unexpected column(s): {', '.join(map(str, extra))}")
    if errors:
        return df, errors
    for c in config.FEATURE_COLS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            bad = pd.to_numeric(df[c], errors="coerce").isna().sum()
            errors.append(f"Column '{c}' must be numeric ({bad} non-numeric value(s)).")
    if errors:
        return df, errors
    n_missing = int(df[expected].isna().sum().sum())
    if n_missing:
        errors.append(f"The file contains {n_missing} missing value(s); please remove or fix them.")
    return df[expected], errors
