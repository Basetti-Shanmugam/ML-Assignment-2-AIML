"""Build, train and save the five classification pipelines.

Run:  python -m src.train_models
"""
from __future__ import annotations

import json
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from src import data_loader, preprocessing


def build_pipelines() -> dict[str, Pipeline]:
    """One Pipeline per algorithm. Every pipeline = StandardScaler + classifier
    (the scaler is kept even for the tree/NB so all five share the same
    preprocessing code path; it does not change tree/NB results materially)."""
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=rs))]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=rs))]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=rs))]),
    }


def train_all(X_train, y_train) -> tuple[dict[str, Pipeline], dict[str, float]]:
    pipes, seconds = build_pipelines(), {}
    for name, pipe in pipes.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)          # TRAIN data only
        seconds[name] = round(time.time() - t0, 2)
        print(f"  trained {name:<20s} in {seconds[name]:>6.2f}s")
    return pipes, seconds


def save_all(pipes: dict[str, Pipeline], split: dict, clean_log: dict, train_seconds: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, pipe in pipes.items():
        joblib.dump(pipe, config.MODELS_DIR / config.MODEL_FILES[name])
    le = split["label_encoder"]
    joblib.dump(le, config.LABEL_ENCODER_PATH)
    import sklearn
    meta = {
        "dataset": "UCI Dry Bean (ID 602)",
        "features": config.FEATURE_COLS,
        "target": config.TARGET_COL,
        "classes": list(le.classes_),
        "n_features": len(config.FEATURE_COLS),
        "n_classes": len(le.classes_),
        "n_instances_raw": clean_log["rows_before"],
        "n_instances_after_cleaning": clean_log["rows_after"],
        "duplicates_removed": clean_log["duplicates_removed"],
        "n_train": len(split["X_train"]),
        "n_test": len(split["X_test"]),
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "sklearn_version": sklearn.__version__,
        "train_seconds": train_seconds,
    }
    config.METADATA_PATH.write_text(json.dumps(meta, indent=2))


def main() -> None:
    print("Loading data ...")
    # Option B (local file) is used for the CLI. For Option A use data_loader.load_from_uci().
    df = data_loader.load_from_file() if config.RAW_EXCEL.exists() else data_loader.load_from_uci()
    data_loader.validate_dataset(df)
    data_loader.save_raw_csv(df)
    df, log = preprocessing.clean_dataset(df)
    df.to_csv(config.CLEAN_CSV, index=False)
    print(f"  cleaning log: {log}")
    split = preprocessing.split_data(df)
    preprocessing.save_splits(split)
    print(f"  train={len(split['X_train'])}  test={len(split['X_test'])}")
    print("Training ...")
    pipes, secs = train_all(split["X_train"], split["y_train"])
    save_all(pipes, split, log, secs)
    print("Saved models to", config.MODELS_DIR)


if __name__ == "__main__":
    main()
