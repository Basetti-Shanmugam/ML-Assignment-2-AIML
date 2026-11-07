"""Metrics, reports and figures for the five saved pipelines.

Run:  python -m src.evaluate_models
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize

import config

METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# ------------------------------------------------------------------ metrics
def safe_macro_auc(y_true, proba, n_classes: int) -> float:
    """Macro-averaged one-vs-rest ROC AUC.

    Identical to roc_auc_score(..., multi_class='ovr', average='macro') when every
    class occurs in y_true. If an uploaded test file lacks some classes (AUC is
    undefined for them) those classes are skipped; returns NaN if < 2 classes remain.
    """
    y_true = np.asarray(y_true)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    aucs = []
    for k in range(n_classes):
        pos = y_bin[:, k].sum()
        if pos == 0 or pos == len(y_bin):
            continue
        aucs.append(roc_auc_score(y_bin[:, k], proba[:, k]))
    return float(np.mean(aucs)) if len(aucs) >= 2 else float("nan")


def compute_metrics(y_true, y_pred, proba, n_classes: int) -> dict:
    """The six mandatory metrics (macro averaging for multiclass).
    zero_division=0 handles classes that are never predicted."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": safe_macro_auc(y_true, proba, n_classes),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def load_models() -> tuple[dict, object, dict]:
    """Load the five saved pipelines, the label encoder and the metadata."""
    pipes = {name: joblib.load(config.MODELS_DIR / fname) for name, fname in config.MODEL_FILES.items()}
    le = joblib.load(config.LABEL_ENCODER_PATH)
    meta = json.loads(config.METADATA_PATH.read_text())
    return pipes, le, meta


def evaluate_all(pipes: dict, X, y_true, n_classes: int) -> tuple[pd.DataFrame, dict]:
    """Return (comparison_df, per-model predictions/probabilities)."""
    rows, store = [], {}
    for name, pipe in pipes.items():
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X)
        m = compute_metrics(y_true, pred, proba, n_classes)
        rows.append({"ML Model Name": name, **m})
        store[name] = {"pred": pred, "proba": proba}
    return pd.DataFrame(rows), store


# --------------------------------------------------------------------- CLI
def main() -> None:
    import matplotlib.pyplot as plt
    from src import visualization as viz

    pipes, le, meta = load_models()
    classes = list(le.classes_)
    k = len(classes)

    # Rebuild the exact held-out split from the saved files.
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    X_train, y_train = train[config.FEATURE_COLS], le.transform(train[config.TARGET_COL])
    X_test, y_test = test[config.FEATURE_COLS], le.transform(test[config.TARGET_COL])

    comparison, store = evaluate_all(pipes, X_test, y_test, k)
    cross_check = {}

    # Extra generalisation diagnostics (training data only; CV never touches the test set)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    details = []
    for name, pipe in pipes.items():
        train_acc = accuracy_score(y_train, pipe.predict(X_train))
        cv_pipe = __import__("sklearn.base", fromlist=["clone"]).clone(pipe)
        if name == "SVM":                      # skip the slow internal Platt CV; accuracy is unaffected
            cv_pipe.set_params(clf__probability=False)
        cv = cross_val_score(cv_pipe, X_train, y_train, cv=skf, scoring="accuracy")
        test_acc = float(comparison.loc[comparison["ML Model Name"] == name, "Accuracy"].iloc[0])
        details.append({"ML Model Name": name, "Train Accuracy": train_acc,
                        "CV Accuracy (5-fold, train only)": cv.mean(), "CV Std": cv.std(),
                        "Test Accuracy": test_acc, "Train-Test Gap": train_acc - test_acc})
        # sanity: our AUC must equal sklearn's built-in OvR macro AUC
        cross_check[name] = abs(roc_auc_score(y_test, store[name]["proba"], multi_class="ovr",
                                              average="macro") - comparison.set_index("ML Model Name").loc[name, "AUC"])
    details = pd.DataFrame(details)
    assert max(cross_check.values()) < 1e-12, f"AUC cross-check failed: {cross_check}"

    comparison.round(4).to_csv(config.COMPARISON_CSV, index=False)
    details.round(4).to_csv(config.DETAILS_CSV, index=False)

    # Classification reports + confusion matrices
    config.CLASS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in pipes:
        slug = config.MODEL_FILES[name].replace(".joblib", "")
        pred = store[name]["pred"]
        rep = classification_report(y_test, pred, target_names=classes, output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.round(4).to_csv(config.CLASS_REPORTS_DIR / f"{slug}_report.csv")
        (config.CLASS_REPORTS_DIR / f"{slug}_report.txt").write_text(
            classification_report(y_test, pred, target_names=classes, digits=4, zero_division=0))
        cm = confusion_matrix(y_test, pred, labels=list(range(k)))
        fig = viz.plot_confusion_matrix(cm, classes, f"Confusion matrix - {name}")
        viz.save_fig(fig, f"confusion_matrix_{slug}.png"); plt.close(fig)

    fig = viz.plot_model_comparison(comparison)
    viz.save_fig(fig, "model_comparison.png"); plt.close(fig)

    curves = {}
    for name in pipes:
        fpr, tpr = viz.macro_roc_curve(y_test, store[name]["proba"], k)
        curves[name] = (fpr, tpr, comparison.set_index("ML Model Name").loc[name, "AUC"])
    fig = viz.plot_roc_comparison(curves)
    viz.save_fig(fig, "roc_comparison.png"); plt.close(fig)

    pd.set_option("display.width", 200)
    print(comparison.round(4).to_string(index=False))
    print()
    print(details.round(4).to_string(index=False))
    print("\nAUC cross-check vs sklearn OvR macro: max abs diff =", max(cross_check.values()))


if __name__ == "__main__":
    main()
