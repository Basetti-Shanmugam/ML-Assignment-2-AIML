"""Plot helpers (Matplotlib + Seaborn). Every function returns a Figure so it can be
saved to reports/figures/ (notebook / CLI) or shown with st.pyplot (Streamlit).

Run:  python -m src.visualization     -> writes the EDA figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe; the notebook overrides with inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

import config

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "viridis"


def save_fig(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return str(path)


# --------------------------------------------------------------------- EDA
def plot_class_distribution(df: pd.DataFrame):
    counts = df[config.TARGET_COL].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.max() * 0.01, f"{v}\n({v / counts.sum():.1%})", ha="center", fontsize=9)
    ax.set_title("Target class distribution")
    ax.set_xlabel("Bean class"); ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.18)
    return fig


def plot_feature_distributions(df: pd.DataFrame):
    cols = config.FEATURE_COLS
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(df[c], kde=True, ax=ax, color="#3b6ea5")
        ax.set_title(c, fontsize=10); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Feature distributions", y=1.01, fontsize=14)
    fig.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for ax, c in zip(axes.ravel(), config.FEATURE_COLS):
        sns.boxplot(y=df[c], ax=ax, color="#8fb8de")
        ax.set_title(c, fontsize=10); ax.set_ylabel("")
    fig.suptitle("Boxplots of numerical features (outlier check)", y=1.01, fontsize=14)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df[config.FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(12, 9.5))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=.3, ax=ax)
    ax.set_title("Feature correlation heatmap (Pearson)")
    return fig


def plot_features_by_class(df: pd.DataFrame, features=("Area", "ShapeFactor1", "Compactness", "Solidity")):
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 4.2))
    for ax, f in zip(np.atleast_1d(axes), features):
        sns.boxplot(data=df, x=config.TARGET_COL, y=f, hue=config.TARGET_COL, palette=PALETTE, legend=False, ax=ax)
        ax.tick_params(axis="x", rotation=60); ax.set_xlabel("")
        ax.set_title(f"{f} by class")
    fig.tight_layout()
    return fig


# -------------------------------------------------------------- evaluation
def plot_confusion_matrix(cm: np.ndarray, classes, title: str):
    """Confusion matrix annotated with the count AND the row percentage
    (row % = share of that true class predicted in each column, i.e. recall on the diagonal).
    Colour encodes the row proportion so rare classes are not washed out."""
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    prop = cm / row_sum
    labels = np.array([[f"{cm[i, j]}\n{prop[i, j]:.1%}" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    sns.heatmap(prop, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=1, xticklabels=classes,
                yticklabels=classes, cbar=False, linewidths=.5, annot_kws={"size": 8}, ax=ax)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"{title}\n(count and row %)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_model_comparison(results: pd.DataFrame):
    """Grouped bar chart of the six metrics for every model."""
    metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    long = results.melt(id_vars="ML Model Name", value_vars=metrics, var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette="Set2", ax=ax)
    lo = max(0.0, float(long["Score"].min()) - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_title("Model performance comparison (test set)")
    ax.set_ylabel("Score (y-axis truncated)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=5, frameon=False)
    fig.tight_layout()
    return fig


def macro_roc_curve(y_true, proba, n_classes: int):
    """Macro-average one-vs-rest ROC curve (average of per-class TPR on a common FPR grid)."""
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    grid = np.linspace(0, 1, 200)
    tprs = []
    for k in range(n_classes):
        if y_bin[:, k].sum() == 0 or y_bin[:, k].sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, k], proba[:, k])
        tprs.append(np.interp(grid, fpr, tpr))
    return grid, np.mean(tprs, axis=0)


def plot_roc_comparison(curves: dict):
    """curves: {model_name: (fpr_grid, tpr_mean, auc)}"""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Macro-average one-vs-rest ROC curves")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def generate_eda_figures(df: pd.DataFrame) -> list[str]:
    paths = []
    for name, fn in [("class_distribution.png", plot_class_distribution),
                     ("feature_distributions.png", plot_feature_distributions),
                     ("boxplots.png", plot_boxplots),
                     ("correlation_heatmap.png", plot_correlation_heatmap),
                     ("features_by_class.png", plot_features_by_class)]:
        fig = fn(df)
        paths.append(save_fig(fig, name))
        plt.close(fig)
    return paths


if __name__ == "__main__":
    data = pd.read_csv(config.RAW_CSV)
    for p in generate_eda_figures(data):
        print("saved", p)
