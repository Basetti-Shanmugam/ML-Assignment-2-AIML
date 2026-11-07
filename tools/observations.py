"""Generates the performance-observation text FROM the saved result files.

Each qualitative claim is asserted against the numbers first, so if you retrain
and the story changes, this script fails loudly instead of printing stale text.
Used by the notebook builder, the README builder and the PDF builder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

M = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def load_results():
    comp = pd.read_csv(config.COMPARISON_CSV).set_index("ML Model Name")
    det = pd.read_csv(config.DETAILS_CSV).set_index("ML Model Name")
    return comp, det


def top_confusion(model_slug: str = "svm"):
    """Most frequent off-diagonal confusion for a model, from its saved predictions."""
    import joblib
    le = joblib.load(config.LABEL_ENCODER_PATH)
    pipe = joblib.load(config.MODELS_DIR / f"{model_slug}.joblib")
    test = pd.read_csv(config.TEST_CSV)
    y = le.transform(test[config.TARGET_COL]); p = pipe.predict(test[config.FEATURE_COLS])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, p); np.fill_diagonal(cm, 0)
    i, j = np.unravel_index(cm.argmax(), cm.shape)
    return le.classes_[i], le.classes_[j], int(cm[i, j]) + int(cm[j, i])


def build():
    c, d = load_results()
    f = lambda m, k: f"{c.loc[m, k]:.4f}"
    best = {k: c[k].idxmax() for k in M}
    # ---- assert every claim the text below makes -------------------------------
    assert best["Accuracy"] == best["Precision"] == best["Recall"] == best["F1"] == best["MCC"] == "SVM"
    assert best["AUC"] == "Logistic Regression" and c["AUC"].nlargest(2).index[1] == "SVM"
    assert c["Accuracy"].idxmin() == "Naive Bayes" and c["MCC"].idxmin() == "Naive Bayes"
    assert c["AUC"].idxmin() == "Decision Tree"
    for _m in ["Accuracy", "Precision", "Recall", "F1", "MCC"]:
        assert c[_m].rank(ascending=False)["Logistic Regression"] == 2, _m
    assert list(c["AUC"].sort_values(ascending=False).index[:3]) == ["Logistic Regression", "SVM", "Naive Bayes"]
    assert d["Train Accuracy"]["KNN"] > 0.999
    assert d["Train-Test Gap"]["Naive Bayes"] < 0.001
    assert d["Train-Test Gap"].drop("KNN").idxmax() == "Decision Tree"
    assert d["CV Accuracy (5-fold, train only)"].idxmax() == "SVM"
    n_test = int(d.shape[0] and pd.read_csv(config.TEST_CSV).shape[0])
    se = float(np.sqrt(c.loc["SVM", "Accuracy"] * (1 - c.loc["SVM", "Accuracy"]) / n_test))
    diff = float(c.loc["SVM", "Accuracy"] - c.loc["Logistic Regression", "Accuracy"])
    assert diff < se, "SVM-vs-LR gap exceeds one standard error; rewrite the caveat"
    a, b, n_conf = top_confusion("svm")
    rep = pd.read_csv(config.CLASS_REPORTS_DIR / "svm_report.csv", index_col=0)
    cls_rows = rep.drop(index=["accuracy", "macro avg", "weighted avg"])
    worst_cls, best_cls = cls_rows["f1-score"].idxmin(), cls_rows["f1-score"].idxmax()
    X = pd.read_csv(config.TEST_CSV)  # only to count correlated pairs on the clean data
    clean = pd.read_csv(config.CLEAN_CSV)[config.FEATURE_COLS].corr().abs()
    tri = clean.where(np.triu(np.ones(clean.shape), 1).astype(bool)).stack().dropna()
    n_corr, n_pairs = int((tri > 0.9).sum()), len(tri)
    assert n_pairs == 120, n_pairs

    g = lambda m: f"{d.loc[m, 'Train-Test Gap']:.4f}"
    obs = {
        "Logistic Regression":
            f"Second-best on Accuracy ({f('Logistic Regression','Accuracy')}), Precision, Recall, F1 and MCC, and the highest AUC "
            f"({f('Logistic Regression','AUC')}), only {c.loc['Logistic Regression','AUC'] - c.loc['SVM','AUC']:.4f} above SVM. "
            f"Train-test accuracy gap is just {g('Logistic Regression')}, so it generalises very well. "
            "A linear decision surface on standardised features is almost as good as the non-linear SVM here.",
        "Decision Tree":
            f"Accuracy {f('Decision Tree','Accuracy')} and the lowest AUC of all models ({f('Decision Tree','AUC')}): a tree outputs "
            "piecewise-constant leaf probabilities, so its class ranking is coarser. "
            f"It shows the largest genuine overfitting among the models (train {d.loc['Decision Tree','Train Accuracy']:.4f} vs test "
            f"{f('Decision Tree','Accuracy')}, gap {g('Decision Tree')}) even with max_depth=10 and min_samples_leaf=5.",
        "KNN":
            f"Mid-table: Accuracy {f('KNN','Accuracy')}, F1 {f('KNN','F1')}, AUC {f('KNN','AUC')}. Its training accuracy is 1.0000 only because "
            "distance weighting makes every training point its own nearest neighbour, so the "
            f"{g('KNN')} train-test gap is partly an artefact; the 5-fold CV accuracy on the training data "
            f"({d.loc['KNN','CV Accuracy (5-fold, train only)']:.4f}) is the fairer indicator. Prediction is also the slowest at inference time because it stores the whole training set.",
        "Naive Bayes":
            f"Weakest on Accuracy ({f('Naive Bayes','Accuracy')}), F1 ({f('Naive Bayes','F1')}) and MCC ({f('Naive Bayes','MCC')}), yet third-highest on AUC "
            f"({f('Naive Bayes','AUC')}): it ranks classes reasonably but its hard decisions are worse. "
            f"The independence assumption is clearly violated - {n_corr} of {n_pairs} feature pairs have an absolute correlation above 0.9 - which is a likely reason. "
            f"Gap {g('Naive Bayes')} means no overfitting; the limitation is bias, not variance.",
        "SVM":
            f"Best on Accuracy ({f('SVM','Accuracy')}), Precision ({f('SVM','Precision')}), Recall ({f('SVM','Recall')}), F1 ({f('SVM','F1')}) "
            f"and MCC ({f('SVM','MCC')}); AUC ({f('SVM','AUC')}) is second, {c.loc['Logistic Regression','AUC'] - c.loc['SVM','AUC']:.4f} behind Logistic Regression. "
            f"It also has the best 5-fold CV accuracy on the training data ({d.loc['SVM','CV Accuracy (5-fold, train only)']:.4f}) and a modest gap of {g('SVM')}.",
    }
    summary = {
        "best_accuracy": best["Accuracy"], "best_auc": best["AUC"],
        "best_balance": f"SVM (highest F1 = {f('SVM','F1')}; Precision {f('SVM','Precision')} and Recall {f('SVM','Recall')} are both the highest, differing by {abs(c.loc['SVM','Precision']-c.loc['SVM','Recall']):.4f})",
        "generalises": f"Naive Bayes has the smallest gap ({g('Naive Bayes')}) but at lower accuracy; among the strong models Logistic Regression ({g('Logistic Regression')}) and SVM ({g('SVM')}) generalise best.",
        "poor": f"Naive Bayes (lowest Accuracy/F1/MCC); Decision Tree has the lowest AUC and the biggest real overfitting gap.",
        "winner": "SVM (RBF kernel)",
        "winner_text": (
            f"**Overall winner: Support Vector Machine (RBF).** It ranks first on five of the six metrics (Accuracy, Precision, Recall, F1, MCC) "
            f"and second on AUC (only {c.loc['Logistic Regression','AUC'] - c.loc['SVM','AUC']:.4f} behind Logistic Regression), and has the best cross-validated accuracy on the training data. "
            f"Caveat: its accuracy lead over Logistic Regression is {diff:.4f}, smaller than one standard error of the test accuracy (about {se:.4f} for n={n_test}), "
            "so SVM and Logistic Regression should be regarded as statistically very close; Logistic Regression is the simpler, faster alternative."),
        "class_note": (f"Per-class (SVM): the hardest class is {worst_cls} (F1 {cls_rows.loc[worst_cls,'f1-score']:.4f}); "
                       f"{best_cls} is the easiest (F1 {cls_rows.loc[best_cls,'f1-score']:.4f}). "
                       f"The most frequent mix-up is {a} vs {b} ({n_conf} test beans misassigned between them)."),
    }
    return c, d, obs, summary


def md_results_table(c: pd.DataFrame) -> str:
    lines = ["| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |", "|---|---|---|---|---|---|---|"]
    for name, r in c.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{r[m]:.4f}" for m in M) + " |")
    return "\n".join(lines)


def md_obs_table(obs: dict) -> str:
    lines = ["| ML Model Name | Observation About Model Performance |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in obs.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    c, d, obs, s = build()
    print(md_results_table(c)); print(); print(md_obs_table(obs)); print(); print(s["winner_text"]); print(s["class_note"])
