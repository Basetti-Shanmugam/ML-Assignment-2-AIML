"""Builds notebooks/ML_Assignment_2.ipynb (then execute with nbconvert)."""
import sys
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import config
import observations as O

c_, d_, obs, summ = O.build()
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(f"""
# Machine Learning – Assignment 2
### Multiclass classification of dry beans · M.Tech (AIML)

**Goal.** Train and compare five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) on the
*same* dataset and the *same* stratified 80:20 split, using six metrics: Accuracy, AUC, Precision, Recall, F1 and MCC.

**Dataset.** UCI *Dry Bean* – 13,611 instances, 16 numeric features, target `Class` (7 bean varieties). License: CC BY 4.0.
- UCI: {config.UCI_URL}  (DOI {config.UCI_DOI})
- Kaggle: {config.KAGGLE_URL}
- Authors' page: {config.AUTHOR_URL}
- Citation: {config.CITATION}

**Notebook map** – Phase 1 data loading → 2 validation & EDA → 3 preprocessing → 4 training → 5 metrics → 6 comparison → 7 saving models.
All reusable logic lives in `src/`; this notebook calls it and explains each step, so the notebook, the CLI scripts and the Streamlit app always agree.
""")

code("""
import sys, os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# Make the project root importable whether the notebook is opened from notebooks/ or from the root.
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.display import display, Markdown

import config
from src import data_loader, preprocessing, train_models, evaluate_models as ev, visualization as viz
%matplotlib inline
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
print("Project root:", ROOT)
""")

md("""
## Phase 1 · Dataset selection and loading

**Concept.** A reproducible project needs a scripted way to obtain the data. Two independent sources are provided below;
**exactly one** should be active. Both are routed through `standardize_columns()` so every later cell sees identical column names
(the original Excel file spells two columns `AspectRation` and `roundness`, the UCI listing uses `AspectRatio` and `Roundness`;
only the *names* are harmonised – values are never touched).

**How to switch source:** comment out the active option and un-comment the other. Nothing else in the notebook changes.

> In this delivered notebook **Option B is active** because it was executed in an environment without access to the UCI server.
> On your machine (with internet) you can make Option A the active one – see the troubleshooting list at the end if it fails.
""")

code("""
# ============================================================================
# OPTION A: UCI  (ucimlrepo, needs internet)
# ----------------------------------------------------------------------------
# To use: un-comment the next line AND comment out Option B below.
# df_raw = data_loader.load_from_uci()          # fetch_ucirepo(id=602) -> features + targets -> one DataFrame
# ============================================================================

# ============================================================================
# OPTION B: KAGGLE / author download (Excel or CSV in data/raw/)
# ----------------------------------------------------------------------------
# 1. Download from Kaggle (https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset)
#    or the authors' page, and put the file in data/raw/ (the .xlsx is already bundled).
# 2. Choose the file and the name of the target column below.
SOURCE_FILE = config.RAW_EXCEL        # e.g. config.RAW_DIR / "Dry_Bean_Dataset.csv" also works
TARGET_COLUMN = "Class"               # change only if your file names the target differently
df_raw = data_loader.load_from_file(SOURCE_FILE, target_col=TARGET_COLUMN)
# ============================================================================

data_loader.save_raw_csv(df_raw)      # keep a CSV copy in data/raw/ (identical values, standardised names)
print("Loaded:", df_raw.shape, "| columns:", list(df_raw.columns)[:4], "...")
""")

md("""
### Validation: does the dataset satisfy the assignment, and do both sources give the same schema?
`validate_dataset` raises an error if any of the 16 features or the target is missing or an unexpected column exists.
We then *simulate* the other source's column spelling and confirm both map to the same schema
(this checks the renaming logic; it is not a download from the other source).
""")

code("""
report = data_loader.validate_dataset(df_raw)
print(f"Dimensions        : {report['shape'][0]} rows x {report['shape'][1]} columns")
print(f"Input features    : {report['n_features']}  (assignment minimum 12 -> {'OK' if report['n_features'] >= 12 else 'FAIL'})")
print(f"Instances         : {report['shape'][0]}  (assignment minimum 500 -> {'OK' if report['shape'][0] >= 500 else 'FAIL'})")
print(f"Target column     : {report['target']}  ({report['n_classes']} classes)")
print(f"Missing values    : {report['missing_values']}")
print(f"Duplicate records : {report['duplicate_rows']}")
print("Class distribution:", report['class_distribution'])
print()
print(df_raw.dtypes.to_string())

# Same-schema check using both naming styles (simulated)
excel_style = df_raw.rename(columns={"AspectRatio": "AspectRation", "Roundness": "roundness"})
uci_style = df_raw.rename(columns={"AspectRatio": "AspectRatio", "Roundness": "Roundness"})
assert data_loader.schemas_match(excel_style, uci_style)
assert data_loader.standardize_columns(excel_style).equals(data_loader.standardize_columns(uci_style))
print("\\nSchema check passed: both naming styles standardise to identical columns and values.")
""")

md("""
## Phase 2 · Exploratory Data Analysis (EDA)
**Why EDA?** Before modelling we must know the data's size, types, quality problems (missing values, duplicates, outliers),
class balance and feature relationships – these drive the preprocessing and the choice of metrics.
Figures are saved to `reports/figures/`.
""")
code("print('First 5 records'); display(df_raw.head())\nprint('Last 5 records'); display(df_raw.tail())")
code("""
print("Shape:", df_raw.shape)
print("\\nColumn names:", list(df_raw.columns))
print("\\nData types and non-null counts:")
df_raw.info()
cat_cols = df_raw.drop(columns=[config.TARGET_COL]).select_dtypes(include=["object", "string", "category"]).columns.tolist()
num_cols = df_raw.drop(columns=[config.TARGET_COL]).select_dtypes(include="number").columns.tolist()
print(f"\\nCategorical predictors: {cat_cols or 'none'}   |   Numerical predictors: {len(num_cols)}")
print("-> every predictor is numeric, so no encoder is needed for X; only the target is label-encoded.")
""")
md("**Statistical summary** – look at the very different scales (e.g. `Area` ~ 10⁴–10⁵ vs `Solidity` ≈ 0.9). Distance- and gradient-based models need scaling; this is why a `StandardScaler` is part of every pipeline.")
code("display(df_raw.describe().T.round(4))")
md("**Missing values and duplicates.** Missing values would need imputation or row removal; exact duplicate rows would let the *same* bean appear in both the train and test sets and inflate scores.")
code("""
missing = df_raw.isna().sum()
print("Missing values per column (non-zero only):", missing[missing > 0].to_dict() or "none")
dups = df_raw.duplicated()
print("Duplicate rows:", int(dups.sum()), f"({dups.mean():.2%})")
display(df_raw[df_raw.duplicated(keep=False)].sort_values(list(df_raw.columns)).head(6))
""")
md("**Target class distribution and imbalance.** If a class is rare, accuracy alone can hide poor performance on it – hence macro-averaged metrics and MCC.")
code("""
counts = df_raw[config.TARGET_COL].value_counts()
display(counts.to_frame("count").assign(share=(counts / counts.sum()).round(4)))
print(f"Imbalance ratio (largest / smallest class): {counts.max() / counts.min():.2f}")
fig = viz.plot_class_distribution(df_raw); viz.save_fig(fig, "class_distribution.png"); plt.show()
""")
md("**Feature distributions** – check skewness and ranges.")
code("fig = viz.plot_feature_distributions(df_raw); viz.save_fig(fig, 'feature_distributions.png'); plt.show()")
md("**Boxplots** – outliers show up as points beyond the whiskers. We do **not** delete them: they are genuine measurements and we must not alter data silently; scaling + robust models cope with them.")
code("fig = viz.plot_boxplots(df_raw); viz.save_fig(fig, 'boxplots.png'); plt.show()")
md("**Correlation heatmap** – many size-related features (Area, ConvexArea, EquivDiameter, Perimeter...) are almost perfectly correlated. This matters for Naive Bayes (assumes independence) and for interpreting Logistic Regression coefficients.")
code("""
fig = viz.plot_correlation_heatmap(df_raw); viz.save_fig(fig, 'correlation_heatmap.png'); plt.show()
corr = df_raw[config.FEATURE_COLS].corr().abs()
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().dropna().sort_values(ascending=False)
print(f"{(pairs > 0.9).sum()} of {len(pairs)} feature pairs have |r| > 0.9. Strongest:")
print(pairs.head(5).round(4).to_string())
""")
md("**Do features separate the classes?** A few features split by class.")
code("fig = viz.plot_features_by_class(df_raw); viz.save_fig(fig, 'features_by_class.png'); plt.show()")

md("""
## Phase 3 · Data preprocessing
Steps (all in `src/preprocessing.py`):
1. Separate features `X` (16 columns) and target `y`.
2. **Missing values** – none are present; the code would drop affected rows rather than invent values.
3. **Duplicates** – exact duplicate rows are removed (first copy kept) *before* splitting so no bean can appear in both sets. The count removed is reported, never hidden.
4. **Encode target** – `LabelEncoder` maps class names to 0..6 (needed for the AUC/MCC calculations); the encoder is saved with the models.
5. **Split** – 80:20, `stratify=y`, `random_state=42`: one split, reused by all five models.
6. **Scaling** – `StandardScaler` is a step *inside each Pipeline*, so it is fitted on training data only. Fitting it on the full dataset would leak test-set mean/variance into training.
""")
code("""
df, clean_log = preprocessing.clean_dataset(df_raw)
print("Cleaning log:", clean_log)
df.to_csv(config.CLEAN_CSV, index=False)

split = preprocessing.split_data(df)
X_train, X_test = split["X_train"], split["X_test"]
y_train, y_test = split["y_train"], split["y_test"]
le = split["label_encoder"]; classes = list(le.classes_); K = len(classes)
preprocessing.save_splits(split)   # writes data/processed/train.csv, test.csv and data/test_data.csv

print(f"Train: {X_train.shape}   Test: {X_test.shape}")
print("Classes (encoded 0..K-1):", dict(enumerate(classes)))
share = pd.DataFrame({"train": y_train.value_counts(normalize=True).sort_index(),
                      "test": y_test.value_counts(normalize=True).sort_index()}); share.index = classes
display(share.round(4))   # stratification check: proportions should match
""")
md("**Leakage check.** The scaler inside each pipeline will be fitted on `X_train` only; below we prove the stored mean equals the *training* mean, not the full-data mean.")

md("""
## Phase 4 · Model implementation
All five models are `Pipeline([StandardScaler, classifier])`. Hyperparameters were set **a priori** (sensible defaults) and were **not** tuned on the test set.

| Model | How it works | Why appropriate here | Key hyper-parameters used | Advantages | Limitations |
|---|---|---|---|---|---|
| **Logistic Regression** | Learns a linear score per class and turns them into probabilities with softmax; minimises cross-entropy. | Fast, strong baseline for numeric features once scaled. | `max_iter=2000`, default L2 `C=1` | Probabilistic, interpretable, little overfitting | Only linear class boundaries |
| **Decision Tree** | Recursively splits features to make child nodes purer (Gini). | Needs no scaling, captures non-linear rules. | `max_depth=10`, `min_samples_leaf=5` (limit overfitting) | Interpretable, handles mixed scales | Unstable, overfits easily, coarse probabilities |
| **KNN** | Predicts by (distance-weighted) vote of the k closest training points. | Classes form clusters in shape space. | `n_neighbors=7`, `weights='distance'` | No training, flexible boundaries | Needs scaling, slow prediction, sensitive to noisy/correlated features |
| **Gaussian Naive Bayes** | Bayes' rule with the assumption that features are independent and Gaussian within a class. | Features are continuous; very cheap baseline. | defaults (`var_smoothing=1e-9`) | Very fast, few parameters | Independence assumption is violated by correlated features |
| **SVM (RBF)** | Finds a maximum-margin boundary in an implicit high-dimensional space via the RBF kernel; multiclass via one-vs-one. | Smooth non-linear boundaries on scaled data. | `C=1.0`, `gamma='scale'`, `probability=True` | Strong accuracy, robust to outliers via margin | Slower on big data, needs scaling, `probability=True` adds internal cross-validation cost |

`probability=True` for SVM makes `predict_proba()` available (Platt scaling) so AUC is computed identically for all five models.
""")
code("""
pipes = train_models.build_pipelines()
for name, p in pipes.items():
    print(f"{name:<20s}", " -> ".join(f"{n}:{type(s).__name__}" for n, s in p.steps))

print("\\nTraining on the TRAIN split only ...")
pipes, train_secs = train_models.train_all(X_train, y_train)

# Leakage proof: scaler statistics equal the training-set statistics
sc = pipes["SVM"].named_steps["scaler"]
assert np.allclose(sc.mean_, X_train.mean().values)
assert not np.allclose(sc.mean_, df[config.FEATURE_COLS].mean().values)
print("\\nLeakage check passed: scaler mean == training mean (and != full-data mean).")
""")

md("""
## Phase 5 · Evaluation metrics
For this 7-class problem:
- **Accuracy** – share of correct predictions.
- **AUC** – multiclass ROC-AUC, **one-vs-rest, macro-averaged**, computed from `predict_proba()`.
- **Precision / Recall / F1** – **macro-averaged** (each class counts equally, which matters because Bombay is only ~4 % of the data).
- **MCC** – Matthews correlation coefficient (multiclass version), a single balanced score that is robust to imbalance.

`zero_division=0` is used so a class that is never predicted gives 0 rather than an error. Our `safe_macro_auc` equals sklearn's `roc_auc_score(multi_class='ovr', average='macro')` when every class is present (asserted below) and only differs by skipping classes that are absent from an uploaded test file.
""")
code("""
from sklearn.metrics import roc_auc_score
results, store = ev.evaluate_all(pipes, X_test, y_test, K)
for name in pipes:   # AUC cross-check against sklearn's built-in
    ref = roc_auc_score(y_test, store[name]["proba"], multi_class="ovr", average="macro")
    assert abs(ref - results.set_index("ML Model Name").loc[name, "AUC"]) < 1e-12
results_rounded = results.round(4)
results_rounded.to_csv(config.COMPARISON_CSV, index=False)
display(results_rounded)
""")
md("### Confusion matrix and classification report for every model")
code("""
from sklearn.metrics import confusion_matrix, classification_report
for name in pipes:
    slug = config.MODEL_FILES[name].replace(".joblib", "")
    pred = store[name]["pred"]
    cm = confusion_matrix(y_test, pred)
    fig = viz.plot_confusion_matrix(cm, classes, f"Confusion matrix - {name}")
    viz.save_fig(fig, f"confusion_matrix_{slug}.png"); plt.show()
    print(f"Classification report - {name}")
    print(classification_report(y_test, pred, target_names=classes, digits=4, zero_division=0))
""")

md("## Phase 6 · Model comparison")
code("""
fig = viz.plot_model_comparison(results); viz.save_fig(fig, "model_comparison.png"); plt.show()
curves = {}
for name in pipes:
    fpr, tpr = viz.macro_roc_curve(y_test, store[name]["proba"], K)
    curves[name] = (fpr, tpr, results.set_index("ML Model Name").loc[name, "AUC"])
fig = viz.plot_roc_comparison(curves); viz.save_fig(fig, "roc_comparison.png"); plt.show()
best = results.set_index("ML Model Name")[ev.METRIC_COLS].idxmax()
print("Best model per metric:"); print(best.to_string())
""")
md("""
### Generalisation diagnostics (training data only; the test set is not used for CV)
Training accuracy vs 5-fold cross-validated accuracy vs test accuracy shows which models over-fit.
""")
code("""
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
skf = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
rows = []
for name, p in pipes.items():
    cvp = clone(p)
    if name == "SVM": cvp.set_params(clf__probability=False)   # accuracy does not need Platt scaling; saves time
    cv = cross_val_score(cvp, X_train, y_train, cv=skf, scoring="accuracy")
    tr = accuracy_score(y_train, p.predict(X_train)); te = accuracy_score(y_test, store[name]["pred"])
    rows.append({"ML Model Name": name, "Train Accuracy": tr, "CV Accuracy (5-fold, train only)": cv.mean(),
                 "CV Std": cv.std(), "Test Accuracy": te, "Train-Test Gap": tr - te})
details = pd.DataFrame(rows).round(4); details.to_csv(config.DETAILS_CSV, index=False); display(details)
""")

md(f"""
### Model performance observations (computed from the saved results)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {n} | " + " | ".join(f"{r[m]:.4f}" for m in O.M) + " |" for n, r in c_.iterrows()) + f"""

{O.md_obs_table(obs)}

**Which model performs best?** {summ['best_accuracy']} (top on Accuracy, Precision, Recall, F1, MCC).  
**Highest AUC:** {summ['best_auc']} (SVM is only 0.0002 behind).  
**Best precision/recall balance:** {summ['best_balance']}.  
**Generalisation:** {summ['generalises']}  
**Poor performers:** {summ['poor']}  
{summ['class_note']}

{summ['winner_text']}

*These statements are generated from `reports/model_comparison.csv` and `reports/model_comparison_details.csv` by `tools/observations.py`, which asserts every claim against the numbers.*
""")

md("## Phase 7 · Saving the trained pipelines and metadata")
code("""
train_models.save_all(pipes, split, clean_log, train_secs)
for f in sorted(config.MODELS_DIR.iterdir()): print(f"{f.name:<28s}{f.stat().st_size/1024:>9.1f} KB")

# Reload test: a saved pipeline must reproduce the in-memory predictions exactly
for name, fname in config.MODEL_FILES.items():
    reloaded = joblib.load(config.MODELS_DIR / fname)
    assert (reloaded.predict(X_test) == store[name]["pred"]).all()
print("\\nAll five saved pipelines reload and reproduce identical predictions.")
loaded_svm = joblib.load(config.MODELS_DIR / "svm.joblib")          # load WITHOUT retraining
print("Loaded SVM predicts the first 5 test beans as:", le.inverse_transform(loaded_svm.predict(X_test.head(5))))
print("True labels of those 5 beans              :", list(le.inverse_transform(y_test.head(5))))
print("Held-out test file for the Streamlit app:", config.SAMPLE_TEST_CSV.relative_to(ROOT))
""")

md("""
## Conclusion
All five models were trained on the same stratified 80:20 split of the cleaned Dry Bean data and scored with the same six metrics.
See the observations above for the winner and the caveats. Next steps: run `python -m src.evaluate_models` to regenerate every report/figure from the CLI, launch `streamlit run app.py`, and follow `deployment/` for GitHub and Streamlit Community Cloud.

### Common errors and fixes
| Symptom | Cause | Fix |
|---|---|---|
| `ConnectionError: Error connecting to server` in Option A | No internet / UCI blocked | Use Option B with the downloaded file |
| `FileNotFoundError ... Dry_Bean_Dataset.xlsx` | File not in `data/raw/` | Download from Kaggle/UCI and place it there |
| `ModuleNotFoundError: src` | Notebook started from another folder | Run Jupyter from the project root or `notebooks/` |
| `ValueError: Schema mismatch` | Different/extra columns in your file | Check the printed lists; target must be named via `TARGET_COLUMN` |
| `InconsistentVersionWarning` when loading `.joblib` | scikit-learn version differs from 1.8.0 | `pip install -r requirements.txt` or retrain with `python -m src.train_models` |
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "ML_Assignment_2.ipynb"
nbf.write(nb, out); print("wrote", out)
