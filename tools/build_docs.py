"""Writes README.md, data/README.md, screenshots/README.md, deployment/*.md, PROJECT_OVERVIEW.md
and submission/ML_Assignment_2_Submission.pdf from the REAL result files.
Run after training/evaluation:  python tools/build_docs.py"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import pandas as pd
import config
import observations as O

c, d, obs, S = O.build()
meta = json.loads(config.METADATA_PATH.read_text())
RES_TABLE = O.md_results_table(c)
OBS_TABLE = O.md_obs_table(obs)
GH = "<YOUR_GITHUB_REPO_URL>   <!-- TODO: paste after you create the repository -->"
APP = "<YOUR_STREAMLIT_APP_URL>   <!-- TODO: paste after you deploy and test the app -->"
n_raw, n_clean, n_tr, n_te = meta["n_instances_raw"], meta["n_instances_after_cleaning"], meta["n_train"], meta["n_test"]
cls = meta["classes"]
counts = pd.read_csv(config.CLEAN_CSV)["Class"].value_counts()
dist_md = ", ".join(f"{k} {v}" for k, v in counts.items())

FEATURES = [("Area", "Number of pixels inside the bean region"), ("Perimeter", "Length of the bean border"),
            ("MajorAxisLength", "Longest line that can be drawn through the bean"),
            ("MinorAxisLength", "Longest line perpendicular to the major axis"),
            ("AspectRatio", "MajorAxisLength / MinorAxisLength (spelled `AspectRation` in the original Excel file)"),
            ("Eccentricity", "Eccentricity of the ellipse with the same moments as the region"),
            ("ConvexArea", "Pixels in the smallest convex polygon containing the bean"),
            ("EquivDiameter", "Diameter of a circle with the same area as the bean"),
            ("Extent", "Bounding-box pixels / bean area"), ("Solidity", "Convex-hull pixels / bean pixels (convexity)"),
            ("Roundness", "4πA / P² (spelled `roundness` in the original Excel file)"),
            ("Compactness", "EquivDiameter / MajorAxisLength"),
            ("ShapeFactor1", "Shape factor 1 (dimensionless)"), ("ShapeFactor2", "Shape factor 2"),
            ("ShapeFactor3", "Shape factor 3"), ("ShapeFactor4", "Shape factor 4")]
feat_md = "| # | Feature | Description |\n|---|---|---|\n" + "\n".join(f"| {i} | `{n}` | {t} |" for i, (n, t) in enumerate(FEATURES, 1))

TREE = """ML_Assignment_2/
├── README.md                   <- this file (also included in the submission PDF)
├── PROJECT_OVERVIEW.md         <- what every file does + how to run/deploy
├── SUBMISSION_CHECKLIST.md     <- the faculty guide's final checklist with honest status
├── requirements.txt            <- runtime dependencies (used by Streamlit Cloud)
├── requirements-dev.txt        <- adds jupyter/nbconvert for running the notebook
├── .gitignore
├── app.py                      <- Streamlit application
├── config.py                   <- paths, constants, dataset citation
├── .streamlit/config.toml      <- theme / upload limit
├── data/
│   ├── README.md
│   ├── raw/                    <- Dry_Bean_Dataset.xlsx (Option B) + dry_bean_raw.csv
│   ├── processed/              <- dry_bean_clean.csv, train.csv, test.csv
│   └── test_data.csv           <- the held-out 20 % test split (use it in the app)
├── notebooks/ML_Assignment_2.ipynb   <- executed notebook (Option A / B loaders, EDA, models)
├── src/
│   ├── data_loader.py          <- Option A (UCI) / Option B (file) loaders, validation
│   ├── preprocessing.py        <- cleaning, encoding, stratified split
│   ├── train_models.py         <- builds, trains, saves the five pipelines
│   ├── evaluate_models.py      <- six metrics, reports, figures
│   └── visualization.py        <- EDA + evaluation plots
├── models/                     <- 5 x .joblib pipelines + label_encoder.joblib + metadata.json
├── reports/
│   ├── model_comparison.csv, model_comparison_details.csv, TESTING_REPORT.md
│   ├── classification_reports/ <- per-model csv + txt
│   └── figures/                <- EDA + confusion matrices + comparison + ROC
├── screenshots/README.md       <- what to capture (BITS Virtual Lab etc.)
├── deployment/                 <- GITHUB_DEPLOYMENT.md, STREAMLIT_DEPLOYMENT.md
├── submission/                 <- ML_Assignment_2_Submission.pdf (template + README content)
├── tests/run_checks.py         <- automated test battery (writes reports/TESTING_REPORT.md)
└── tools/                      <- scripts that regenerate notebook / docs from the results
"""

README = f"""# Dry Bean Classification — Machine Learning Assignment 2
*M.Tech Artificial Intelligence & Machine Learning (AIML)*

## a. Problem Statement
Dry beans of different registered varieties look alike, and manual sorting is slow and subjective. The task is to build, evaluate and
compare **five supervised classifiers** that predict the **bean variety** (`Class`, 7 categories) from 16 geometric measurements
obtained from grain images. All models use the same data and the same stratified 80:20 train/test split and are scored with
Accuracy, AUC, Precision, Recall, F1 and MCC. The best model is then exposed through an interactive Streamlit application.

## b. Dataset Description
- **Name:** Dry Bean Dataset (UCI ID 602) — multiclass classification
- **Instances:** {n_raw:,} (raw) → **{n_clean:,}** after removing {meta['duplicates_removed']} exact duplicate rows
- **Input features:** 16 numeric (12 dimension features + 4 shape factors) — assignment minimum is 12
- **Target:** `Class` — {meta['n_classes']} classes: {', '.join(cls)}
- **Missing values:** 0 · **Class counts after cleaning:** {dist_md}
  (imbalance ratio {counts.max() / counts.min():.1f}:1, largest/smallest)

{feat_md}

## c. Dataset Source and Attribution
- UCI Machine Learning Repository: {config.UCI_URL} (DOI {config.UCI_DOI})
- Kaggle mirror: {config.KAGGLE_URL}
- Authors' dataset page: {config.AUTHOR_URL}
- License: Creative Commons Attribution 4.0 (CC BY 4.0), so redistribution with attribution is permitted; the Excel file is bundled in `data/raw/`.
- **Citation:** {config.CITATION}
- UCI citation: *Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. {config.UCI_DOI}*

> Two loaders are provided in the notebook (**Option A** UCI via `ucimlrepo`, **Option B** downloaded Kaggle/author file); both are
> converted to one standard schema by `src/data_loader.py`. Option B was used to produce all results in this repository.

## d. GitHub Repository Link
{GH}

## e. Data Preprocessing
1. Features and target separated; schema validated (16 features + `Class`).
2. Missing values: none found (the code would drop such rows; nothing is imputed or invented).
3. Duplicates: **{meta['duplicates_removed']} exact duplicate rows removed** (first copy kept) *before* splitting, so no sample can appear in both train and test.
4. Target encoded with `LabelEncoder` (saved in `models/label_encoder.joblib`).
5. **Stratified 80:20 split**, `random_state=42` → **{n_tr:,} train / {n_te:,} test**, shared by all five models.
6. `StandardScaler` is the first step of every `Pipeline`, therefore fitted on training data only (no leakage; verified in `tests/run_checks.py`).
7. Outliers are kept (genuine measurements); nothing in the data is modified.

## f. Models Used
| Model | Pipeline | Key hyper-parameters (fixed a priori, not tuned on test data) |
|---|---|---|
| Logistic Regression | StandardScaler → LogisticRegression | `max_iter=2000`, L2, `C=1` |
| Decision Tree | StandardScaler → DecisionTreeClassifier | `max_depth=10`, `min_samples_leaf=5` |
| K-Nearest Neighbors | StandardScaler → KNeighborsClassifier | `n_neighbors=7`, `weights='distance'` |
| Naive Bayes | StandardScaler → GaussianNB | defaults |
| SVM | StandardScaler → SVC | RBF, `C=1`, `gamma='scale'`, `probability=True` |

How each works, why it suits this data and its pros/cons are explained in the notebook (Phase 4).

## g. Model Evaluation Metrics
Computed on the {n_te:,}-row held-out test set; multiclass settings: **Accuracy**; **AUC** = one-vs-rest, macro-averaged, from `predict_proba`;
**Precision / Recall / F1** = macro-averaged; **MCC** = multiclass Matthews correlation coefficient. `zero_division=0` handles classes that are never predicted.

## h. Model Comparison Table
{RES_TABLE}

![Model comparison](reports/figures/model_comparison.png)

Additional generalisation diagnostics (`reports/model_comparison_details.csv`; CV uses the training data only):

| Model | Train acc. | 5-fold CV acc. | Test acc. | Train–test gap |
|---|---|---|---|---|
""" + "\n".join(f"| {n} | {r['Train Accuracy']:.4f} | {r['CV Accuracy (5-fold, train only)']:.4f} ± {r['CV Std']:.4f} | {r['Test Accuracy']:.4f} | {r['Train-Test Gap']:.4f} |" for n, r in d.iterrows()) + f"""

## i. Model Performance Observations
{OBS_TABLE}

- **Best model overall by accuracy:** {S['best_accuracy']} · **Highest AUC:** {S['best_auc']}
- **Best precision/recall balance:** {S['best_balance']}
- **Generalisation:** {S['generalises']}
- **Poor performers:** {S['poor']}
- {S['class_note']}

## j. Overall Winning Model
{S['winner_text']}

## k. Streamlit Application Description
`app.py` loads the five saved pipelines (it never retrains) and provides: a dashboard (dataset facts), CSV upload with validation
(missing/unexpected columns, non-numeric or missing values, empty file, unknown class labels), a model dropdown, the six metrics,
a confusion matrix (counts or row-normalised), a per-class classification report, a five-model comparison table + chart on the uploaded data,
and a predictions table (actual vs predicted, confidence, misclassified filter) with CSV download.
Each model has a short description in the sidebar, the confusion matrix shows both counts and row percentages, and the Evaluation tab states that uploaded data is not used for retraining.
A bundled-sample button uses `data/test_data.csv` (the held-out test split). The uploaded file must contain the 16 feature columns and `Class`.

**Saved-model / no-retraining deployment method.** Training happens once, offline (`python -m src.train_models` or the notebook). Each model is saved as a complete
scikit-learn `Pipeline` (StandardScaler + classifier) with `joblib.dump`, so the scaler statistics learned from the training split travel with the model.
At run time `app.py` only calls `joblib.load`, `predict`, `predict_proba` and metric functions - it contains no `fit()` call (checked automatically in `tests/run_checks.py`)
and does not import the training module. Uploaded rows are used only for prediction and scoring. Because pickled pipelines depend on library versions,
`requirements.txt` pins `scikit-learn==1.8.0` and the other packages to the versions used for training.

## l. Deployment Link
{APP}

## m. Project Folder Structure
```
{TREE}```

## n. Installation Instructions
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd ML_Assignment_2
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt      # add requirements-dev.txt if you want to run the notebook
```
Developed and tested with Python 3.12 (scikit-learn 1.8 requires Python ≥ 3.11). **scikit-learn is pinned to 1.8.0** because the `.joblib` files were saved with that version.

## o. Execution Instructions
```bash
python -m src.train_models        # clean data, split, train the 5 pipelines, save to models/
python -m src.evaluate_models     # six metrics, reports, confusion matrices, figures
python -m src.visualization       # EDA figures
python tests/run_checks.py        # automated checks -> reports/TESTING_REPORT.md
streamlit run app.py              # launch the app, then upload data/test_data.csv
jupyter notebook notebooks/ML_Assignment_2.ipynb   # full walkthrough (needs requirements-dev.txt)
```
To use **Option A (UCI)** in the notebook, un-comment `data_loader.load_from_uci()` and comment out the Option B lines in the loading cell.

## p. Conclusion
All five classifiers reach roughly {c['Accuracy'].min():.0%}–{c['Accuracy'].max():.0%} test accuracy on the 7-class Dry Bean problem, with macro-AUC between {c['AUC'].min():.3f} and {c['AUC'].max():.3f}.
{S['winner']} gives the best overall result, with Logistic Regression statistically very close and much simpler; Naive Bayes is weakest, consistent with the strong
feature correlations that violate its independence assumption. The hardest distinction is {O.top_confusion('svm')[0]} vs {O.top_confusion('svm')[1]}.
Limitations: hyper-parameters were not tuned, a single train/test split was used for the headline numbers, and differences between the top two models are within sampling noise.
Possible extensions: grid/random search with nested cross-validation, feature selection to remove redundant size features, and ensemble methods.

---
Note: the results in this README are generated from the CSV files in `reports/`; `reports/TESTING_REPORT.md` lists what was and was not tested.
"""
(ROOT / "README.md").write_text(README)

# ---------------------------------------------------------------- data/README
(ROOT / "data" / "README.md").write_text(f"""# Data folder

| Path | Content |
|---|---|
| `raw/Dry_Bean_Dataset.xlsx` | Original author file ({n_raw:,} rows × 17 columns), bundled under CC BY 4.0 |
| `raw/dry_bean_raw.csv` | Same values with standardised column names (written by the loader) |
| `processed/dry_bean_clean.csv` | After duplicate removal ({n_clean:,} rows) |
| `processed/train.csv`, `processed/test.csv` | The 80:20 stratified split ({n_tr:,} / {n_te:,}) with the readable `Class` label |
| `test_data.csv` | Copy of `processed/test.csv` — upload this in the Streamlit app |

## Getting the data yourself
- **Option A (UCI):** `pip install ucimlrepo`, then `from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=602)` (wrapped in `src.data_loader.load_from_uci()`).
- **Option B (Kaggle / authors):** download from {config.KAGGLE_URL} or {config.AUTHOR_URL}, place the `.xlsx`/`.csv` in `data/raw/`.
  Never commit your `kaggle.json` token.

Column spelling differs between sources (`AspectRation`/`roundness` vs `AspectRatio`/`Roundness`); `standardize_columns()` renames them — values are never changed.

**Attribution:** {config.CITATION} · UCI DOI {config.UCI_DOI} · License CC BY 4.0.
""")

# ---------------------------------------------------------------- screenshots
(ROOT / "screenshots" / "README.md").write_text("""# Screenshots to capture (manual - the project cannot generate these)

## 1. BITS Virtual Lab / Prayogshala evidence (required by the assignment guide)
After your app is deployed:
1. Open the assigned **Virtual Lab / Prayogshala VM**.
2. Inside the VM, open your **GitHub repository** and your **live Streamlit URL**.
3. In the live app, **upload `test_data.csv`** (download it from the app's sidebar or from the repository's `data/` folder).
4. **Select at least one model** and make sure the six metric cards are visible.
5. Take **one screenshot** in which all of these are visible together:
   - the Virtual Lab environment (VM window / desktop),
   - the live app URL in the address bar,
   - the uploaded CSV file name,
   - the application heading ("Dry Bean Variety Classifier"),
   - the results (metrics, and ideally the confusion matrix).
6. Save it as `screenshots/bits_virtual_lab.png` and put it into the PDF:
   `python tools/build_submission_pdf.py --screenshot screenshots/bits_virtual_lab.png ...`

## 2. Optional extras
| File name | What to show |
|---|---|
| `app_dashboard.png` | Dashboard tab |
| `app_evaluation.png` | Evaluation tab (metrics + confusion matrix + classification report) |
| `app_comparison.png` | Model-comparison tab |
| `app_predictions.png` | Predictions tab with the download button |
| `github_repo.png` | Your repository page |
""")

# ---------------------------------------------------------------- deployment
(ROOT / "deployment" / "GITHUB_DEPLOYMENT.md").write_text(f"""# Publishing the project to GitHub (manual steps)

> Nothing has been uploaded for you. Follow these steps with your own account. Never commit `kaggle.json`, tokens or `.env` files (already in `.gitignore`).
> The whole project is ~20 MB; the largest file is the 3 MB Excel dataset, far below GitHub's 100 MB limit.

## 1. Create a GitHub account (skip if you have one)
1. Go to https://github.com/signup, enter email, password, username; verify the email.
2. (Recommended) enable two-factor authentication under *Settings → Password and authentication*.

## 2. Create an empty repository
1. Click **+ → New repository**.
2. Repository name: `ML_Assignment_2` (or similar). Visibility: **Public** (Streamlit Community Cloud's free tier needs access to the repo; public is simplest, and your evaluator can open it).
3. **Do NOT** tick "Add a README", ".gitignore" or "license" (the project already has them - this avoids merge conflicts).
4. Click **Create repository** and copy the HTTPS URL, e.g. `https://github.com/<your-username>/ML_Assignment_2.git`.

## 3. Install and configure Git (once)
```bash
git --version                                   # if missing: https://git-scm.com/downloads
git config --global user.name  "Your Name"
git config --global user.email "your-github-email@example.com"
```

## 4. Initialise, commit and push

**About commit history.** Your faculty guide asks for a *genuine* sequence of commits and says not to manufacture history after finishing a project.
Splitting finished work into fake "milestones", back-dating commits or reusing generic messages would misrepresent how you worked, so this guide does not do that.
Commit when you *really* complete a step, with a message describing that step.

```bash
cd ML_Assignment_2
git init -b main
git status                                      # check: no kaggle.json / secrets listed
git remote add origin https://github.com/<your-username>/ML_Assignment_2.git
```

**If you are still working through the project (recommended):** run, read and adapt each part yourself, and commit after each real step, e.g.
```bash
git add README.md requirements.txt .gitignore config.py data
git commit -m "Initial project setup and dataset selection"           # when you have chosen/checked the dataset
git add src/data_loader.py src/preprocessing.py notebooks
git commit -m "Add dataset exploration and preprocessing"             # after you ran and understood the EDA
git add src/train_models.py
git commit -m "Implement the five classifiers as scikit-learn pipelines"
git add src/evaluate_models.py src/visualization.py reports
git commit -m "Add evaluation metrics, comparison table and observations"
git add models data/test_data.csv
git commit -m "Save trained pipelines and held-out test data"
git add app.py .streamlit
git commit -m "Add Streamlit evaluation application"
git add deployment tests tools PROJECT_OVERVIEW.md SUBMISSION_CHECKLIST.md screenshots submission
git commit -m "Add deployment documentation and tests"
git push -u origin main
```
Only commit a step after you have really done it; edit the messages so they describe what *you* changed.

**If you are submitting the finished project as it is:** use one honest commit and then record the genuine later changes as separate commits.
```bash
git add .
git commit -m "Add complete Dry Bean classification project (notebook, models, Streamlit app)"
git push -u origin main
# ...later, each genuinely separate change:
git add README.md && git commit -m "Add GitHub and Streamlit links to README" && git push
```
**Authentication:** GitHub no longer accepts account passwords for `git push`. When prompted, use your username and a **Personal Access Token** as the password
(*Settings → Developer settings → Personal access tokens → Fine-grained → repository access to this repo, permission "Contents: Read and write"*), or install the GitHub CLI and run `gh auth login`, or use Git Credential Manager (bundled with Git for Windows).

## 5. Verify
1. Refresh the repository page: you should see `app.py`, `requirements.txt`, `models/`, `src/`, `data/`, `notebooks/` etc. and the README rendered with tables and the comparison image.
2. Open `models/` - the five `.joblib` files must be listed (if they are missing, check they are not excluded by `.gitignore`).
3. Open the notebook on GitHub - outputs should render.
4. Optional full test: clone into a new folder and run it
```bash
git clone https://github.com/<your-username>/ML_Assignment_2.git test_clone && cd test_clone
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python tests/run_checks.py
```
5. Paste the repository URL into README section **d** and commit it as its own change:
```bash
git add README.md && git commit -m "Add GitHub repository link to README" && git push
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `fatal: not a git repository` | Run `git init -b main` inside the project folder |
| `remote origin already exists` | `git remote set-url origin <URL>` |
| `Authentication failed` | Use a Personal Access Token (not your password) or `gh auth login` |
| `rejected ... fetch first` | You created the repo with a README: `git pull origin main --rebase` then push |
| Large-file warning | Not expected here; never commit raw Kaggle zips >50 MB |
""")

(ROOT / "deployment" / "STREAMLIT_DEPLOYMENT.md").write_text(f"""# Deploying on Streamlit Community Cloud (manual steps)

> This has **not** been deployed for you. Only claim a deployment after you have opened the public URL and ticked the checklist below.

## Before you start
- The code must already be on GitHub (see `GITHUB_DEPLOYMENT.md`), repo public (or connected to your Streamlit account), branch `main`.
- Files required by the app: `app.py`, `requirements.txt`, `config.py`, `src/`, `models/*.joblib`, `models/label_encoder.joblib`, `models/metadata.json`, `data/test_data.csv`.

## Steps
1. Open https://share.streamlit.io and click **Continue with GitHub** (or sign up with the same email as your GitHub account). Authorise Streamlit to access your repositories.
2. Click **Create app** (top right) → **Deploy a public app from GitHub**.
3. **Repository:** choose `<your-username>/ML_Assignment_2` (use *Paste GitHub URL* if it is not listed).
4. **Branch:** `main`.
5. **Main file path:** `app.py`.
6. (Optional) choose a custom subdomain, e.g. `dry-bean-<your-name>`.
7. Click **Advanced settings** → **Python version: 3.12** (the project was tested with Python 3.12; scikit-learn 1.8.0 needs ≥ 3.11). No secrets are needed.
8. Click **Deploy**. The first build installs `requirements.txt` (typically 2-5 minutes). Watch the log on the right.
9. When it finishes, copy the URL (`https://<subdomain>.streamlit.app`) into README section **l**, the PDF template, then `git add README.md && git commit -m "Add Streamlit deployment link" && git push`.

## Verification checklist (tick every item on the PUBLIC URL)
- [ ] The app opens without an error and shows the Dry Bean dashboard (no red error box).
- [ ] All five models load (the model dropdown lists Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM and no "Could not load models" message).
- [ ] CSV upload works: upload `data/test_data.csv` from your computer.
- [ ] Model selection works: switching the dropdown changes the numbers.
- [ ] Metrics display correctly. With `test_data.csv` they must equal `reports/model_comparison.csv`, e.g. {c.index[-1]}: Accuracy {c.loc[c.index[-1], 'Accuracy']:.4f}, AUC {c.loc[c.index[-1], 'AUC']:.4f}.
- [ ] Confusion matrix displays (and the normalise toggle works).
- [ ] Classification report shows precision / recall / F1 / support per class.
- [ ] Model-comparison tab shows the table and chart.
- [ ] Predictions tab shows Actual vs Predicted and **Download predictions (CSV)** downloads a file.
- [ ] Metrics are identical to the notebook for the same test file (same CSV, target mapping, feature order, saved model, no retraining).
- [ ] The note "uploaded data is not used for retraining" is visible on the Evaluation tab.
- [ ] Negative test: upload a CSV with a missing column → a clear red error message, no crash.

## Troubleshooting
| Symptom in the log / app | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'seaborn'` (or similar) | Package missing from `requirements.txt` | Add it, commit, push; app redeploys automatically (or *Reboot app*) |
| `ModuleNotFoundError: No module named 'src'` / `config` | Files not pushed, or app file is not in the repo root | Check GitHub shows `src/` and `config.py` next to `app.py`; main file path must be `app.py` |
| `FileNotFoundError: ... models/svm.joblib` | `models/` not pushed, or a different working directory | Check the folder on GitHub; paths come from `config.py` (relative to the project) so no code changes needed |
| `InconsistentVersionWarning` / `AttributeError` / `ValueError: node array ...` when loading a model | scikit-learn version on the server differs from the one used for training | Keep `scikit-learn==1.8.0` in `requirements.txt` (do not loosen it), or retrain locally with `python -m src.train_models` and push the new `models/` |
| Build fails: "No matching distribution found for scikit-learn==1.8.0" | Python version too old on the server | In *Advanced settings* choose Python 3.11-3.13, then *Reboot* |
| `TypeError: ... unexpected keyword 'width'` | Very old Streamlit installed | Keep `streamlit>=1.50` in `requirements.txt` |
| App shows "Oh no. Error running app" with a long traceback | Any exception | Open *Manage app → logs*, read the last lines, fix, push |
| Changed the Python version after deploying | Streamlit Cloud may require deleting and redeploying the app | Choose Python 3.12 at first deployment; if you must change it, delete the app and deploy again |
| App sleeps ("This app has gone to sleep") | Free apps hibernate after inactivity | Click *Yes, get this app back up* before your demo and open the URL once beforehand |
""")

(ROOT / "PROJECT_OVERVIEW.md").write_text(f"""# Project overview

## What each file does
| File | Purpose |
|---|---|
| `config.py` | Single source of paths (relative, no hard-coded folders), seeds, class/feature names, dataset links and citation |
| `src/data_loader.py` | **Option A** `load_from_uci()` (ucimlrepo) and **Option B** `load_from_file()` (Excel/CSV); `standardize_columns()` makes both sources identical; `validate_dataset()`; `validate_uploaded_df()` used by the app |
| `src/preprocessing.py` | `clean_dataset()` (missing/duplicates), `encode_target()`, `split_data()` (stratified 80:20, seed 42), `save_splits()` |
| `src/train_models.py` | Builds the five `Pipeline(StandardScaler → classifier)` objects, trains on the training split only, saves `.joblib` + metadata |
| `src/evaluate_models.py` | `compute_metrics()` (Accuracy, AUC, Precision, Recall, F1, MCC), `safe_macro_auc()`, writes comparison CSVs, classification reports, figures |
| `src/visualization.py` | EDA plots, confusion matrix, comparison chart, macro ROC curves |
| `app.py` | Streamlit UI: upload → validate → choose model → metrics, confusion matrix, report, comparison, predictions |
| `notebooks/ML_Assignment_2.ipynb` | Teaching notebook: both data loaders, EDA, preprocessing, models, metrics, observations (already executed) |
| `tests/run_checks.py` | Automated checks; writes `reports/TESTING_REPORT.md` |
| `tools/*.py` | Regenerate the notebook and the documents/PDF from the results (not needed to run the app) |
| `models/` | Trained pipelines (`logistic_regression`, `decision_tree`, `knn`, `naive_bayes`, `svm`), `label_encoder.joblib`, `metadata.json` |
| `reports/` | `model_comparison.csv`, details, classification reports, figures, testing report |
| `data/` | Raw + processed data and `test_data.csv` (held-out 20 % split) |
| `deployment/` | Step-by-step GitHub and Streamlit Cloud guides |
| `submission/ML_Assignment_2_Submission.pdf` | Submission template + full README content |

## Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt                        # app + training
pip install -r requirements-dev.txt                    # + Jupyter, to run the notebook
```

## Run the notebook
`jupyter notebook notebooks/ML_Assignment_2.ipynb` → *Kernel → Restart & Run All*. Switch data source in the Phase 1 cell by commenting Option B and un-commenting Option A (needs internet).

## Train the models (command line)
`python -m src.train_models` then `python -m src.evaluate_models` (about 17 seconds in our test run).

## Launch Streamlit
`streamlit run app.py` → open http://localhost:8501 → click **Use bundled sample test set** or upload `data/test_data.csv`.

## Upload to GitHub / deploy
Follow `deployment/GITHUB_DEPLOYMENT.md` then `deployment/STREAMLIT_DEPLOYMENT.md`. Then paste the two links into `README.md` (sections d and l) and regenerate the PDF with your details (needs `requirements-dev.txt`):
```bash
python tools/build_submission_pdf.py --name "Your Name" --student-id "2025XXNNNN" --email "you@example.com" \\
    --github "https://github.com/<you>/ML_Assignment_2" --app "https://<sub>.streamlit.app" \\
    --screenshot screenshots/bits_virtual_lab.png
```
Any option you omit stays as a visible placeholder.

## Viva cheat-sheet (what to be able to explain)
1. Why duplicates are removed *before* the split and why scaling sits *inside* the pipeline (leakage).
2. Why macro-averaged metrics + MCC (class imbalance {counts.max() / counts.min():.1f}:1) and how multiclass AUC is computed (OvR, macro, from `predict_proba`).
3. Why SVM wins on 5 of 6 metrics, yet Logistic Regression has a slightly higher AUC and the gap is within noise.
4. Why KNN's training accuracy is 1.0 (distance weighting) and why Naive Bayes is weakest (correlated features).
5. That the app only loads saved pipelines and evaluates uploaded data - it never retrains.
""")
(ROOT / "SUBMISSION_CHECKLIST.md").write_text(f"""# Final quality checklist (from the faculty step-by-step guide)

Status key: ✅ verified by this project's automated checks / notebook run · ⬜ **you must do/confirm this** · ⚠️ note

| # | Guide item | Status | Evidence / what to do |
|---|---|---|---|
| 1 | Dataset satisfies assignment minimums (≥12 features, ≥500 rows) | ✅ | 16 features, {n_clean:,} rows after cleaning (13,611 raw) |
| 2 | Notebook executes top to bottom without manual variable recovery | ✅ | Executed with nbconvert, zero error outputs; rerun it yourself: *Restart & Run All* |
| 3 | All five models use the same split and preprocessing | ✅ | One stratified 80:20 split (seed 42); every model is `StandardScaler → classifier` |
| 4 | All six required metrics are calculated | ✅ | `reports/model_comparison.csv` (multiclass: macro P/R/F1, OvR macro AUC, MCC) |
| 5 | Observations match actual results | ✅ | Generated from the CSVs; every claim asserted in `tools/observations.py` |
| 6 | Five complete pipelines are saved | ✅ | `models/*.joblib` (+ label encoder, metadata); reload test passes |
| 7 | `test_data.csv` has predictor columns and true target labels | ✅ | `data/test_data.csv` = 16 features + `Class`, {n_te:,} rows |
| 8 | `app.py` contains no `fit()` call | ✅ | AST check in `tests/run_checks.py` |
| 9 | Streamlit outputs reproduce notebook results for the same test set | ✅ locally / ⬜ on the live URL | AppTest matches the CSV for all 5 models; repeat on your deployed app |
| 10 | GitHub commit history reflects genuine milestones | ⬜ | Only you can make this true - see `deployment/GITHUB_DEPLOYMENT.md` §4 |
| 11 | README, live URL, Virtual Lab evidence, final PDF complete | ⬜ | Add the 2 links + screenshot, regenerate the PDF (`PROJECT_OVERVIEW.md`) |
| 12 | Original code, wording, UI and observations | ⬜ | Read, run and adapt the project until you can explain every part; edit text in your own words |

## Other guide requirements
| Item | Status |
|---|---|
| Dataset URL, row count, predictor count, target, license, citation recorded | ✅ README §b–c, `data/README.md` |
| Preprocessing without leakage (split first, scaler inside pipeline) | ✅ verified by test (scaler mean == training mean) |
| `requirements.txt` pinned to training versions | ✅ |
| Deploy with Python 3.12 on Streamlit Community Cloud, main branch, `app.py` | ⬜ `deployment/STREAMLIT_DEPLOYMENT.md` |
| Virtual Lab screenshot (VM + live URL + uploaded CSV + heading + results) | ⬜ `screenshots/README.md` |
| Final PDF order: GitHub link → app link → screenshot → README | ✅ template ready; ⬜ fill in |

## Layout differences to check against your faculty template
The guide's *example* tree uses `model/` (folder of pipelines), a root-level `ML_Assignment2.ipynb` and a root-level `test_data.csv`.
This project follows the structure in your own project brief (`models/`, `notebooks/ML_Assignment_2.ipynb`, `data/test_data.csv`).
If your faculty template requires the other names, change `MODELS_DIR` / `SAMPLE_TEST_CSV` in `config.py` (one line each), move the files and update the README tree.
""")
print("markdown docs written")
