# Deploying on Streamlit Community Cloud (manual steps)

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
- [ ] Metrics display correctly. With `test_data.csv` they must equal `reports/model_comparison.csv`, e.g. SVM: Accuracy 0.9210, AUC 0.9945.
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
