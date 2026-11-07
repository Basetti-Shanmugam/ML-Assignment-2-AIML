# Final quality checklist (from the faculty step-by-step guide)

Status key: ✅ verified by this project's automated checks / notebook run · ⬜ **you must do/confirm this** · ⚠️ note

| # | Guide item | Status | Evidence / what to do |
|---|---|---|---|
| 1 | Dataset satisfies assignment minimums (≥12 features, ≥500 rows) | ✅ | 16 features, 13,543 rows after cleaning (13,611 raw) |
| 2 | Notebook executes top to bottom without manual variable recovery | ✅ | Executed with nbconvert, zero error outputs; rerun it yourself: *Restart & Run All* |
| 3 | All five models use the same split and preprocessing | ✅ | One stratified 80:20 split (seed 42); every model is `StandardScaler → classifier` |
| 4 | All six required metrics are calculated | ✅ | `reports/model_comparison.csv` (multiclass: macro P/R/F1, OvR macro AUC, MCC) |
| 5 | Observations match actual results | ✅ | Generated from the CSVs; every claim asserted in `tools/observations.py` |
| 6 | Five complete pipelines are saved | ✅ | `models/*.joblib` (+ label encoder, metadata); reload test passes |
| 7 | `test_data.csv` has predictor columns and true target labels | ✅ | `data/test_data.csv` = 16 features + `Class`, 2,709 rows |
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
