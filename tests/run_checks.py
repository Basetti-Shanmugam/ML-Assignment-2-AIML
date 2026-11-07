"""Project test battery.  Run:  python tests/run_checks.py
Writes reports/TESTING_REPORT.md. A check is PASS only if it actually ran and held;
anything that could not be exercised in this environment is reported as SKIPPED / MANUAL."""
from __future__ import annotations

import ast, importlib, json, py_compile, re, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

import config
from src import data_loader, evaluate_models as ev, preprocessing, train_models

RESULTS: list[tuple[str, str, str]] = []   # (status, name, detail)


def check(name):
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append(("PASS", name, str(detail)))
        except AssertionError as e:
            RESULTS.append(("FAIL", name, f"AssertionError: {e}"))
        except Exception as e:  # noqa
            RESULTS.append(("FAIL", name, f"{type(e).__name__}: {e}"))
        return fn
    return deco


def skip(name, why):
    RESULTS.append(("SKIPPED", name, why))


PY_FILES = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts]

# ------------------------------------------------------------------ static
@check("Every Markdown table in the project docs/notebook text is well-formed (same cell count on every row)")
def _():
    n_tables = 0
    for f in [ROOT / "README.md", ROOT / "PROJECT_OVERVIEW.md", ROOT / "data" / "README.md", ROOT / "screenshots" / "README.md",
              ROOT / "deployment" / "GITHUB_DEPLOYMENT.md", ROOT / "deployment" / "STREAMLIT_DEPLOYMENT.md"]:
        block = []
        for ln in f.read_text().splitlines() + [""]:
            if ln.startswith("|"):
                block.append(ln)
            elif block:
                widths = {len(b.strip().strip("|").split("|")) for b in block}
                assert len(widths) == 1, f"{f.name}: table with inconsistent cell counts {widths}: {block[0][:60]}"
                n_tables += 1; block = []
    return f"{n_tables} tables checked"

@check("app.py never trains: no fit()/fit_transform()/partial_fit() call anywhere in app.py")
def _():
    tree = ast.parse((ROOT / "app.py").read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    bad = [c for c in calls if c in {"fit", "fit_transform", "partial_fit", "fit_predict"}]
    assert not bad, f"training call found in app.py: {bad}"
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            imported |= {n.module or ""} | {f"{n.module}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
    assert not any("train_models" in m for m in imported), f"app imports the training module: {imported}"

@check("Syntax check of every Python file")
def _():
    for p in PY_FILES:
        py_compile.compile(str(p), doraise=True)
    return f"{len(PY_FILES)} files compiled"

@check("No empty .py files / placeholder functions")
def _():
    for p in PY_FILES:
        txt = p.read_text().strip()
        assert txt, f"{p.name} is empty"
        if p.name == "run_checks.py":
            continue   # this checker mentions the keyword in its own assertion text
        tree = ast.parse(txt)
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                body = [b for b in n.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant))]
                assert body and not all(isinstance(b, ast.Pass) for b in body), f"{p.name}:{n.name} is a stub"
                assert "NotImplementedError" not in ast.unparse(n), f"{p.name}:{n.name} raises NotImplementedError"

@check("No hard-coded local/absolute paths in source files")
def _():
    bad = []
    pat = re.compile(r"(/home/|/mnt/|C:\\\\|C:/|/Users/)")
    for p in PY_FILES:
        if p.name == "run_checks.py":
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line) and not line.strip().startswith("#"):
                bad.append(f"{p.name}:{i}")
    assert not bad, bad

@check("Required files/folders from the assignment structure exist")
def _():
    need = ["README.md", "requirements.txt", ".gitignore", "app.py", "config.py", "data/README.md", "data/raw",
            "data/processed", "data/test_data.csv", "notebooks/ML_Assignment_2.ipynb", "src/__init__.py",
            "src/data_loader.py", "src/preprocessing.py", "src/train_models.py", "src/evaluate_models.py",
            "src/visualization.py", "models/logistic_regression.joblib", "models/decision_tree.joblib",
            "models/knn.joblib", "models/naive_bayes.joblib", "models/svm.joblib", "reports/model_comparison.csv",
            "reports/classification_reports", "reports/figures", "screenshots/README.md",
            "deployment/GITHUB_DEPLOYMENT.md", "deployment/STREAMLIT_DEPLOYMENT.md"]
    missing = [n for n in need if not (ROOT / n).exists()]
    assert not missing, missing
    return f"{len(need)} paths present"

@check("requirements.txt covers every third-party import")
def _():
    std = set(sys.stdlib_module_names)
    local = {"config", "src", "observations"}
    imp = set()
    for p in PY_FILES:
        if p.parent.name in {"tools", "tests"}:
            continue   # build/test helpers (nbformat etc.) are in requirements-dev.txt
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Import):
                imp |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                imp.add(n.module.split(".")[0])
    third = imp - std - local
    pkg = {"sklearn": "scikit-learn", "streamlit": "streamlit", "pandas": "pandas", "numpy": "numpy",
           "matplotlib": "matplotlib", "seaborn": "seaborn", "joblib": "joblib", "ucimlrepo": "ucimlrepo"}
    reqs = (ROOT / "requirements.txt").read_text().lower()
    unknown = [t for t in third if t not in pkg]
    assert not unknown, f"unmapped imports: {unknown}"
    missing = [pkg[t] for t in third if pkg[t].lower() not in reqs]
    assert not missing, missing
    assert "openpyxl" in reqs   # needed by pandas.read_excel (Option B)
    return "imports -> " + ", ".join(sorted(pkg[t] for t in third))

@check("All src modules and app helpers import")
def _():
    for m in ["config", "src.data_loader", "src.preprocessing", "src.train_models", "src.evaluate_models", "src.visualization"]:
        importlib.import_module(m)

# ------------------------------------------------------------------ data
df_raw = data_loader.load_from_file()

@check("Option B (file) loads and validates: 13,611 x 17, 16 features, target 'Class', no missing")
def _():
    rep = data_loader.validate_dataset(df_raw)
    assert rep["shape"] == (13611, 17) and rep["n_features"] == 16 and rep["missing_values"] == 0 and rep["n_classes"] == 7
    assert rep["n_features"] >= 12 and rep["shape"][0] >= 500
    return f"duplicates reported: {rep['duplicate_rows']}"

@check("Both source naming styles (Excel 'AspectRation/roundness' vs UCI 'AspectRatio/Roundness') standardise identically (simulated)")
def _():
    a = df_raw.rename(columns={"AspectRatio": "AspectRation", "Roundness": "roundness"})
    b = df_raw.rename(columns={"AspectRatio": "AspectRatio", "Roundness": "Roundness"})
    assert data_loader.standardize_columns(a).equals(data_loader.standardize_columns(b))
    assert data_loader.schemas_match(a, b)
    assert not data_loader.schemas_match(df_raw, df_raw.drop(columns=["Area"]))

@check("Raw CSV written by loader has the same values as the Excel source")
def _():
    csv = pd.read_csv(config.RAW_CSV)
    pd.testing.assert_frame_equal(csv, df_raw, check_dtype=False)

try:
    from ucimlrepo import fetch_ucirepo
    fetch_ucirepo(id=config.UCI_DATASET_ID)
    _uci_ok = True
except Exception as e:  # noqa
    _uci_ok = False
    skip("Option A (ucimlrepo download from UCI)", f"Not testable here: {type(e).__name__} - the sandbox has no access to archive.ics.uci.edu. MANUAL: run `python -c \"from src.data_loader import load_from_uci as f; print(f().shape)\"` on a machine with internet; expected (13611, 17).")
if _uci_ok:
    @check("Option A (ucimlrepo) yields the same standardised schema as Option B")
    def _():
        a = data_loader.load_from_uci()
        assert data_loader.schemas_match(a, df_raw) and a.shape == df_raw.shape

# ------------------------------------------------------------------ preprocessing / leakage
clean, log = preprocessing.clean_dataset(df_raw)
split = preprocessing.split_data(clean)

@check("Cleaning removes only exact duplicates and never alters values")
def _():
    assert log["duplicates_removed"] == 68 and len(clean) == 13611 - 68 and not clean.duplicated().any()
    assert clean.equals(df_raw.drop_duplicates().reset_index(drop=True))

@check("Split: 80:20, stratified, reproducible, disjoint, no duplicate row shared by train and test")
def _():
    n = len(clean)
    assert abs(len(split["X_test"]) / n - 0.20) < 0.001
    again = preprocessing.split_data(clean)
    assert split["X_test"].index.equals(again["X_test"].index) and split["X_train"].index.equals(again["X_train"].index)
    assert set(split["X_train"].index).isdisjoint(split["X_test"].index)
    tr = split["y_train"].value_counts(normalize=True).sort_index(); te = split["y_test"].value_counts(normalize=True).sort_index()
    assert (tr - te).abs().max() < 0.002, (tr - te).abs().max()
    merged = pd.concat([split["X_train"], split["X_test"]]).duplicated().sum()
    assert merged == 0

@check("Saved processed/test files equal the split (test_data.csv == data/processed/test.csv == X_test)")
def _():
    t1 = pd.read_csv(config.SAMPLE_TEST_CSV); t2 = pd.read_csv(config.TEST_CSV)
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(t1[config.FEATURE_COLS], split["X_test"].reset_index(drop=True), check_dtype=False)
    tr = pd.read_csv(config.TRAIN_CSV)
    assert len(tr) + len(t1) == len(clean)

pipes, le, meta = ev.load_models()

@check("All five pipelines load, are Pipeline(StandardScaler -> clf) and are fitted")
def _():
    assert set(pipes) == set(config.MODEL_FILES)
    for name, p in pipes.items():
        assert [s for s, _ in p.steps] == ["scaler", "clf"], name
        assert hasattr(p.named_steps["scaler"], "mean_") and hasattr(p, "predict_proba"), name
    assert len(le.classes_) == 7

@check("No data leakage: every saved scaler was fitted on TRAIN only (mean == train mean != full mean)")
def _():
    for name, p in pipes.items():
        m = p.named_steps["scaler"].mean_
        assert np.allclose(m, split["X_train"].mean().values), name
        assert not np.allclose(m, clean[config.FEATURE_COLS].mean().values), name

@check("Models were trained on the training rows only (training fit never sees test rows; code review + scaler n_samples_seen_)")
def _():
    for name, p in pipes.items():
        assert int(p.named_steps["scaler"].n_samples_seen_) == len(split["X_train"]), name
    src = (ROOT / "src" / "train_models.py").read_text()
    assert "pipe.fit(X_train, y_train)" in src and "X_test" not in src.split("def train_all")[1].split("def save_all")[0]

@check("Serialization round-trip: re-saving and re-loading a pipeline gives identical predictions")
def _():
    import tempfile
    Xt = split["X_test"]
    for name, p in pipes.items():
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.joblib"; joblib.dump(p, f)
            assert (joblib.load(f).predict(Xt) == p.predict(Xt)).all(), name

@check("All six metrics finite & in range for all five models; AUC equals sklearn OvR-macro; matches reports/model_comparison.csv")
def _():
    from sklearn.metrics import roc_auc_score
    Xt, yt = split["X_test"], split["y_test"].values
    tab, store = ev.evaluate_all(pipes, Xt, yt, 7)
    saved = pd.read_csv(config.COMPARISON_CSV).set_index("ML Model Name")
    for _, r in tab.iterrows():
        for m in ev.METRIC_COLS:
            v = r[m]; assert np.isfinite(v), (r["ML Model Name"], m)
            assert (-1 <= v <= 1) if m == "MCC" else (0 <= v <= 1)
            assert abs(round(v, 4) - saved.loc[r["ML Model Name"], m]) < 1e-9, (r["ML Model Name"], m)
        ref = roc_auc_score(yt, store[r["ML Model Name"]]["proba"], multi_class="ovr", average="macro")
        assert abs(ref - r["AUC"]) < 1e-12
    assert list(saved.columns) == ev.METRIC_COLS

@check("Undefined-metric handling: test file with only 3 of 7 classes gives finite metrics (AUC over scorable classes)")
def _():
    t = pd.read_csv(config.SAMPLE_TEST_CSV); t = t[t["Class"].isin(["SEKER", "CALI", "BOMBAY"])]
    y = le.transform(t["Class"]); X = t[config.FEATURE_COLS]
    tab, _ = ev.evaluate_all(pipes, X, y, 7)
    assert np.isfinite(tab[ev.METRIC_COLS].values).all()
    one = t[t["Class"] == "SEKER"]
    tab1, _ = ev.evaluate_all(pipes, one[config.FEATURE_COLS], le.transform(one["Class"]), 7)
    assert tab1["AUC"].isna().all()   # AUC genuinely undefined with one class -> NaN, no crash

@check("Reports exist: 5 classification reports (csv+txt), 5 confusion matrices, comparison + ROC figures, 5 EDA figures")
def _():
    for slug in ("logistic_regression", "decision_tree", "knn", "naive_bayes", "svm"):
        assert (config.CLASS_REPORTS_DIR / f"{slug}_report.csv").exists() and (config.CLASS_REPORTS_DIR / f"{slug}_report.txt").exists()
        assert (config.FIGURES_DIR / f"confusion_matrix_{slug}.png").exists()
    for f in ("model_comparison.png", "roc_comparison.png", "class_distribution.png", "feature_distributions.png",
              "boxplots.png", "correlation_heatmap.png"):
        assert (config.FIGURES_DIR / f).exists(), f

# ------------------------------------------------------------------ uploaded-CSV validation
@check("Upload validation: sample file accepted; also accepts the 'AspectRation'/'roundness' spelling")
def _():
    t = pd.read_csv(config.SAMPLE_TEST_CSV)
    _, e = data_loader.validate_uploaded_df(t); assert not e, e
    _, e = data_loader.validate_uploaded_df(t.rename(columns={"AspectRatio": "AspectRation", "Roundness": "roundness"})); assert not e, e

@check("Upload validation rejects: empty file, missing column, extra column, non-numeric value, NaN")
def _():
    t = pd.read_csv(config.SAMPLE_TEST_CSV).head(50)
    assert data_loader.validate_uploaded_df(pd.DataFrame())[1]
    assert "Missing" in data_loader.validate_uploaded_df(t.drop(columns=["Area"]))[1][0]
    assert "Unexpected" in data_loader.validate_uploaded_df(t.assign(Junk=1))[1][0]
    bad = t.copy(); bad["Area"] = bad["Area"].astype(object); bad.iloc[0, 0] = "abc"
    assert "numeric" in data_loader.validate_uploaded_df(bad)[1][0]
    nan = t.copy(); nan.iloc[0, 1] = np.nan
    assert "missing value" in data_loader.validate_uploaded_df(nan)[1][0]
    assert "Missing" in data_loader.validate_uploaded_df(t.drop(columns=["Class"]))[1][0]

# ------------------------------------------------------------------ Streamlit
@check("Streamlit app (headless AppTest): starts, sample flow works for all 5 models, metrics equal reports/model_comparison.csv")
def _():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs][:2] == ["📋 Dashboard", "📊 Evaluation"]
    sel = at.selectbox(key="model_select"); assert sel.options == list(config.MODEL_FILES)
    at.button(key="use_sample").click().run(); assert not at.exception, at.exception
    saved = pd.read_csv(config.COMPARISON_CSV).set_index("ML Model Name")
    for name in config.MODEL_FILES:
        at.selectbox(key="model_select").select(name).run(); assert not at.exception, (name, at.exception)
        shown = {m.label.split(" ")[0]: m.value for m in at.metric if m.label.split(" ")[0] in ev.METRIC_COLS}
        for k in ev.METRIC_COLS:
            assert abs(float(shown[k]) - saved.loc[name, k]) < 1e-4, (name, k, shown[k], saved.loc[name, k])
    assert not at.error, [e.value for e in at.error]
    return "metrics shown in the app equal the saved CSV for all five models"

@check("Streamlit app: friendly prompt before any data; error banner (no crash) when the loaded CSV is invalid")
def _():
    from streamlit.testing.v1 import AppTest
    import tempfile
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception and not at.error
    assert any("Upload a test CSV" in i.value for i in at.info)
    # Point the app's 'sample' path at a deliberately broken CSV to exercise its error display
    good = config.SAMPLE_TEST_CSV
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.csv"
        pd.read_csv(good).head(20).drop(columns=["Area"]).assign(Junk=1).to_csv(bad, index=False)
        config.SAMPLE_TEST_CSV = bad
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
            at.button(key="use_sample").click().run()
            assert not at.exception, at.exception
            msgs = " ".join(e.value for e in at.error)
            assert "Missing required column" in msgs and "Unexpected column" in msgs, msgs
        finally:
            config.SAMPLE_TEST_CSV = good

@check("Real server start: `streamlit run app.py` serves /_stcore/health = ok")
def _():
    port = 8765
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
                             "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT)
    try:
        ok = False
        for _ in range(40):
            time.sleep(1)
            try:
                if urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).read().decode() == "ok":
                    ok = True; break
            except Exception:
                pass
        assert ok, "health endpoint never returned ok"
    finally:
        proc.terminate(); proc.wait(timeout=10)

# ------------------------------------------------------------------ report
skip("Streamlit Community Cloud deployment / public URL", "MANUAL: requires your GitHub + Streamlit account (see deployment/STREAMLIT_DEPLOYMENT.md). Not deployed or tested by the author of this package.")
skip("GitHub repository creation / push", "MANUAL: requires your GitHub account (see deployment/GITHUB_DEPLOYMENT.md). Nothing was uploaded.")
skip("Browser rendering of the app (visual layout, mobile responsiveness, file-upload widget with a real browser upload)", "MANUAL: AppTest cannot drive the st.file_uploader widget or judge visuals. Upload data/test_data.csv in a browser and check each tab.")
skip("BITS Virtual Lab execution + screenshot", "MANUAL: required by the assignment; see screenshots/README.md.")

n = {s: sum(1 for r in RESULTS if r[0] == s) for s in ("PASS", "FAIL", "SKIPPED")}
lines = ["# Testing Report", "",
         f"Generated by `python tests/run_checks.py` on {time.strftime('%Y-%m-%d %H:%M')}  ",
         f"Environment: Python {sys.version.split()[0]}, scikit-learn {__import__('sklearn').__version__}, "
         f"pandas {pd.__version__}, streamlit {__import__('streamlit').__version__}", "",
         f"**Summary: {n['PASS']} passed · {n['FAIL']} failed · {n['SKIPPED']} skipped / manual**", "",
         "| Status | Check | Detail |", "|---|---|---|"]
for s, name, d in RESULTS:
    lines.append(f"| {s} | {name} | {d.replace('|', '/').replace(chr(10), ' ')} |")
lines += ["", "## Known limitations",
          "- The test split is 2,709 rows; the standard error of an accuracy of ~0.92 is about 0.005, so differences between the top models (SVM vs Logistic Regression) are within noise.",
          "- Hyperparameters are sensible defaults chosen up front, not tuned (tuning would need a separate validation procedure).",
          "- Option A (UCI download) could not be executed in the authoring environment (no network access to UCI); it relies on the same `standardize_columns()` logic that was tested with simulated UCI-style column names.",
          "- `.joblib` models are tied to scikit-learn 1.8.0 (pinned in requirements.txt).",
          "", "## Steps requiring manual verification (by you)",
          "1. Run Option A on a machine with internet and confirm `(13611, 17)`.",
          "2. Push to GitHub; deploy on Streamlit Community Cloud; open the public URL and tick every item in `deployment/STREAMLIT_DEPLOYMENT.md` (section 'Verification checklist').",
          "3. In a browser upload `data/test_data.csv` and confirm the metrics equal `reports/model_comparison.csv`.",
          "4. Run the notebook in the BITS Virtual Lab and take the required screenshot."]
(ROOT / "reports" / "TESTING_REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(f"{s:8s} {name}" + (f"  -> {d[:140]}" if s != 'PASS' else '') for s, name, d in RESULTS))
print(f"\n{n}")
sys.exit(1 if n["FAIL"] else 0)
