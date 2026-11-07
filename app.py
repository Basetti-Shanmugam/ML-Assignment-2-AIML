"""Streamlit app - Dry Bean classification (ML Assignment 2).

The app LOADS the five saved pipelines from models/ and never retrains.
An uploaded CSV is only used for prediction + evaluation.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

import config
from src import data_loader
from src import evaluate_models as ev
from src import visualization as viz

st.set_page_config(page_title="Dry Bean Classifier | ML Assignment 2", page_icon="🫘", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {background:#F1F5EE; border:1px solid #D7E3D0; border-radius:10px; padding:10px 14px;}
.hero {background:linear-gradient(90deg,#2E7D32,#66A36B); color:white; padding:18px 24px; border-radius:12px; margin-bottom:14px;}
.hero h1 {margin:0; font-size:1.9rem; color:white;}
.hero p {margin:4px 0 0 0; opacity:.92;}
</style>
""", unsafe_allow_html=True)


MODEL_INFO = {
    "Logistic Regression": "**Logistic Regression** - linear model; learns one weighted sum of the scaled features per class and converts the scores to probabilities (softmax). Fast, stable, hard to over-fit.",
    "Decision Tree": "**Decision Tree** - learns a hierarchy of if/else splits (max depth 10). Easy to interpret, but can over-fit and gives coarse probabilities.",
    "KNN": "**K-Nearest Neighbors** - classifies a bean by a distance-weighted vote of its 7 closest training beans in scaled feature space. No training step, slower predictions.",
    "Naive Bayes": "**Gaussian Naive Bayes** - applies Bayes' rule assuming features are independent and normally distributed within each class. Very fast; weakened here by strongly correlated features.",
    "SVM": "**Support Vector Machine (RBF)** - finds a maximum-margin, non-linear boundary using the RBF kernel; probabilities via Platt scaling.",
}


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading the five saved models ...")
def get_models():
    return ev.load_models()


@st.cache_data(show_spinner=False)
def evaluate_everything(df: pd.DataFrame):
    """Predict + score all five models on the uploaded data (no training)."""
    pipes, le, _ = get_models()
    k = len(le.classes_)
    X = df[config.FEATURE_COLS]
    y = le.transform(df[config.TARGET_COL])
    table, store = ev.evaluate_all(pipes, X, y, k)
    return table, store, y


try:
    pipes, le, meta = get_models()
except Exception as exc:  # missing / incompatible model files
    st.error(f"Could not load the saved models from `{config.MODELS_DIR.name}/`: {exc}\n\n"
             "Run `python -m src.train_models` to regenerate them (see README).")
    st.stop()
classes = list(le.classes_)

# ------------------------------------------------------------------ header
st.markdown(f"""<div class="hero"><h1>🫘 Dry Bean Variety Classifier</h1>
<p>M.Tech (AIML) · Machine Learning · Assignment 2 — five classifiers compared on the UCI Dry Bean dataset</p></div>""",
            unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload a test CSV", type=["csv"],
                                help="Must contain the 16 feature columns and the 'Class' column.")
    if st.button("Use bundled sample test set", key="use_sample", width="stretch",
                 help="The held-out 20 % split that none of the models saw during training."):
        st.session_state["source"] = "sample"
    if uploaded is not None:
        st.session_state["source"] = "upload"
    if config.SAMPLE_TEST_CSV.exists():
        st.download_button("Download sample test CSV", config.SAMPLE_TEST_CSV.read_bytes(),
                           "test_data.csv", "text/csv", width="stretch")
    st.header("2 · Model")
    model_name = st.selectbox("Choose a model", list(pipes.keys()), key="model_select")
    st.caption("Models are loaded from `models/*.joblib`; nothing is retrained.")
    st.info(MODEL_INFO[model_name])

# ------------------------------------------------------------------ data intake
df, errors, source_label = None, [], None
if st.session_state.get("source") == "upload" and uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        df, errors = data_loader.validate_uploaded_df(raw)
        source_label = f"uploaded file `{uploaded.name}`"
    except pd.errors.EmptyDataError:
        errors = ["The uploaded file is empty."]
    except Exception as exc:
        errors = [f"Could not read the file as CSV: {exc}"]
elif st.session_state.get("source") == "sample":
    df, errors = data_loader.validate_uploaded_df(pd.read_csv(config.SAMPLE_TEST_CSV))
    source_label = "bundled sample test set (`data/test_data.csv`)"

if not errors and df is not None:
    unknown = sorted(set(df[config.TARGET_COL].astype(str)) - set(classes))
    if unknown:
        errors.append(f"Unknown label(s) in '{config.TARGET_COL}': {unknown}. Allowed: {classes}")

ready = df is not None and not errors

# ------------------------------------------------------------------ tabs
tab_dash, tab_eval, tab_cmp, tab_pred = st.tabs(
    ["📋 Dashboard", "📊 Evaluation", "⚖️ Model comparison", "🔮 Predictions"])

with tab_dash:
    st.subheader("About this application")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instances (raw)", f"{meta['n_instances_raw']:,}")
    c2.metric("After duplicate removal", f"{meta['n_instances_after_cleaning']:,}")
    c3.metric("Input features", meta["n_features"])
    c4.metric("Target classes", meta["n_classes"])
    st.markdown(f"""
**Dataset.** UCI *Dry Bean* (Koklu & Ozkan, 2020): morphological measurements extracted from images of
13,611 grains of seven registered dry-bean varieties — **{', '.join(classes)}**. The target column is `Class`.
Source: [{config.UCI_URL}]({config.UCI_URL}) · Also on [Kaggle]({config.KAGGLE_URL}).

**Training protocol.** {meta['n_train']:,} training / {meta['n_test']:,} test rows (80:20, stratified, `random_state={meta['random_state']}`).
Each model is a scikit-learn `Pipeline` (StandardScaler → classifier) fitted on the training split only.
**Uploaded files are never used for training** - the app only loads the saved pipelines, predicts and evaluates.

**How to use.** ① Upload a CSV (or use the bundled sample) → ② pick a model → ③ explore the tabs.
""")
    with st.expander("Required CSV format"):
        st.write("Columns (exact names; `AspectRation`/`roundness` spellings are also accepted):")
        st.code(", ".join(config.FEATURE_COLS + [config.TARGET_COL]))
    if ready:
        st.success(f"Loaded {source_label}: {len(df):,} rows.")

if errors:
    for e in errors:
        st.error(e)
elif df is None:
    for t in (tab_eval, tab_cmp, tab_pred):
        with t:
            st.info("👈 Upload a test CSV or click **Use bundled sample test set** in the sidebar.")

if ready:
    table, store, y_true = evaluate_everything(df)
    n_classes_present = len(np.unique(y_true))
    if n_classes_present < len(classes):
        st.warning(f"The data contains only {n_classes_present} of {len(classes)} classes. "
                   "AUC is averaged over the classes that can be scored.")
    pred = store[model_name]["pred"]

    with tab_eval:
        st.subheader(f"{model_name} — evaluation on {source_label}")
        st.caption("ℹ️ The uploaded data is used only for prediction and evaluation. It is **not** used to retrain or modify any model.")
        row = table.set_index("ML Model Name").loc[model_name]
        cols = st.columns(6)
        for col, m in zip(cols, ev.METRIC_COLS):
            val = row[m]
            col.metric(m if m != "AUC" else "AUC (OvR macro)", "n/a" if pd.isna(val) else f"{val:.4f}")
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Confusion matrix** (count and row percentage)")
            cm = confusion_matrix(y_true, pred, labels=list(range(len(classes))))
            fig = viz.plot_confusion_matrix(cm, classes, f"{model_name}")
            st.pyplot(fig); plt.close(fig)
        with right:
            st.markdown("**Classification report**")
            rep = classification_report(y_true, pred, labels=list(range(len(classes))),
                                        target_names=classes, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(rep).T.rename(columns={"f1-score": "f1"})
            rep_df["support"] = rep_df["support"].astype(int)
            st.dataframe(rep_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}),
                         width="stretch", height=370)
            st.caption("Macro-averaged precision / recall / F1 are used for the multiclass summary metrics.")

    with tab_cmp:
        st.subheader("All five models on the same data")
        styled = table.style.format({m: "{:.4f}" for m in ev.METRIC_COLS}, na_rep="n/a") \
                            .highlight_max(subset=ev.METRIC_COLS, color="#C8E6C9")
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption("Green = best value in each column.")
        fig = viz.plot_model_comparison(table)
        st.pyplot(fig); plt.close(fig)
        best = table.set_index("ML Model Name")
        st.info("Best per metric: " + " · ".join(f"**{m}** → {best[m].idxmax()}" for m in ev.METRIC_COLS
                                                  if best[m].notna().any()))
        st.download_button("Download comparison table (CSV)", table.round(4).to_csv(index=False).encode(),
                           "model_comparison_uploaded.csv", "text/csv")

    with tab_pred:
        st.subheader(f"Predictions — {model_name}")
        proba = store[model_name]["proba"]
        out = df.copy()
        out = out.rename(columns={config.TARGET_COL: "Actual"})
        out["Predicted"] = le.inverse_transform(pred)
        out["Confidence"] = proba.max(axis=1).round(4)
        out["Correct"] = out["Actual"] == out["Predicted"]
        n_wrong = int((~out["Correct"]).sum())
        c1, c2 = st.columns(2)
        c1.metric("Correct", f"{len(out) - n_wrong:,}")
        c2.metric("Misclassified", f"{n_wrong:,}")
        only_wrong = st.checkbox("Show only misclassified rows", value=False)
        view = out[~out["Correct"]] if only_wrong else out
        front = ["Actual", "Predicted", "Correct", "Confidence"]
        st.dataframe(view[front + config.FEATURE_COLS], width="stretch", height=420)
        buf = io.StringIO(); out[front + config.FEATURE_COLS].to_csv(buf, index=False)
        st.download_button("⬇️ Download predictions (CSV)", buf.getvalue().encode(),
                           f"predictions_{config.MODEL_FILES[model_name].replace('.joblib', '')}.csv", "text/csv")

st.divider()
st.caption("ML Assignment 2 · Data: Koklu & Ozkan (2020), CC BY 4.0 · Models trained offline; this app only predicts and evaluates.")
