"""Central configuration. All paths are relative to this file, so the project
runs on any machine (local, Streamlit Cloud, BITS Virtual Lab)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CLASS_REPORTS_DIR = REPORTS_DIR / "classification_reports"

# Dataset files
RAW_EXCEL = RAW_DIR / "Dry_Bean_Dataset.xlsx"       # Kaggle / Koklu download (Option B)
RAW_CSV = RAW_DIR / "dry_bean_raw.csv"              # written by Option A (UCI) or B
CLEAN_CSV = PROCESSED_DIR / "dry_bean_clean.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
SAMPLE_TEST_CSV = DATA_DIR / "test_data.csv"        # the held-out 20 % test split

# Saved artefacts
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
DETAILS_CSV = REPORTS_DIR / "model_comparison_details.csv"

# Experiment settings
UCI_DATASET_ID = 602
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "Class"

# model key -> (display name, file name)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

# Canonical (standardised) schema: 16 features + target
FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRatio",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "Roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3",
    "ShapeFactor4",
]

# Sources / attribution (verified against the UCI page)
UCI_URL = "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset"
UCI_DOI = "https://doi.org/10.24432/C50S4B"
KAGGLE_URL = "https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset"
AUTHOR_URL = "https://www.muratkoklu.com/datasets/"
CITATION = ("KOKLU, M. and OZKAN, I.A., (2020), Multiclass Classification of Dry Beans "
            "Using Computer Vision and Machine Learning Techniques. Computers and "
            "Electronics in Agriculture, 174, 105507. "
            "DOI: https://doi.org/10.1016/j.compag.2020.105507")
