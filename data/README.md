# Data folder

| Path | Content |
|---|---|
| `raw/Dry_Bean_Dataset.xlsx` | Original author file (13,611 rows × 17 columns), bundled under CC BY 4.0 |
| `raw/dry_bean_raw.csv` | Same values with standardised column names (written by the loader) |
| `processed/dry_bean_clean.csv` | After duplicate removal (13,543 rows) |
| `processed/train.csv`, `processed/test.csv` | The 80:20 stratified split (10,834 / 2,709) with the readable `Class` label |
| `test_data.csv` | Copy of `processed/test.csv` — upload this in the Streamlit app |

## Getting the data yourself
- **Option A (UCI):** `pip install ucimlrepo`, then `from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=602)` (wrapped in `src.data_loader.load_from_uci()`).
- **Option B (Kaggle / authors):** download from https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset or https://www.muratkoklu.com/datasets/, place the `.xlsx`/`.csv` in `data/raw/`.
  Never commit your `kaggle.json` token.

Column spelling differs between sources (`AspectRation`/`roundness` vs `AspectRatio`/`Roundness`); `standardize_columns()` renames them — values are never changed.

**Attribution:** KOKLU, M. and OZKAN, I.A., (2020), Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. Computers and Electronics in Agriculture, 174, 105507. DOI: https://doi.org/10.1016/j.compag.2020.105507 · UCI DOI https://doi.org/10.24432/C50S4B · License CC BY 4.0.
