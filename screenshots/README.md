# Screenshots to capture (manual - the project cannot generate these)

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
