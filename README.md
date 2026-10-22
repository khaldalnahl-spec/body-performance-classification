# Body Performance Classification

**Course:** AINT351 — Data Mining  
**Student:** Khalid Jurayyad Alharbi  
**University:** University of Ha'il

## Project overview
This project uses supervised machine learning to classify physical-performance grades **A, B, C, and D** using the public Body Performance dataset (13,393 observations).

The repository provides an improved, readable Python implementation of the **Milestone 2** workflow: preprocessing, a Logistic Regression baseline, Random Forest feature importance, an age ablation experiment, XGBoost cross-validation, hyperparameter tuning, and model export.

## Reported results
| Experiment | Accuracy | ROC-AUC (OvR) |
|---|---:|---:|
| Logistic Regression baseline, held-out test | ~63% | ~0.853 (5-fold CV) |
| Age ablation, held-out test | ~58% | ~0.823 (5-fold CV) |
| XGBoost, out-of-fold CV | ~75% | ~0.925 |

A **separate Milestone 3** benchmark compared ten algorithms using **11 predictors** and an 80/20 held-out test split (10,714 training; 2,679 test samples). XGBoost and LightGBM both achieved **75.92%** accuracy. The benchmark's recorded metrics are in [milestone3_results.csv](milestone3_results.csv). Its original executable training cell was not included in the available notebook; the CSV records the reported results, rather than claiming to reproduce that experiment.

**Important:** Milestone 2 and Milestone 3 used different feature sets and evaluation protocols. Their scores are **not** directly comparable as a controlled improvement.

## Run the code
1. Download the original `bodyPerformance.csv` from [Kaggle Body Performance Data](https://www.kaggle.com/datasets/kukuroo3/body-performance-data).
2. Place the CSV beside `milestone2_improved.py`.
3. Install packages: `pip install -r requirements.txt`.
4. Run: `python milestone2_improved.py`.

The script saves `xgboost_body_performance.joblib`. Dataset and model binaries are intentionally excluded from the repository.

## Notes on reproducibility
- The source is a **cleaned, readable implementation** based on the Milestone 2 notebook, not a byte-for-byte copy of the original Kaggle notebook.
- The original notebook contained a deprecated Logistic Regression argument and a data-consistency check referencing an older dataframe; these were corrected.
- GridSearchCV and subsequent cross-validation use the same dataset. A rigorous publication-level assessment would use nested CV or a separate untouched test set.
- Body-performance grades are not medical diagnoses.

## Files
- [milestone2_improved.py](milestone2_improved.py) — end-to-end Milestone 2 code.
- [milestone3_results.csv](milestone3_results.csv) — recorded ten-model comparison.
- [requirements.txt](requirements.txt) — dependencies.
