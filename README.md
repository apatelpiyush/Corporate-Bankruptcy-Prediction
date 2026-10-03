# Corporate Bankruptcy Prediction

Binary classification (Bankrupt vs Non-Bankrupt) on the UCI Polish Companies Bankruptcy dataset (id=365), fetched at runtime with `ucimlrepo`. No manual downloads. Every number is computed at run time.

## Run
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py                  # full pipeline (needs internet for the UCI fetch)
python src/train.py --cv-folds 3 --n-iter 3     # optional: faster run
python src/predict.py --csv results/sample_input.csv
python src/predict.py --template my_input.csv   # blank CSV with required columns
python src/predict.py --json '{"X1": 0.05, ...}'
jupyter notebook notebooks/bankruptcy_analysis.ipynb
```

## Pipeline
Load (UCI) → inspect → drop exact duplicate rows → EDA → imbalance analysis → stratified 80/20 split → per-model sklearn `Pipeline` + `RandomizedSearchCV` (stratified 5-fold, training data only) → test-set evaluation → comparison → interpretation → save best model.

## Design decisions
- **Preprocessing** (inside pipelines, refit per fold): inf→NaN, median imputation, zero-variance filter; plus RobustScaler and fixed ±10 clipping for Logistic Regression and Linear SVM only. Outliers are analysed, not removed.
- **Imbalance**: class weights (`balanced`). The code prints the actual ratio and why undersampling, oversampling and SMOTE were not chosen. No resampling, so no test contamination.
- **Models**: Logistic Regression, Decision Tree, Random Forest, Linear SVM (LinearSVC; kernel SVC is impractical at this row count), HistGradientBoosting (runtime-friendly gradient boosting). XGBoost omitted to avoid an extra dependency.
- **Tuning / selection**: scored by PR-AUC (accuracy misleads under imbalance). The final model is chosen by **cross-validated PR-AUC on training data**, so the test set is never used for selection; test metrics are reported alongside.
- **Interpretation**: LR coefficients, tree plot (top 3 levels), RF importances, permutation importance for the selected model. SHAP not used.
- Default 0.5 decision threshold; no threshold tuning.
- Linear SVM has no probabilities; `predict.py` then prints the label only.

## Layout
`src/` (utils, data_loader, eda, preprocessing, train, evaluate, predict) · `notebooks/` · `results/figures` · `results/metrics` · `models/` · `docs/`

## Generated outputs (after `train.py`)
- `results/metrics/`: dataset_inspection.json, feature_summary.csv, class_distribution.csv, summary_statistics.csv, top_correlated_pairs.csv, imbalance_analysis.json, hyperparameters.json, cv_results_*.csv, classification_report_*.txt, confusion_matrix_*.csv, model_comparison.csv, logistic_regression_coefficients.csv, random_forest_feature_importance.csv, permutation_importance_best_model.csv
- `results/figures/`: class_distribution, missing_values (if any missing), correlation_heatmap, feature_distributions, confusion_matrix_*, roc_curves, pr_curves, interpret_*
- `results/sample_input.csv`, `models/best_model.joblib`, `models/model_metadata.json`

Run `predict.py` with the same scikit-learn version used for training (recorded in the metadata).
