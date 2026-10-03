"""Training / tuning / selection pipeline. Run from anywhere:  python src/train.py"""
import argparse
import platform
import time

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import clean_data, inspect_dataset, load_data
from eda import run_eda
from evaluate import build_comparison, evaluate_models, interpret_models
from preprocessing import analyze_imbalance, build_preprocessor
from utils import (CLASS_NAMES, CV_FOLDS, METRICS_DIR, MODELS_DIR, N_ITER, RANDOM_STATE, ROOT, TEST_SIZE,
                   TUNING_METRIC, ensure_dirs, save_json, section, set_seed, slugify)

SCORING = {"average_precision": "average_precision", "roc_auc": "roc_auc", "f1": "f1"}


def get_model_specs():
    """Models, why they are included, and (deliberately small) search spaces. All use class weighting."""
    rs = RANDOM_STATE
    return {
        "Logistic Regression": dict(
            estimator=LogisticRegression(class_weight="balanced", max_iter=2000, random_state=rs),
            scale=True, grid={"clf__C": [0.01, 0.1, 1, 10]},
            why="Interpretable linear baseline; coefficients show direction/size of effects."),
        "Decision Tree": dict(
            estimator=DecisionTreeClassifier(class_weight="balanced", random_state=rs),
            scale=False, grid={"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [5, 20, 50]},
            why="Transparent rule-based model; captures non-linearity; depth/leaf limits control overfitting."),
        "Random Forest": dict(
            estimator=RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=rs),
            scale=False, grid={"clf__max_depth": [8, 16, None], "clf__min_samples_leaf": [1, 5, 10],
                               "clf__max_features": ["sqrt", 0.3]},
            why="Bagged trees: reduces tree variance, handles interactions, gives feature importances."),
        "Linear SVM": dict(
            estimator=LinearSVC(class_weight="balanced", dual=False, max_iter=5000, random_state=rs),
            scale=True, grid={"clf__C": [0.01, 0.1, 1]},
            why="Margin-based classifier. LinearSVC (not kernel SVC) is used because kernel SVC training "
                "scales poorly with tens of thousands of rows. Provides margins, not probabilities."),
        "Gradient Boosting": dict(
            estimator=HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=rs),
            scale=False, grid={"clf__learning_rate": [0.03, 0.1], "clf__max_iter": [200, 400],
                               "clf__max_leaf_nodes": [15, 31], "clf__l2_regularization": [0.0, 1.0]},
            why="Boosted trees (sklearn histogram-based implementation, chosen over GradientBoostingClassifier "
                "for runtime on tens of thousands of rows); typically strong on tabular data."),
    }


def split_data(X, y):
    section("STEP 4 — STRATIFIED TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    for nm, yy in (("train", y_train), ("test", y_test)):
        c = yy.value_counts().sort_index()
        print(f"{nm}: {len(yy):,} rows | " + ", ".join(f"{CLASS_NAMES[k]}={v:,} ({v / len(yy) * 100:.2f}%)" for k, v in c.items()))
    print("Test set is not touched until the final evaluation. Imputation/scaling/etc. are fitted inside "
          "pipelines on training data only (refit per CV fold).")
    return X_train, X_test, y_train, y_test


def make_cv(n_splits=CV_FOLDS):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def tune_all(specs, X_train, y_train, cv, n_iter=N_ITER):
    section("STEPS 5-7 — PREPROCESSING PIPELINES + HYPERPARAMETER TUNING (training data only)")
    print(f"RandomizedSearchCV | {cv.get_n_splits()}-fold stratified CV | refit/select on '{TUNING_METRIC}' "
          f"(PR-AUC) | max {n_iter} candidates per model")
    searches = {}
    for name, spec in specs.items():
        print(f"\n>>> {name}\n    why: {spec['why']}")
        pipe = Pipeline([("prep", build_preprocessor(spec["scale"])), ("clf", spec["estimator"])])
        n_cand = min(n_iter, len(ParameterGrid(spec["grid"])))
        search = RandomizedSearchCV(pipe, spec["grid"], n_iter=n_cand, scoring=SCORING, refit=TUNING_METRIC,
                                    cv=cv, n_jobs=-1, random_state=RANDOM_STATE, error_score="raise")
        t0 = time.time()
        search.fit(X_train, y_train)
        i = search.best_index_
        print(f"    best params: {search.best_params_}")
        print(f"    CV PR-AUC={search.cv_results_['mean_test_average_precision'][i]:.4f} "
              f"ROC-AUC={search.cv_results_['mean_test_roc_auc'][i]:.4f} "
              f"F1={search.cv_results_['mean_test_f1'][i]:.4f} | {time.time() - t0:.0f}s")
        pd.DataFrame(search.cv_results_).to_csv(METRICS_DIR / f"cv_results_{slugify(name)}.csv", index=False)
        searches[name] = search
    save_json({n: {"fixed_hyperparameters": specs[n]["estimator"].get_params(deep=False),
                   "search_space": specs[n]["grid"], "scaled": specs[n]["scale"],
                   "best_params": s.best_params_, "why": specs[n]["why"]} for n, s in searches.items()},
              METRICS_DIR / "hyperparameters.json")
    return searches


def save_final_model(best_name, models, searches, comparison, X_train):
    section("STEP 11 — SAVE FINAL MODEL")
    pipe = models[best_name]
    path = MODELS_DIR / "best_model.joblib"
    joblib.dump(pipe, path)
    meta = {
        "model_name": best_name,
        "selection_rule": "highest cross-validated PR-AUC on training data",
        "feature_names": list(X_train.columns),
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "has_predict_proba": hasattr(pipe, "predict_proba"),
        "best_params": searches[best_name].best_params_,
        "test_metrics": comparison.loc[best_name].to_dict(),
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
    }
    save_json(meta, MODELS_DIR / "model_metadata.json")
    print(f"  [saved model] {path.relative_to(ROOT)}  ({best_name})")
    return path


def save_sample_input(X_test, y_test, n=5):
    out = X_test.head(n).copy()
    out["true_class"] = y_test.loc[out.index].values   # ignored by predict.py
    p = ROOT / "results" / "sample_input.csv"
    out.to_csv(p, index=False)
    print(f"  [saved] {p.relative_to(ROOT)} ({n} held-out test rows for trying predict.py)")


def run_pipeline(cv_folds=CV_FOLDS, n_iter=N_ITER):
    set_seed(); ensure_dirs()
    X, y, info = load_data()
    inspect_dataset(X, y, info)
    X, y, _ = clean_data(X, y)
    run_eda(X, y)
    imbalance = analyze_imbalance(y)
    save_json(imbalance, METRICS_DIR / "imbalance_analysis.json")
    X_train, X_test, y_train, y_test = split_data(X, y)
    searches = tune_all(get_model_specs(), X_train, y_train, make_cv(cv_folds), n_iter)
    models = {n: s.best_estimator_ for n, s in searches.items()}
    test_df, _ = evaluate_models(models, X_test, y_test)
    comparison = build_comparison(test_df, searches)
    best_name = comparison.index[0]
    interpret_models(models, best_name, X_test, y_test)
    save_final_model(best_name, models, searches, comparison, X_train)
    save_sample_input(X_test, y_test)
    section("DONE — see results/ and models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and evaluate bankruptcy prediction models.")
    ap.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    ap.add_argument("--n-iter", type=int, default=N_ITER, help="max sampled candidates per model")
    a = ap.parse_args()
    run_pipeline(a.cv_folds, a.n_iter)
