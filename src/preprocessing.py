"""Preprocessing pipelines (Step 5) and class-imbalance analysis (Step 3)."""
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from utils import CLASS_NAMES, section

CLIP_VALUE = 10.0  # fixed constant (not learned from data) applied after robust scaling


def _inf_to_nan(X):
    X = np.array(X, dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X


def _clip(X):
    return np.clip(X, -CLIP_VALUE, CLIP_VALUE)


def build_preprocessor(scale: bool) -> Pipeline:
    """Leakage-safe preprocessing; every learned step is fitted on training data only
    (the pipeline is refit inside each CV fold).

    inf -> NaN      : infinities (ratios with zero denominators) cannot be imputed/scaled.
    median impute   : financial ratios are heavy-tailed, the median is robust; the dataset has missing values.
    variance filter : drops features that are constant in the training data (no information).
    RobustScaler    : (scale=True) median/IQR scaling, robust to outliers; needed by LR / linear SVM.
    clip +-10       : (scale=True) bounds extreme robust-scaled values so a few points cannot dominate
                      the linear models. Not applied to tree models (scale-invariant, split on ranks).
    """
    steps = [
        ("inf_to_nan", FunctionTransformer(_inf_to_nan, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]
    if scale:
        steps += [("scaler", RobustScaler()),
                  ("clip", FunctionTransformer(_clip, feature_names_out="one-to-one"))]
    return Pipeline(steps)


def analyze_imbalance(y):
    section("STEP 3 — CLASS IMBALANCE ANALYSIS")
    counts = y.value_counts()
    minority, majority = counts.idxmin(), counts.idxmax()
    n_min, n_maj, n = int(counts.min()), int(counts.max()), len(y)
    ratio = n_maj / n_min
    print(f"Minority: {CLASS_NAMES[minority]} = {n_min:,} | Majority: {CLASS_NAMES[majority]} = {n_maj:,}")
    print(f"Imbalance ratio (majority:minority) = {ratio:.1f}:1")
    print(f"A trivial 'always {CLASS_NAMES[majority]}' classifier would reach {n_maj / n * 100:.2f}% accuracy "
          "-> accuracy is misleading; PR-AUC / recall / F1 are emphasised.\n")

    print("Options evaluated against these numbers:")
    print(f"  class weights        : no data change; reweights loss by ~{ratio:.1f}x for the minority class.")
    print(f"  random undersampling : would discard {n_maj - n_min:,} majority rows ({(n_maj - n_min) / n * 100:.1f}% of data).")
    print(f"  random oversampling  : would duplicate minority rows ~{ratio:.1f}x -> overfitting risk.")
    print(f"  SMOTE                : would synthesise ~{n_maj - n_min:,} points by interpolation in a heavy-tailed, "
          "highly correlated ratio space; costly and can create unrealistic firms.")
    strategy = "class_weight"
    print(f"\nCHOSEN: {strategy}. The minority class has {n_min:,} real examples, so reweighting is sufficient, "
          "keeps all real data, adds no synthetic rows, and cannot leak into the test set.")
    if n_min < 500:
        print("NOTE: minority class is small (<500); consider SMOTE inside an imblearn Pipeline "
              "(training folds only) -- not implemented here.")
    return {"minority_class": CLASS_NAMES[minority], "minority_count": n_min, "majority_count": n_maj,
            "imbalance_ratio": float(ratio), "strategy": strategy}
