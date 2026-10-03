"""Dataset loading (directly from UCI), cleaning and inspection (Step 1)."""
import numpy as np
import pandas as pd

from utils import CLASS_NAMES, METRICS_DIR, save_json, section


def load_data():
    """Fetch the UCI Polish Companies Bankruptcy dataset (id=365) -- no manual CSV."""
    from ucimlrepo import fetch_ucirepo

    polish_companies_bankruptcy = fetch_ucirepo(id=365)
    X = polish_companies_bankruptcy.data.features
    y = polish_companies_bankruptcy.data.targets

    md = polish_companies_bankruptcy.metadata
    print(f"Source: UCI Machine Learning Repository (id=365) | name: {md.get('name')}")
    print(f"        {md.get('repository_url')}")

    if y.shape[1] != 1:
        raise ValueError(f"Expected one target column, got {list(y.columns)}")
    target_name = y.columns[0]
    y = pd.to_numeric(y.iloc[:, 0], errors="raise")
    if y.isna().any():
        raise ValueError("Target contains missing values.")
    if not set(y.unique()) <= {0, 1}:
        raise ValueError(f"Target labels must be {{0,1}} (1 = bankrupt); found {sorted(y.unique())}")
    y = y.astype(int).reset_index(drop=True)

    raw_dtypes = X.dtypes.astype(str)
    X_num = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    coerced = int(X_num.isna().sum().sum() - X.isna().sum().sum())
    if coerced:
        print(f"WARNING: {coerced} non-numeric feature values were coerced to NaN.")

    info = {"target_name": target_name, "raw_dtypes": raw_dtypes, "coerced_to_nan": coerced,
            "source": "UCI id=365"}
    return X_num, y, info


def clean_data(X, y):
    """Drop exact duplicate rows (features + target) so a row cannot land in both train and test."""
    section("DATA CLEANING")
    dup = pd.concat([X, y.rename("__target__")], axis=1).duplicated(keep="first").values
    n_dup = int(dup.sum())
    print(f"Exact duplicate rows (features + target) removed: {n_dup}")
    X, y = X[~dup].reset_index(drop=True), y[~dup].reset_index(drop=True)
    print(f"Shape after cleaning: {X.shape}")
    return X, y, n_dup


def inspect_dataset(X, y, info):
    section("STEP 1 — DATASET INSPECTION")
    n_rows, n_feat = X.shape
    raw = info["raw_dtypes"]
    report = {"rows": n_rows, "features": n_feat}

    print(f"Rows: {n_rows:,}")
    print(f"Feature columns: {n_feat} (+ target '{info['target_name']}' = {n_feat + 1} columns total)")
    print("\nFeature names and raw dtypes:\n  " + ", ".join(f"{c}:{raw[c]}" for c in X.columns))
    print(f"Dtype summary: {raw.value_counts().to_dict()}")

    # Target
    counts = y.value_counts().sort_index()
    print(f"\nTarget column: {info['target_name']}")
    for lab, c in counts.items():
        print(f"  class {lab} ({CLASS_NAMES[lab]}): {c:,} ({c / n_rows * 100:.2f}%)")
    report["class_counts"] = {CLASS_NAMES[k]: int(v) for k, v in counts.items()}

    # Missing
    miss = X.isna().sum()
    total_missing = int(miss.sum())
    print(f"\nTotal missing values: {total_missing:,} "
          f"({total_missing / X.size * 100:.2f}% of cells); "
          f"rows with >=1 missing: {int(X.isna().any(axis=1).sum()):,}")
    print(f"Features with any missing: {int((miss > 0).sum())}/{n_feat}")
    top_miss = miss[miss > 0].sort_values(ascending=False).head(15)
    if len(top_miss):
        print("Top features by missing count (full table in feature_summary.csv):")
        print(top_miss.to_frame("n_missing").assign(pct=lambda d: (d.n_missing / n_rows * 100).round(2)).to_string())
    arr = X.to_numpy(dtype=float)
    n_inf = int(np.isinf(arr).sum())
    print(f"Infinite values: {n_inf}")
    report.update(total_missing=total_missing, infinite_values=n_inf)

    # Duplicates
    dup_feat = int(X.duplicated().sum())
    dup_full = int(pd.concat([X, y.rename("__t__")], axis=1).duplicated().sum())
    print(f"\nDuplicate rows: {dup_full} (features+target), {dup_feat} (features only)")
    report.update(duplicate_rows_full=dup_full, duplicate_rows_features_only=dup_feat)

    # Constant / near-constant
    nunique = X.nunique(dropna=True)
    top_share = X.apply(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan)
    constant = list(nunique[nunique <= 1].index)
    near_const = list(top_share[(top_share >= 0.99) & (nunique > 1)].index)
    print(f"Constant features: {constant if constant else 'none'}")
    print(f"Near-constant features (>=99% one value): {near_const if near_const else 'none'}")
    report.update(constant_features=constant, near_constant_features=near_const)

    # Outliers (IQR rule) -- analysis only, nothing is removed
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    ok = iqr[iqr > 0].index
    mask = X[ok].lt(q1[ok] - 1.5 * iqr[ok], axis=1) | X[ok].gt(q3[ok] + 1.5 * iqr[ok], axis=1)
    out_frac = mask.mean()
    per_class = mask.sum(axis=1).groupby(y.values).mean()
    print("\nOUTLIER ANALYSIS (1.5*IQR rule; analysis only, nothing removed)")
    print(f"  Median share of values flagged per feature: {out_frac.median() * 100:.2f}%")
    print(f"  Features with >10% flagged values: {int((out_frac > 0.10).sum())}/{len(ok)}")
    print("  Top 10 features by flagged share (%):")
    print((out_frac.sort_values(ascending=False).head(10) * 100).round(2).to_string())
    print(f"  Largest |value| in any feature: {np.abs(arr[np.isfinite(arr)]).max():.4g}")
    print("  Mean number of outlier-flagged features per row, by class:")
    for lab, v in per_class.items():
        print(f"    {CLASS_NAMES[lab]}: {v:.2f}")
    if per_class.get(1, 0) > per_class.get(0, 0):
        print("  Finding: bankrupt firms are flagged in MORE features per row, so extremes carry class signal.")
    else:
        print("  Finding: bankrupt firms do not show a higher outlier load in this data.")
    print("  Decision: outliers are KEPT (financial ratios are heavy-tailed). Their influence is handled in "
          "the pipeline via median imputation, RobustScaler + fixed clipping (scale-sensitive models) and "
          "tree-based models.")
    report["outlier_median_share"] = float(out_frac.median())
    report["outlier_features_per_row_by_class"] = {CLASS_NAMES[k]: float(v) for k, v in per_class.items()}

    table = pd.DataFrame({
        "raw_dtype": raw, "n_missing": miss, "pct_missing": miss / n_rows * 100,
        "n_unique": nunique, "top_value_share": top_share,
        "outlier_fraction_iqr": out_frac.reindex(X.columns),
        "skewness": X.skew(), "min": X.min(), "max": X.max()})
    table.to_csv(METRICS_DIR / "feature_summary.csv")
    save_json(report, METRICS_DIR / "dataset_inspection.json")
    return report
