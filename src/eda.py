"""Exploratory data analysis (Step 2). Descriptive only -- nothing here is used to fit models."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import CLASS_NAMES, METRICS_DIR, save_fig, section


def run_eda(X, y):
    section("STEP 2 — EXPLORATORY DATA ANALYSIS")
    n = len(y)

    # Target analysis
    counts = y.value_counts().sort_index()
    tgt = pd.DataFrame({"class": [CLASS_NAMES[i] for i in counts.index], "count": counts.values,
                        "percent": (counts.values / n * 100).round(3)})
    tgt.to_csv(METRICS_DIR / "class_distribution.csv", index=False)
    print(tgt.to_string(index=False))
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(tgt["class"], tgt["count"], color=["#4c72b0", "#c44e52"])
    for b, c, p in zip(bars, tgt["count"], tgt["percent"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c:,}\n({p:.2f}%)", ha="center", va="bottom")
    ax.set_title("Class distribution"); ax.set_ylabel("Companies")
    ax.set_ylim(0, tgt["count"].max() * 1.18)
    save_fig(fig, "class_distribution.png")

    # Summary statistics
    summary = X.describe().T
    summary["median"] = X.median()
    summary["skew"] = X.skew()
    summary.to_csv(METRICS_DIR / "summary_statistics.csv")
    print(f"\nSummary statistics saved ({summary.shape[0]} features). Most skewed features (|skew|):")
    print(summary["skew"].abs().sort_values(ascending=False).head(5).round(2).to_string())

    # Missing values
    miss_pct = (X.isna().mean() * 100).sort_values(ascending=False)
    if (miss_pct > 0).any():
        top = miss_pct[miss_pct > 0].head(25)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top.index[::-1], top.values[::-1], color="#dd8452")
        ax.set_xlabel("% missing"); ax.set_title("Missing values (top features)")
        save_fig(fig, "missing_values.png")
    else:
        print("No missing values -> missing-value figure skipped.")

    # Correlations (Spearman: robust to the heavy tails of financial ratios)
    corr = X.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(np.ma.masked_invalid(corr.values), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(corr.columns, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title("Feature correlation heatmap (Spearman)")
    save_fig(fig, "correlation_heatmap.png")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index)
    pairs.head(30).rename("spearman_rho").to_csv(METRICS_DIR / "top_correlated_pairs.csv")
    print(f"\nFeature pairs with |rho| >= 0.9: {int((pairs.abs() >= 0.9).sum())} "
          f"(top pairs in top_correlated_pairs.csv)")

    # Distributions: 12 features most associated with the target (display choice only)
    assoc = X.corrwith(y, method="spearman").abs().sort_values(ascending=False).head(12)
    fig, axes = plt.subplots(4, 3, figsize=(13, 12))
    for ax, col in zip(axes.ravel(), assoc.index):
        lo, hi = X[col].quantile([0.01, 0.99])
        for lab, color in ((0, "#4c72b0"), (1, "#c44e52")):
            vals = X.loc[y == lab, col].dropna().clip(lo, hi)
            ax.hist(vals, bins=40, range=(lo, hi), density=True, alpha=0.55, color=color, label=CLASS_NAMES[lab])
        ax.set_title(f"{col} (|rho| with target = {assoc[col]:.2f})", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Feature distributions by class (clipped to 1st-99th percentile for display)")
    fig.tight_layout()
    save_fig(fig, "feature_distributions.png")
