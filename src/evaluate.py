"""Evaluation (Step 8), comparison (Step 9) and interpretation (Step 10)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.tree import plot_tree

from utils import (CLASS_NAMES, METRICS_DIR, RANDOM_STATE, save_fig, section, slugify)

LABELS = [CLASS_NAMES[0], CLASS_NAMES[1]]


def get_scores(model, X):
    """Probability of the positive (Bankrupt) class, or the margin if no probabilities exist."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, scores):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, scores),
        "PR-AUC": average_precision_score(y_true, scores),
    }


def evaluate_models(models, X_test, y_test):
    """Evaluate fitted pipelines on the untouched test set. Returns (metrics DataFrame, scores dict)."""
    section("STEP 8 — EVALUATION (held-out test set)")
    rows, all_scores = {}, {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        scores = get_scores(model, X_test)
        all_scores[name] = scores
        rows[name] = compute_metrics(y_test, y_pred, scores)
        slug = slugify(name)

        report = classification_report(y_test, y_pred, target_names=LABELS, zero_division=0)
        (METRICS_DIR / f"classification_report_{slug}.txt").write_text(report)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        pd.DataFrame(cm, index=[f"True {l}" for l in LABELS], columns=[f"Pred {l}" for l in LABELS]) \
            .to_csv(METRICS_DIR / f"confusion_matrix_{slug}.csv")
        print(f"\n--- {name} ---\n{report}Confusion matrix [rows=true, cols=pred] (order: {LABELS}):\n{cm}")

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=LABELS, ax=ax,
                                                colorbar=False, values_format="d")
        ax.set_title(f"Confusion matrix — {name}")
        save_fig(fig, f"confusion_matrix_{slug}.png")

    # ROC and Precision-Recall curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in all_scores.items():
        fpr, tpr, _ = roc_curve(y_test, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={rows[name]['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)"); ax.legend(fontsize=8)
    save_fig(fig, "roc_curves.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in all_scores.items():
        p, r, _ = precision_recall_curve(y_test, s)
        ax.plot(r, p, label=f"{name} (AP={rows[name]['PR-AUC']:.3f})")
    ax.axhline(float(np.mean(y_test)), color="k", ls="--", lw=0.8, label="No-skill (prevalence)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)"); ax.legend(fontsize=8)
    save_fig(fig, "pr_curves.png")

    return pd.DataFrame(rows).T, all_scores


def build_comparison(test_df, searches):
    """Comparison table. Models are RANKED by cross-validated PR-AUC on the training data only, so the
    test set plays no role in model selection; test metrics are reported alongside."""
    section("STEP 9 — MODEL COMPARISON")
    cv = {}
    for name, s in searches.items():
        i = s.best_index_
        cv[name] = {"CV PR-AUC": s.cv_results_["mean_test_average_precision"][i],
                    "CV ROC-AUC": s.cv_results_["mean_test_roc_auc"][i],
                    "CV F1": s.cv_results_["mean_test_f1"][i]}
    table = test_df.join(pd.DataFrame(cv).T)
    table["Rank (CV PR-AUC)"] = table["CV PR-AUC"].rank(ascending=False, method="min").astype(int)
    table = table.sort_values("Rank (CV PR-AUC)")
    table.index.name = "Model"
    table.to_csv(METRICS_DIR / "model_comparison.csv")
    print("Test-set metrics + cross-validation metrics (ranked by CV PR-AUC, training data only):")
    print(table.round(4).to_string())
    print(f"\nSelected model (best CV PR-AUC): {table.index[0]}")
    return table


# ----------------------------------------------------------------- interpretation
def transformed_feature_names(pipe, input_cols):
    try:
        return list(pipe[:-1].get_feature_names_out(np.asarray(input_cols, dtype=object)))
    except Exception:
        n = pipe[:-1].transform(pd.DataFrame(columns=input_cols, data=np.zeros((1, len(input_cols))))).shape[1]
        return [f"f{i}" for i in range(n)]


def _barh(series, title, xlabel, filename, xerr=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(series.index[::-1], series.values[::-1], xerr=None if xerr is None else xerr[::-1], color="#4c72b0")
    ax.set_title(title); ax.set_xlabel(xlabel)
    save_fig(fig, filename)


def interpret_models(models, best_name, X_test, y_test):
    section("STEP 10 — MODEL INTERPRETATION")
    cols = list(X_test.columns)

    if "Logistic Regression" in models:
        pipe = models["Logistic Regression"]
        coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=transformed_feature_names(pipe, cols))
        coef.sort_values(key=np.abs, ascending=False).to_csv(METRICS_DIR / "logistic_regression_coefficients.csv",
                                                             header=["coefficient"])
        top = coef.reindex(coef.abs().sort_values(ascending=False).head(20).index)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(top.index[::-1], top.values[::-1], color=["#c44e52" if v > 0 else "#4c72b0" for v in top.values[::-1]])
        ax.set_title("Logistic Regression — top 20 coefficients (scaled features)\nred: raises Bankrupt odds, blue: lowers")
        ax.set_xlabel("Coefficient")
        save_fig(fig, "interpret_logistic_coefficients.png")

    if "Decision Tree" in models:
        pipe = models["Decision Tree"]
        tree = pipe.named_steps["clf"]
        fig, ax = plt.subplots(figsize=(24, 10))
        plot_tree(tree, max_depth=3, feature_names=transformed_feature_names(pipe, cols), class_names=LABELS,
                  filled=True, rounded=True, fontsize=7, proportion=True, ax=ax)
        ax.set_title(f"Decision Tree — top 3 levels shown (fitted tree depth = {tree.get_depth()})")
        save_fig(fig, "interpret_decision_tree.png")

    if "Random Forest" in models:
        pipe = models["Random Forest"]
        imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=transformed_feature_names(pipe, cols))
        imp = imp.sort_values(ascending=False)
        imp.to_csv(METRICS_DIR / "random_forest_feature_importance.csv", header=["importance"])
        _barh(imp.head(20), "Random Forest — top 20 impurity-based importances", "Mean decrease in impurity",
              "interpret_random_forest_importance.png")

    # Permutation importance of the selected model, on raw input features (test set, post-selection)
    print(f"Computing permutation importance for the selected model ({best_name}) ...")
    r = permutation_importance(models[best_name], X_test, y_test, scoring="average_precision",
                               n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    pi = pd.DataFrame({"importance_mean": r.importances_mean, "importance_std": r.importances_std}, index=cols)
    pi = pi.sort_values("importance_mean", ascending=False)
    pi.to_csv(METRICS_DIR / "permutation_importance_best_model.csv")
    top = pi.head(20)
    _barh(top["importance_mean"], f"Permutation importance — {best_name}\n(drop in PR-AUC when feature is shuffled)",
          "Mean decrease in PR-AUC", "interpret_permutation_importance.png", xerr=top["importance_std"].values)
    print("Top 10 features by permutation importance:")
    print(pi.head(10).round(5).to_string())
