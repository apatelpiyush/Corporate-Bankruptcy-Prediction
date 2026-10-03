"""Shared constants and helpers for the Corporate Bankruptcy Prediction project."""
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless backend for script runs
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER = 6                      # max sampled candidates per model in RandomizedSearchCV
TUNING_METRIC = "average_precision"   # PR-AUC; used to refit/select (see README)
CLASS_NAMES = {0: "Non-Bankrupt", 1: "Bankrupt"}
POSITIVE_CLASS = 1


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    for d in (FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_fig(fig, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"  [saved figure] {path.relative_to(ROOT)}")
    return path


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  [saved] {Path(path).relative_to(ROOT)}")


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
