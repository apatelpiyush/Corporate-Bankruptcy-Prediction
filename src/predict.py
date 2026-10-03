"""Predict bankruptcy for new companies using the saved model.

Examples (run from anywhere):
  python src/predict.py --template my_input.csv      # writes an empty CSV with the required columns
  python src/predict.py --csv results/sample_input.csv
  python src/predict.py --json '{"X1": 0.05, "X2": 0.4, ...}'   # or a path to a .json file
Missing values (blank / null) are allowed; they are imputed by the saved pipeline.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from utils import MODELS_DIR


def load_artifacts():
    model_path, meta_path = MODELS_DIR / "best_model.joblib", MODELS_DIR / "model_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        sys.exit("Saved model not found. Run `python src/train.py` first.")
    return joblib.load(model_path), json.loads(meta_path.read_text())


def predict_frame(df, model, meta):
    cols = meta["feature_names"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns ({len(missing)}): {missing[:10]}"
                         f"{' ...' if len(missing) > 10 else ''}")
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return pred, proba


def format_prediction(pred, proba, meta, row=None):
    names = {int(k): v for k, v in meta["class_names"].items()}
    lines = [f"--- Row {row} ---"] if row is not None else []
    lines.append(f"Prediction: {names[int(pred)]}")
    if proba is not None:
        lines += [f"Bankrupt Probability: {proba * 100:.1f}%", f"Non-Bankrupt Probability: {(1 - proba) * 100:.1f}%"]
    else:
        lines.append(f"(Model '{meta['model_name']}' does not output probabilities.)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="CSV with one row per company; columns must include all feature names")
    g.add_argument("--json", help="JSON string or .json file mapping feature name -> value (one company)")
    g.add_argument("--template", help="write an empty CSV template with the required columns and exit")
    a = ap.parse_args()

    model, meta = load_artifacts()
    if a.template:
        pd.DataFrame(columns=meta["feature_names"]).to_csv(a.template, index=False)
        print(f"Template written to {a.template}")
        return
    if a.csv:
        df = pd.read_csv(a.csv)
    else:
        p = Path(a.json)
        data = json.loads(p.read_text() if p.exists() else a.json)
        df = pd.DataFrame([data])

    pred, proba = predict_frame(df, model, meta)
    print(f"Model: {meta['model_name']}")
    for i, p in enumerate(pred):
        print(format_prediction(p, None if proba is None else proba[i], meta, row=i if len(pred) > 1 else None))
    print("\nNote: probabilities are model estimates from historical data, not certainty.")


if __name__ == "__main__":
    main()
