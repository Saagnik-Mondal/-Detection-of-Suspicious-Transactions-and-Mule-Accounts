"""
Score new transactions/accounts with the trained mule-detection model.

Produces, per row: a fraud-risk probability, a HOLD/REVIEW/PASS action based on
the tuned threshold, and the top SHAP reasons explaining the score
(the "why was this flagged" needed for analyst review & audit).

Usage:
    python score.py <input.csv> [--out scored.csv] [--threshold 0.476]

Input must contain the same feature columns used in training (extra columns are
ignored, missing ones are filled as NaN — XGBoost handles missing natively).
"""

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb


def load_model():
    model = xgb.XGBClassifier()
    model.load_model("model.json")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


def action(p, thr):
    if p >= thr:
        return "HOLD"
    if p >= thr / 3:          # grey zone -> queue for analyst
        return "REVIEW"
    return "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="scored.csv")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    model, meta = load_model()
    feats = meta["features"]
    thr = args.threshold if args.threshold is not None else meta["threshold"]

    df = pd.read_csv(args.input, index_col=0, low_memory=False)
    X = df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")

    proba = model.predict_proba(X.values)[:, 1]

    # per-row top-3 SHAP reasons
    import shap
    sv = shap.TreeExplainer(model).shap_values(X.values)
    reasons = []
    for i in range(len(X)):
        order = np.argsort(np.abs(sv[i]))[::-1][:3]
        reasons.append("; ".join(
            f"{feats[j]}({'+' if sv[i, j] > 0 else '-'}{abs(sv[i, j]):.2f})"
            for j in order
        ))

    out = pd.DataFrame({
        "risk_score": proba.round(4),
        "action": [action(p, thr) for p in proba],
        "top_reasons": reasons,
    }, index=df.index)

    out.to_csv(args.out)
    n_hold = (out.action == "HOLD").sum()
    n_rev = (out.action == "REVIEW").sum()
    print(f"Scored {len(out):,} rows at threshold {thr:.3f}: "
          f"{n_hold} HOLD, {n_rev} REVIEW, {len(out) - n_hold - n_rev} PASS")
    print(f"Wrote {args.out}")
    print("\nHighest-risk rows:")
    print(out.sort_values("risk_score", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
