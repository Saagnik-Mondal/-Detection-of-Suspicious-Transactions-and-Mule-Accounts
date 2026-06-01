"""
Mule / suspicious-transaction detection on DataSet.csv

A highly-imbalanced (0.89% positive), high-dimensional (~3900 anonymized
features), high-missingness tabular fraud problem.

Pipeline:
  1. Load + clean (drop >90%-missing columns, drop constant columns)
  2. Stratified 5-fold CV with XGBoost (native NaN handling, scale_pos_weight)
     -> out-of-fold (OOF) predictions for an honest, low-variance estimate
        despite only 81 positives.
  3. Imbalance-appropriate metrics: PR-AUC, ROC-AUC, recall@precision,
     confusion matrix at a chosen operating point.
  4. Operating-point / threshold table so an analyst can pick a precision/
     recall trade-off.
  5. Refit on all data, dump SHAP global feature importance + the model.

Run:  python train.py
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb

warnings.filterwarnings("ignore")
RNG = 42
DATA = "DataSet.csv"
MISSING_DROP_THRESHOLD = 0.90
N_FOLDS = 5

LEAK_MAXMISS = 0.02
LEAK_MINAUC = 0.97
LEAK_MAXUNIQUE = 3


def load_and_clean():
    df = pd.read_csv(DATA, index_col=0, low_memory=False)
    label = df.columns[-1]
    y = df[label].astype(int).values
    X = df.drop(columns=[label])

    X = X.apply(pd.to_numeric, errors="coerce")

    n0 = X.shape[1]
    keep_missing = X.isna().mean() <= MISSING_DROP_THRESHOLD
    X = X.loc[:, keep_missing]
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]

    print(f"  rows                : {X.shape[0]:,}")
    print(f"  positives / negatives: {int(y.sum())} / {int((1 - y).sum())} "
          f"({100 * y.mean():.2f}% positive)")
    print(f"  features: {n0:,} -> {X.shape[1]:,} after pruning "
          f"(dropped {n0 - X.shape[1]:,})")

    leaks = detect_leakage(X, y)
    if leaks:
        print(f"\n  [LEAKAGE] outcome-proxy features removed: {leaks}")
        X = X.drop(columns=leaks)
        print(f"  features after leakage removal: {X.shape[1]:,}")
    return X, y, list(X.columns), leaks


def detect_leakage(X, y):
    """Flag fully-observed, near-binary, label-mirroring columns."""
    leaks = []
    for c in X.columns:
        s = X[c]
        if s.isna().mean() > LEAK_MAXMISS:
            continue
        if s.nunique(dropna=True) > LEAK_MAXUNIQUE:
            continue
        auc = roc_auc_score(y, s.fillna(s.median()))
        if max(auc, 1 - auc) >= LEAK_MINAUC:
            leaks.append(c)
    return leaks


def make_model(y_train):
    pos = max(int(y_train.sum()), 1)
    neg = int((1 - y_train).sum())
    spw = neg / pos
    return xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=2,
        reg_lambda=2.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RNG,
    )


def cross_validate(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y))
    fold_ap, fold_auc = [], []
    Xv = X.values

    for k, (tr, va) in enumerate(skf.split(Xv, y), 1):
        model = make_model(y[tr])
        model.fit(Xv[tr], y[tr])
        p = model.predict_proba(Xv[va])[:, 1]
        oof[va] = p
        ap, auc = average_precision_score(y[va], p), roc_auc_score(y[va], p)
        fold_ap.append(ap)
        fold_auc.append(auc)
        print(f"  fold {k}: PR-AUC={ap:.3f}  ROC-AUC={auc:.3f}  "
              f"(val pos={int(y[va].sum())})")

    print(f"  CV mean PR-AUC = {np.mean(fold_ap):.3f} ± {np.std(fold_ap):.3f}")
    print(f"  CV mean ROC-AUC= {np.mean(fold_auc):.3f} ± {np.std(fold_auc):.3f}")
    return oof


def operating_points(y, oof):
    ap = average_precision_score(y, oof)
    auc = roc_auc_score(y, oof)
    print(f"\n  Overall OOF  PR-AUC={ap:.3f}   ROC-AUC={auc:.3f}   "
          f"(baseline PR-AUC = positive rate = {y.mean():.4f})")

    prec, rec, thr = precision_recall_curve(y, oof)
    print("\n  Operating points (pick analyst false-positive budget):")
    print(f"  {'target prec':>11} | {'thresh':>7} | {'precision':>9} | "
          f"{'recall':>6} | {'frauds caught':>13} | {'false alarms':>12}")
    P = int(y.sum())
    N = int((1 - y).sum())
    for target in (0.10, 0.25, 0.50, 0.75, 0.90):
        mask = prec[:-1] >= target
        if not mask.any():
            print(f"  {target:>11.0%} |   (not achievable)")
            continue
        idx = np.where(mask)[0]
        best = idx[np.argmax(rec[:-1][idx])]
        t = thr[best]
        pr, rc = prec[best], rec[best]
        tp = int(round(rc * P))
        fp = int(round(tp / pr - tp)) if pr > 0 else N
        print(f"  {target:>11.0%} | {t:>7.3f} | {pr:>9.3f} | {rc:>6.3f} | "
              f"{tp:>3}/{P:<9} | {fp:>12}")

    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    bi = int(np.argmax(f1))
    t = thr[bi]
    print(f"\n  Recommended default (max-F1) threshold = {t:.3f}  "
          f"(precision={prec[bi]:.3f}, recall={rec[bi]:.3f}, F1={f1[bi]:.3f})")
    yhat = (oof >= t).astype(int)
    cm = confusion_matrix(y, yhat)
    print("\n  Confusion matrix at recommended threshold:")
    print(f"            pred legit   pred fraud")
    print(f"  legit  {cm[0,0]:>11} {cm[0,1]:>12}")
    print(f"  fraud  {cm[1,0]:>11} {cm[1,1]:>12}")
    print("\n" + classification_report(y, yhat, digits=3,
                                        target_names=["legit", "fraud"]))
    return float(t)


def finalize(X, y, feat_names, threshold):
    print("\n[5] Refitting on all data + SHAP importance ...")
    model = make_model(y)
    model.fit(X.values, y)
    model.save_model("model.json")

    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X.values)
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:20]
    print("\n  Top 20 features by mean |SHAP| (drivers of fraud score):")
    for r, i in enumerate(order, 1):
        print(f"   {r:>2}. {feat_names[i]:<8} {imp[i]:.4f}")

    with open("model_meta.json", "w") as f:
        json.dump(
            {
                "features": feat_names,
                "threshold": threshold,
                "top_features": [feat_names[i] for i in order],
                "n_features": len(feat_names),
            },
            f,
            indent=2,
        )
    print("\n  Saved: model.json, model_meta.json")


def main():
    print("[1] Loading + cleaning ...")
    X, y, feat_names, leaks = load_and_clean()

    print("\n[2-3] Stratified 5-fold CV (XGBoost) — HONEST model (leakage removed) ...")
    oof = cross_validate(X, y)

    print("\n[4] Threshold / operating-point analysis (honest model) ...")
    threshold = operating_points(y, oof)

    finalize(X, y, feat_names, threshold)

    if leaks:
        print("\n[ABLATION] Same model WITH the leaky outcome flag(s) "
              f"{leaks} added back:")
        df = pd.read_csv(DATA, index_col=0, low_memory=False)
        Xl = X.copy()
        for c in leaks:
            Xl[c] = pd.to_numeric(df[c], errors="coerce")
        oof_leak = cross_validate(Xl, y)
        print(f"  -> PR-AUC {average_precision_score(y, oof_leak):.3f} / "
              f"ROC-AUC {roc_auc_score(y, oof_leak):.3f}  "
              f"(vs honest PR-AUC {average_precision_score(y, oof):.3f}). "
              "The gap is the leakage.")
    print("\nDone.")


if __name__ == "__main__":
    main()
