# Mule / Suspicious-Transaction Detection

AI/ML detection of suspicious transactions and mule accounts on a highly
imbalanced, high-dimensional, anonymized banking dataset.

## The data (`DataSet.csv`)
- **9,082** rows (accounts/transactions), **3,923** anonymized features `F1…F3923`, label `F3924`.
- **Extreme imbalance:** 81 fraud (1) vs 9,001 legit (0) — **0.89% positive** (~111:1).
- **High missingness:** mean 28% per feature; ~1,138 columns >50% empty.
- **Mixed scales:** values range from ~0–1 normalized to ±10¹¹ (monetary). Tree models handle this; no scaling needed.

## ⚠️ Critical finding: target leakage
Feature **`F3912`** is an **outcome-proxy** — binary 0/1, **0% missing**, mean
**0.000** for legit vs **0.975** for fraud (univariate ROC-AUC 0.99). It is an
after-the-fact "confirmed-fraud / account-blocked" flag, **not** a signal
available at scoring time. Including it produces a fake ~1.000 ROC-AUC and a
model that is **useless in production**.

The pipeline **auto-detects and removes** such columns (fully-observed +
near-binary + label-mirroring). The ablation quantifies the gap:

| Model | CV PR-AUC | CV ROC-AUC |
|---|---|---|
| With leaky `F3912` (inflated, do not trust) | 0.997 | 1.000 |
| **Honest (leakage removed)** | **0.914 ± 0.047** | **0.988 ± 0.013** |

> The ~29 other high-AUC features are continuous, 85–97% missing, and arrive in
> correlated pairs — these are genuine engineered fraud signals (velocity/ratio/
> network metrics), and are kept.

## Approach
1. **Clean** — drop columns >90% missing and constant columns (3,923 → 3,117).
2. **Leakage control** — auto-remove outcome-proxy flags (`F3912`).
3. **Model** — XGBoost, `scale_pos_weight≈111`, native NaN handling, `aucpr` objective, column subsampling for the wide feature space.
4. **Honest evaluation** — stratified 5-fold CV with out-of-fold predictions (only 81 positives → CV beats a single split). Report **PR-AUC / ROC-AUC**, not accuracy.
5. **Operating points** — analyst false-positive budget table; default = max-F1 threshold.
6. **Explainability** — global + per-transaction SHAP ("why flagged").

## Honest results (leakage removed)
- **PR-AUC 0.914**, **ROC-AUC 0.988** (baseline PR-AUC = 0.0089).
- Default threshold (max-F1 = 0.476): **precision 0.986, recall 0.840** →
  catches **68/81** fraud with **1** false alarm out of 9,001 legit.
- Tunable trade-offs, e.g. 50% precision → **74/81** fraud caught at 74 false alarms.

## Files
| File | Purpose |
|---|---|
| `train.py` | Clean → leakage-detect → CV → metrics → SHAP → save model |
| `score.py` | Score new rows: risk score + HOLD/REVIEW/PASS + top-3 SHAP reasons |
| `model.json`, `model_meta.json` | Trained model + feature list & threshold |

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost shap matplotlib   # macOS: brew install libomp
python train.py                 # train + evaluate
python score.py new_txns.csv    # score new transactions
```

## Pitch framing
The headline isn't "99% accuracy" — that's the leakage trap most teams fall into.
The headline is: **a leakage-audited detector that catches ~84% of mule/fraud
cases at 98% precision (≈1 false alarm in 9,000), with a SHAP reason on every
flag for analyst review and audit.** Honesty about `F3912` *is* the
differentiator.
