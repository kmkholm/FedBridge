"""
E1 (baseline part) — classical models on tabular features, all frozen splits.

Feature sets:
  tx        final-step per-transaction features (14)
  tx+ctx    + bridge-level context (10)                       <- deployable now
  tx+ctx+h  + mean/max-pooled account history (28)            <- pooled, no seq model

Models: logistic regression, histogram gradient boosting.
Metrics: PR-AUC (average precision), ROC-AUC, recall@FPR<=0.1%.
Writes results/baselines.csv and prints the table.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"
RES = PROJECT / "results"
RES.mkdir(exist_ok=True)

d = np.load(OUT / "histories.npz", allow_pickle=False)
X_seq, lengths, y = d["X"], d["lengths"], d["y"]
tx_hash = d["tx_hash"]
splits = dict(np.load(OUT / "splits.npz", allow_pickle=False).items())

# final step = current tx; pooled history for the "+h" variant
final_step = X_seq[:, -1, :]
mask = (np.arange(X_seq.shape[1])[None, :] >= (X_seq.shape[1] - lengths[:, None]))
m3 = mask[:, :, None]
hist_mean = (X_seq * m3).sum(1) / np.maximum(m3.sum(1), 1)
hist_max = np.where(m3, X_seq, -np.inf).max(1)
hist_max[~np.isfinite(hist_max)] = 0.0

ctx = pd.read_parquet(OUT / "context.parquet").set_index("tx_hash")
ctx = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)

FEATURES = {
    "tx": final_step,
    "tx+ctx": np.hstack([final_step, ctx]),
    "tx+ctx+h": np.hstack([final_step, ctx, hist_mean, hist_max]),
}
MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "hgb": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
}
PROTOCOLS = ["random", "temporal", "lobo_ronin", "lobo_nomad"]


def recall_at_fpr(y_true, score, max_fpr=0.001):
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


rows = []
for proto in PROTOCOLS:
    tr, te = splits[f"{proto}_train"], splits[f"{proto}_test"]
    if y[tr].sum() == 0:
        print(f"[skip] {proto}: no positive train samples for supervised model")
        continue
    for fname, X in FEATURES.items():
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        for mname, make in MODELS.items():
            clf = make().fit(Xtr, y[tr])
            s = clf.predict_proba(Xte)[:, 1]
            rows.append({
                "protocol": proto, "features": fname, "model": mname,
                "pr_auc": average_precision_score(y[te], s),
                "roc_auc": roc_auc_score(y[te], s),
                "recall@fpr0.1%": recall_at_fpr(y[te], s),
                "n_test": int(te.sum()), "n_test_attack": int(y[te].sum()),
            })
            print(f"{proto:11s} {fname:9s} {mname:7s} "
                  f"PR-AUC {rows[-1]['pr_auc']:.4f}  ROC {rows[-1]['roc_auc']:.4f}  "
                  f"R@0.1%FPR {rows[-1]['recall@fpr0.1%']:.3f}")

res = pd.DataFrame(rows)
res.to_csv(RES / "baselines.csv", index=False)
print(f"\nwrote {RES / 'baselines.csv'}")
