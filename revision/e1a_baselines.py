"""E1a -- Table 3 classical baselines, exact v1 configuration, repeated over seeds.

Identical to src/train_baselines.py (StandardScaler fitted on train,
LogisticRegression(class_weight="balanced"), HistGradientBoosting(class_weight="balanced"))
except that the HGB random_state is the seed. Logistic regression is deterministic, so
its spread across seeds is zero by construction.

usage: python e1a_baselines.py --seeds 0 1 2 3 4 5 6 7 8 9
writes results/e1a_baselines.csv (one row per protocol x features x model x seed)
"""
import argparse
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

DATA = Path(os.environ.get("FB_DATA", Path(__file__).resolve().parents[1] / "processed")).expanduser()
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)

d = np.load(DATA / "histories.npz", allow_pickle=False)
X_seq, lengths, y, tx_hash = d["X"], d["lengths"], d["y"], d["tx_hash"]
splits = dict(np.load(DATA / "splits.npz", allow_pickle=False).items())

final_step = X_seq[:, -1, :]
m3 = (np.arange(X_seq.shape[1])[None, :] >= (X_seq.shape[1] - lengths[:, None]))[:, :, None]
hist_mean = (X_seq * m3).sum(1) / np.maximum(m3.sum(1), 1)
hist_max = np.where(m3, X_seq, -np.inf).max(1)
hist_max[~np.isfinite(hist_max)] = 0.0
ctx = pd.read_parquet(DATA / "context.parquet").set_index("tx_hash")
ctx = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)

FEATURES = {
    "tx": final_step,
    "tx+ctx": np.hstack([final_step, ctx]),
    "tx+ctx+h": np.hstack([final_step, ctx, hist_mean, hist_max]),
}
MODELS = {
    "logreg": lambda s: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "hgb": lambda s: HistGradientBoostingClassifier(class_weight="balanced", random_state=s),
}


def recall_at_fpr(yt, s, max_fpr=0.001):
    fpr, tpr, _ = roc_curve(yt, s)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def main(seeds, protocols):
    out = RES / "e1a_baselines.csv"
    rows = []
    for proto in protocols:
        tr, te = splits[f"{proto}_train"], splits[f"{proto}_test"]
        for fname, X in FEATURES.items():
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            for mname, make in MODELS.items():
                for seed in seeds:
                    if mname == "logreg" and seed != seeds[0] and rows and rows[-1]["model"] == "logreg":
                        r = dict(rows[-1]); r["seed"] = seed; rows.append(r); continue
                    t = time.time()
                    s = make(seed).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
                    rows.append(dict(protocol=proto, features=fname, model=mname, seed=seed,
                                     pr_auc=average_precision_score(y[te], s),
                                     roc_auc=roc_auc_score(y[te], s),
                                     recall_at_fpr=recall_at_fpr(y[te], s),
                                     chance=float(y[te].mean()), n_test=int(te.sum()),
                                     n_pos=int(y[te].sum()), secs=round(time.time() - t, 1)))
                v = [r["pr_auc"] for r in rows if (r["protocol"], r["features"], r["model"]) == (proto, fname, mname)]
                print(f"{proto:11s} {fname:9s} {mname:7s} PR-AUC {np.mean(v):.4f} +/- {np.std(v, ddof=1) if len(v) > 1 else 0:.4f} (n={len(v)})", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42] + list(range(9)))
    ap.add_argument("--protocols", nargs="+", default=["random", "temporal", "lobo_ronin", "lobo_nomad"])
    a = ap.parse_args()
    main(a.seeds, a.protocols)
