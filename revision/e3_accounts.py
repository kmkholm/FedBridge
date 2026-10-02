"""E3 -- Table 5 exploiter-account detection on Across, repeated over seeds, with
operational (label-free) operating points and paired model comparisons.

Data, features, split and model settings are those of src/account_detection.py
(imported, not copied). BridgeMamba-KAN runs on the GPU with the same settings
(d_model 48, 10 epochs, batch 512, AdamW 2e-3, pos_weight cap 100).

Operating points: v1 flagged exactly as many accounts as there are test exploiters, which
needs the test labels. V2 reports precision/recall at fixed alert budgets k in {100, 250, 500}
accounts, chosen a priori as analyst queue sizes, and keeps the matched-count point only as a
retrospective reference.

writes results/e3_runs.csv, e3_paired.csv, e3_scores/seed{n}.npz
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")   # leave CPU headroom for the GPU stream
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parents[1] / "src"
RES = Path(__file__).resolve().parent / "results"
SC = RES / "e3_scores"
SC.mkdir(parents=True, exist_ok=True)
CACHE = RES / "e3_accounts_cache.npz"
SEEDS = [42] + list(range(9))   # 42 = seed of the original submission
BUDGETS = (100, 250, 500)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("acct", SRC / "account_detection.py")
acct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acct)
from model import BridgeMambaKAN  # noqa: E402  (src/model.py, same file as in v1)


def build():
    if CACHE.exists():
        z = np.load(CACHE)
        return z["X"], z["M"], z["C"], z["y"], z["tr"]
    d = acct.load()
    d, cols = acct.per_tx_features(d)
    meta, X, M, C, _ = acct.build_accounts(d, cols)
    y = meta.y.to_numpy(np.float32)
    tr = (meta.first_t <= meta.first_t.quantile(0.7)).to_numpy()
    np.savez_compressed(CACHE, X=X, M=M, C=C, y=y, tr=tr)
    return X, M, C, y, tr


def train_kan(seed, Xtr, Mtr, Ctr, ytr, Xte, Mte, Cte, epochs=10, bs=512):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    net = BridgeMambaKAN(d_feat=Xtr.shape[2], d_ctx=Ctr.shape[1], d_model=48).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    pw = torch.tensor(float(min((ytr == 0).sum() / max(ytr.sum(), 1), 100.0)), device=DEV)
    xt, mt, ct, yt = (torch.from_numpy(a).to(DEV) for a in (Xtr, Mtr, Ctr, ytr))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(xt[b], mt[b], ct[b]), yt[b], pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 4096):
            sl = slice(i, i + 4096)
            out.append(torch.sigmoid(net(*(torch.from_numpy(a[sl]).to(DEV) for a in (Xte, Mte, Cte)))).cpu())
    return torch.cat(out).numpy()


def rec_at_fpr(y, s, f=0.001):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def op_points(y, s):
    order = np.argsort(-s, kind="stable")
    r = {}
    for k in BUDGETS:
        tp = y[order[:k]].sum()
        r[f"p@{k}"], r[f"r@{k}"] = tp / k, tp / y.sum()
    k = int(y.sum())                                   # retrospective, label-matched count
    tp = y[order[:k]].sum()
    r["p_matched"] = r["r_matched"] = r["f1_matched"] = tp / k
    return r


def main():
    X, M, C, y, tr = build()
    te = ~tr
    print(f"accounts {len(y):,} | train {tr.sum():,} ({int(y[tr].sum())} exploiters) | "
          f"test {te.sum():,} ({int(y[te].sum())} exploiters) | device {DEV}", flush=True)
    Cs = StandardScaler().fit(C[tr]).transform(C).astype(np.float32)
    mu = np.nanmean(X[tr].reshape(-1, X.shape[2]), 0)
    sd = np.nanstd(X[tr].reshape(-1, X.shape[2]), 0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    yte = y[te]

    out = RES / "e3_runs.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["seed"] for r in rows}
    for seed in SEEDS:
        if seed in done:
            continue
        t = time.time()
        s = {"BridgeMamba-KAN": train_kan(seed, Xs[tr], M[tr], Cs[tr], y[tr], Xs[te], M[te], Cs[te]),
             "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=6,
                                                    random_state=seed).fit(Cs[tr], y[tr]).predict_proba(Cs[te])[:, 1],
             "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
                 .fit(Cs[tr], y[tr]).predict_proba(Cs[te])[:, 1],
             "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced")
                 .fit(Cs[tr], y[tr]).predict_proba(Cs[te])[:, 1]}
        np.savez_compressed(SC / f"seed{seed}.npz", y=yte, **{k.replace("-", "_"): v for k, v in s.items()})
        for name, sc in s.items():
            rows.append(dict(seed=seed, model=name, pr_auc=ap(yte, sc), roc_auc=roc_auc_score(yte, sc),
                             recall_at_fpr=rec_at_fpr(yte, sc), **op_points(yte, sc),
                             n_test=len(yte), test_pos=int(yte.sum()), prevalence=float(yte.mean())))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"seed {seed} ({time.time() - t:.0f}s): " +
              "  ".join(f"{r['model'][:6]} {r['pr_auc']:.3f}" for r in rows[-4:]), flush=True)

    runs = pd.DataFrame(rows)
    print(runs.groupby("model").agg(["mean", "std"]).round(4).drop(columns=["seed", "n_test", "test_pos", "prevalence"],
                                                                   level=0).T.to_string(), flush=True)

    # paired comparison BridgeMamba-KAN vs each baseline: Wilcoxon over seeds and a paired
    # bootstrap of test accounts for the seed-averaged PR-AUC difference
    rng = np.random.default_rng(2026)
    Z = [np.load(SC / f"seed{s}.npz") for s in SEEDS]
    pos, neg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    boots = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(1000)]
    paired = []
    for other in ["RandomForest", "HistGradientBoosting", "LogisticRegression"]:
        a = runs[runs.model == "BridgeMamba-KAN"].sort_values("seed").pr_auc.to_numpy()
        b = runs[runs.model == other].sort_values("seed").pr_auc.to_numpy()
        w = wilcoxon(a, b) if np.any(a != b) else None
        diffs = [np.mean([ap(yte[i], z["BridgeMamba_KAN"][i]) - ap(yte[i], z[other][i]) for z in Z]) for i in boots]
        paired.append(dict(comparison=f"BridgeMamba-KAN vs {other}", mean_diff=float(np.mean(a - b)),
                           wins=int((a > b).sum()), n=len(a), wilcoxon_p=w.pvalue if w else np.nan,
                           boot_lo=float(np.percentile(diffs, 2.5)), boot_hi=float(np.percentile(diffs, 97.5))))
        print(paired[-1], flush=True)
    pd.DataFrame(paired).to_csv(RES / "e3_paired.csv", index=False)


if __name__ == "__main__":
    main()
