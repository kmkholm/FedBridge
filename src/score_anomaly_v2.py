"""
E1b iteration 2 — density-based zero-day scoring on the TRAINED AE encoders.

Loads each bridgemamba_ae_<protocol>.pt checkpoint (no retraining), extracts
the encoder latent (concat of masked-mean and last-step states, 96-d), and
scores test samples with:

  mahal    Mahalanobis distance to the benign-train latent Gaussian
  knn      mean distance to k=10 nearest benign-train latents
           (on a 20k benign subsample for CPU speed)
  ctx_gmm  Mahalanobis on the 10 raw context features only (no deep model —
           the "is the deep encoder even needed" reference)

Appends to results/model_results.csv (models: BridgeMambaMahal, BridgeMambaKNN,
CtxGaussian). Usage: python -u score_anomaly_v2.py [protocol ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from train_anomaly import SeqAE  # reuse architecture

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"
RES = PROJECT / "results"

SEED = 42
MAX_FIT = 60_000
KNN_FIT = 20_000
K = 10

d = np.load(OUT / "histories.npz", allow_pickle=False)
X, lengths, y = d["X"], d["lengths"], d["y"].astype(np.float32)
tx_hash = d["tx_hash"]
splits = dict(np.load(OUT / "splits.npz", allow_pickle=False).items())
ctx = pd.read_parquet(OUT / "context.parquet").set_index("tx_hash")
CTX = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)
L = X.shape[1]
MASK = np.arange(L)[None, :] >= (L - lengths[:, None])
TARGET = np.hstack([X[:, -1, :], CTX])


@torch.no_grad()
def latents(net, idx, mu_x, sd_x, batch=2048):
    zs = []
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        x = torch.from_numpy((X[b] - mu_x) / sd_x)
        m = torch.from_numpy(MASK[b])
        h = net.embed(x)
        for blk in net.blocks:
            h = blk(h)
        h = net.norm(h)
        mm = m.unsqueeze(-1).float()
        mean = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
        zs.append(torch.cat([mean, h[:, -1]], dim=-1))
    return torch.cat(zs).numpy().astype(np.float64)


def mahal_scores(fit, query):
    mu = fit.mean(0)
    cov = np.cov(fit, rowvar=False) + 1e-3 * np.eye(fit.shape[1])
    prec = np.linalg.inv(cov)
    dlt = query - mu
    return np.einsum("ij,jk,ik->i", dlt, prec, dlt)


def knn_scores(fit, query, k=K, block=1024):
    out = np.empty(len(query))
    fit_sq = (fit ** 2).sum(1)
    for i in range(0, len(query), block):
        q = query[i:i + block]
        d2 = (q ** 2).sum(1)[:, None] + fit_sq[None, :] - 2 * q @ fit.T
        part = np.partition(d2, k, axis=1)[:, :k]
        out[i:i + block] = part.mean(1)
    return out


def recall_at_fpr(y_true, score, max_fpr=0.001):
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def evaluate(name, proto, ye, s, rows):
    r = {"protocol": proto, "model": name,
         "pr_auc": average_precision_score(ye, s),
         "roc_auc": roc_auc_score(ye, s),
         "recall@fpr0.1%": recall_at_fpr(ye, s),
         "n_test": len(ye), "n_test_attack": int(ye.sum()),
         "prevalence": float(ye.mean())}
    rows.append(r)
    print(f"  [{proto}] {name:18s} PR-AUC {r['pr_auc']:.4f} "
          f"ROC {r['roc_auc']:.4f} R@0.1%FPR {r['recall@fpr0.1%']:.3f}",
          flush=True)


def run(proto, rows):
    ck_path = RES / f"bridgemamba_ae_{proto}.pt"
    if not ck_path.exists():
        print(f"[skip] {proto}: no checkpoint", flush=True)
        return
    ck = torch.load(ck_path, weights_only=False)
    mu_x, sd_x, _, _ = ck["norm"]
    net = SeqAE(X.shape[2], TARGET.shape[1])
    net.load_state_dict(ck["state_dict"])
    net.eval()

    rng = np.random.default_rng(SEED)
    ben = np.flatnonzero(splits[f"{proto}_train"] & (y == 0))
    if len(ben) > MAX_FIT:
        ben = rng.choice(ben, size=MAX_FIT, replace=False)
    te_idx = np.flatnonzero(splits[f"{proto}_test"])
    ye = y[te_idx]

    z_fit = latents(net, ben, mu_x, sd_x)
    z_te = latents(net, te_idx, mu_x, sd_x)
    evaluate("BridgeMambaMahal", proto, ye, mahal_scores(z_fit, z_te), rows)

    sub = z_fit[rng.choice(len(z_fit), size=min(KNN_FIT, len(z_fit)),
                           replace=False)]
    evaluate("BridgeMambaKNN", proto, ye, knn_scores(sub, z_te), rows)

    mu_c = CTX[ben].mean(0)
    sd_c = CTX[ben].std(0) + 1e-6
    cfit = (CTX[ben] - mu_c) / sd_c
    cte = (CTX[te_idx] - mu_c) / sd_c
    evaluate("CtxGaussian", proto, ye, mahal_scores(cfit, cte), rows)


if __name__ == "__main__":
    protocols = sys.argv[1:] or ["temporal", "lobo_ronin", "lobo_nomad", "random"]
    rows = []
    for p in protocols:
        run(p, rows)
    out = RES / "model_results.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
