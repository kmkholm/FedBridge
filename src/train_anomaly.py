"""
E1b — benign-only anomaly detection with the shared BiMamba encoder.

Train an autoencoder on BENIGN train samples only: BiMamba encodes the
account-history sequence; an MLP decoder reconstructs the current
transaction's feature vector [final-step feats (14) + bridge context (10)].
Anomaly score = per-feature-standardized reconstruction MSE.
No attack labels are used at any point -> valid for zero-day protocols.

Usage: python -u train_anomaly.py [protocol ...]
       (default: temporal lobo_ronin lobo_nomad random)
Appends to results/model_results.csv (model = BridgeMambaAE).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from model import BiMamba

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"
RES = PROJECT / "results"

SEED = 42
EPOCHS = 6
BATCH = 512
LR = 2e-3
MAX_TRAIN = 60_000

torch.manual_seed(SEED)
np.random.seed(SEED)

d = np.load(OUT / "histories.npz", allow_pickle=False)
X, lengths, y = d["X"], d["lengths"], d["y"].astype(np.float32)
tx_hash = d["tx_hash"]
splits = dict(np.load(OUT / "splits.npz", allow_pickle=False).items())
ctx = pd.read_parquet(OUT / "context.parquet").set_index("tx_hash")
CTX = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)

L = X.shape[1]
MASK = np.arange(L)[None, :] >= (L - lengths[:, None])
TARGET = np.hstack([X[:, -1, :], CTX])          # what the decoder reconstructs


class SeqAE(nn.Module):
    def __init__(self, d_feat, d_target, d_model=48, n_layers=2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d_feat, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(BiMamba(d_model) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.dec = nn.Sequential(nn.Linear(2 * d_model, 64), nn.SiLU(),
                                 nn.Linear(64, d_target))

    def forward(self, x, mask):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.dec(torch.cat([mean, h[:, -1]], dim=-1))


def recall_at_fpr(y_true, score, max_fpr=0.001):
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def run(protocol):
    tr_mask, te_mask = splits[f"{protocol}_train"], splits[f"{protocol}_test"]
    rng = np.random.default_rng(SEED)
    ben = np.flatnonzero(tr_mask & (y == 0))          # benign-only training
    if len(ben) > MAX_TRAIN:
        ben = rng.choice(ben, size=MAX_TRAIN, replace=False)
    te_idx = np.flatnonzero(te_mask)

    mu_x, sd_x = X[ben].mean((0, 1)), X[ben].std((0, 1)) + 1e-6
    mu_t, sd_t = TARGET[ben].mean(0), TARGET[ben].std(0) + 1e-6

    def tens(idx):
        return (torch.from_numpy((X[idx] - mu_x) / sd_x),
                torch.from_numpy(MASK[idx]),
                torch.from_numpy((TARGET[idx] - mu_t) / sd_t))

    xt, mt, tt = tens(ben)
    net = SeqAE(X.shape[2], TARGET.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)

    n = len(ben)
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.mse_loss(net(xt[b], mt[b]), tt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        print(f"  [{protocol}] epoch {epoch + 1}/{EPOCHS} "
              f"recon {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)

    net.eval()
    xe, me, te_t = tens(te_idx)
    errs = []
    with torch.no_grad():
        for i in range(0, len(te_idx), 2048):
            r = net(xe[i:i + 2048], me[i:i + 2048])
            errs.append(((r - te_t[i:i + 2048]) ** 2).mean(-1))
    s = torch.cat(errs).numpy()
    ye = y[te_idx]
    res = {"protocol": protocol, "model": "BridgeMambaAE(benign-only)",
           "pr_auc": average_precision_score(ye, s),
           "roc_auc": roc_auc_score(ye, s),
           "recall@fpr0.1%": recall_at_fpr(ye, s),
           "n_test": len(te_idx), "n_test_attack": int(ye.sum()),
           "prevalence": float(ye.mean())}
    torch.save({"state_dict": net.state_dict(),
                "norm": (mu_x, sd_x, mu_t, sd_t)},
               RES / f"bridgemamba_ae_{protocol}.pt")
    print(f"  [{protocol}] TEST PR-AUC {res['pr_auc']:.4f} "
          f"ROC {res['roc_auc']:.4f} R@0.1%FPR {res['recall@fpr0.1%']:.3f} "
          f"(prevalence {res['prevalence']:.4f})", flush=True)
    return res


if __name__ == "__main__":
    protocols = sys.argv[1:] or ["temporal", "lobo_ronin", "lobo_nomad", "random"]
    rows = [run(p) for p in protocols]
    out = RES / "model_results.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
