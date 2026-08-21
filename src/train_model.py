"""
E1 (model part) — train BridgeMamba-KAN on the frozen splits.

Usage: python train_model.py [protocol ...]   (default: random temporal)

Class imbalance handled with pos_weight-capped BCE. 10% of train is held
out (stratified) for early stopping on val PR-AUC. Metrics identical to
train_baselines.py; appends to results/model_results.csv.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from model import BridgeMambaKAN

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"
RES = PROJECT / "results"
RES.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 8
BATCH = 512
LR = 2e-3

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

# standardize features on the fly per split (fit on train only)
def tensors(idx, mu_x, sd_x, mu_c, sd_c):
    return (torch.from_numpy((X[idx] - mu_x) / sd_x),
            torch.from_numpy(MASK[idx]),
            torch.from_numpy((CTX[idx] - mu_c) / sd_c),
            torch.from_numpy(y[idx]))


def recall_at_fpr(y_true, score, max_fpr=0.001):
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


@torch.no_grad()
def predict(net, xt, mt, ct):
    net.eval()
    outs = []
    for i in range(0, len(xt), 2048):
        outs.append(torch.sigmoid(net(xt[i:i + 2048], mt[i:i + 2048],
                                      ct[i:i + 2048])))
    return torch.cat(outs).numpy()


def run(protocol: str) -> dict:
    tr_mask, te_mask = splits[f"{protocol}_train"], splits[f"{protocol}_test"]
    tr_idx = np.flatnonzero(tr_mask)
    if y[tr_idx].sum() == 0:
        print(f"[skip] {protocol}: no positives in train")
        return {}
    # stratified 10% val for early stopping
    rng = np.random.default_rng(SEED)
    val_pick = np.zeros(len(tr_idx), dtype=bool)
    for cls in (0, 1):
        c = np.flatnonzero(y[tr_idx] == cls)
        val_pick[rng.choice(c, size=max(1, int(0.1 * len(c))), replace=False)] = True
    va_idx, tr_idx = tr_idx[val_pick], tr_idx[~val_pick]

    # subsample benign train rows for CPU speed (all positives kept;
    # class weighting below adapts to the new ratio)
    MAX_NEG = 50_000
    neg = np.flatnonzero(y[tr_idx] == 0)
    if len(neg) > MAX_NEG:
        drop = np.setdiff1d(neg, rng.choice(neg, size=MAX_NEG, replace=False))
        tr_idx = np.delete(tr_idx, drop)

    mu_x, sd_x = X[tr_idx].mean((0, 1)), X[tr_idx].std((0, 1)) + 1e-6
    mu_c, sd_c = CTX[tr_idx].mean(0), CTX[tr_idx].std(0) + 1e-6
    xt, mt, ct, yt = tensors(tr_idx, mu_x, sd_x, mu_c, sd_c)
    xv, mv, cv, yv = tensors(va_idx, mu_x, sd_x, mu_c, sd_c)
    te_idx = np.flatnonzero(te_mask)
    xe, me, ce, ye = tensors(te_idx, mu_x, sd_x, mu_c, sd_c)

    net = BridgeMambaKAN(d_feat=X.shape[2], d_ctx=CTX.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    pos_weight = torch.tensor(min((yt == 0).sum().item() / max(yt.sum().item(), 1), 300.0))

    best_val, best_state, patience = -1.0, None, 0
    n = len(xt)
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        t0, tot = time.time(), 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            logit = net(xt[b], mt[b], ct[b])
            loss = F.binary_cross_entropy_with_logits(logit, yt[b],
                                                      pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        val_ap = average_precision_score(yv.numpy(), predict(net, xv, mv, cv))
        print(f"  [{protocol}] epoch {epoch + 1}/{EPOCHS} "
              f"loss {tot / n:.4f} val PR-AUC {val_ap:.4f} "
              f"({time.time() - t0:.0f}s)")
        if val_ap > best_val:
            best_val, patience = val_ap, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break
    net.load_state_dict(best_state)

    s = predict(net, xe, me, ce)
    res = {
        "protocol": protocol, "model": "BridgeMambaKAN(cpu-ref)",
        "pr_auc": average_precision_score(ye.numpy(), s),
        "roc_auc": roc_auc_score(ye.numpy(), s),
        "recall@fpr0.1%": recall_at_fpr(ye.numpy(), s),
        "n_test": len(te_idx), "n_test_attack": int(ye.sum().item()),
        "prevalence": float(ye.mean().item()),
    }
    torch.save({"state_dict": best_state,
                "norm": (mu_x, sd_x, mu_c, sd_c)},
               RES / f"bridgemambakan_{protocol}.pt")
    print(f"  [{protocol}] TEST PR-AUC {res['pr_auc']:.4f} "
          f"ROC {res['roc_auc']:.4f} R@0.1%FPR {res['recall@fpr0.1%']:.3f} "
          f"(prevalence {res['prevalence']:.4f})")
    return res


if __name__ == "__main__":
    protocols = sys.argv[1:] or ["random", "temporal"]
    rows = [r for p in protocols if (r := run(p))]
    out = RES / "model_results.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
