"""Multi-seed BridgeMambaKAN across the four frozen protocols, on GPU if available.

Same architecture, data and metrics as train_model.py; the differences are:
  * every stochastic component is seeded per run, and the run is repeated
  * device is CUDA when present (train_model.py was written CPU-only)
  * writes one row per (protocol, seed) plus a mean +/- 95% CI summary

Rationale: every number in v1 is single-seed. BridgeGuard reports 10 repeats and
BridgeShield 5, so a paper about evaluation rigour cannot report one.

Usage:
  python train_model_multiseed.py --seeds 0 1 2 3 4 --protocols random temporal lobo_ronin lobo_nomad
"""
import argparse
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

EPOCHS = 8
BATCH = 512
LR = 2e-3
MAX_NEG = 50_000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = np.load(OUT / "histories.npz", allow_pickle=False)
X, lengths, y = d["X"], d["lengths"], d["y"].astype(np.float32)
tx_hash = d["tx_hash"]
splits = dict(np.load(OUT / "splits.npz", allow_pickle=False).items())
ctx = pd.read_parquet(OUT / "context.parquet").set_index("tx_hash")
CTX = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)

L = X.shape[1]
MASK = np.arange(L)[None, :] >= (L - lengths[:, None])


def tensors(idx, mu_x, sd_x, mu_c, sd_c):
    return (torch.from_numpy((X[idx] - mu_x) / sd_x).to(DEV),
            torch.from_numpy(MASK[idx]).to(DEV),
            torch.from_numpy((CTX[idx] - mu_c) / sd_c).to(DEV),
            torch.from_numpy(y[idx]).to(DEV))


def recall_at_fpr(y_true, score, max_fpr=0.001):
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


@torch.no_grad()
def predict(net, xt, mt, ct):
    net.eval()
    outs = []
    for i in range(0, len(xt), 4096):
        outs.append(torch.sigmoid(net(xt[i:i + 4096], mt[i:i + 4096], ct[i:i + 4096])))
    return torch.cat(outs).cpu().numpy()


def run(protocol, seed, save_scores=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEV.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    tr_mask, te_mask = splits[f"{protocol}_train"], splits[f"{protocol}_test"]
    tr_idx = np.flatnonzero(tr_mask)
    if y[tr_idx].sum() == 0:
        print(f"[skip] {protocol}: no positives in train")
        return None

    rng = np.random.default_rng(seed)
    val_pick = np.zeros(len(tr_idx), dtype=bool)
    for cls in (0, 1):
        c = np.flatnonzero(y[tr_idx] == cls)
        val_pick[rng.choice(c, size=max(1, int(0.1 * len(c))), replace=False)] = True
    va_idx, tr_idx = tr_idx[val_pick], tr_idx[~val_pick]

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

    net = BridgeMambaKAN(d_feat=X.shape[2], d_ctx=CTX.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    pw = min((yt == 0).sum().item() / max(yt.sum().item(), 1), 300.0)
    pos_weight = torch.tensor(pw, device=DEV)

    best_val, best_state, patience = -1.0, None, 0
    n = len(xt)
    t_train = time.time()
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            logit = net(xt[b], mt[b], ct[b])
            loss = F.binary_cross_entropy_with_logits(logit, yt[b], pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        val_ap = average_precision_score(yv.cpu().numpy(), predict(net, xv, mv, cv))
        if val_ap > best_val:
            best_val, patience = val_ap, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break
    net.load_state_dict(best_state)
    train_s = time.time() - t_train

    t_inf = time.time()
    s = predict(net, xe, me, ce)
    inf_ms = 1000.0 * (time.time() - t_inf) / max(len(te_idx), 1)

    yn = ye.cpu().numpy()
    prev = float(yn.mean())
    pr = average_precision_score(yn, s)
    roc = roc_auc_score(yn, s)
    if save_scores:
        np.savez_compressed(save_scores, y=yn, score=s, tx_hash=tx_hash[te_idx])
    return dict(protocol=protocol, seed=seed, model="BridgeMambaKAN",
                pr_auc=pr, chance=prev, lift=pr / prev if prev else np.nan,
                roc_auc=roc, rho=2 * roc - 1,
                recall_at_fpr=recall_at_fpr(yn, s),
                n_test=len(te_idx), n_pos=int(yn.sum()),
                train_s=round(train_s, 1), inf_ms_per_tx=round(inf_ms, 4),
                device=DEV.type)


def ci95(v):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return 0.0
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def main(seeds, protocols):
    print(f"device={DEV}  samples={len(X):,}  feat={X.shape[2]}  ctx={CTX.shape[1]}")
    print(f"seeds={seeds}  protocols={protocols}\n", flush=True)
    rows = []
    for protocol in protocols:
        for seed in seeds:
            t = time.time()
            sc = RES / f"scores_{protocol}_seed{seed}.npz" if seed == seeds[0] else None
            r = run(protocol, seed, save_scores=sc)
            if r is None:
                break
            rows.append(r)
            print(f"  {protocol:<12} seed {seed}  PR-AUC {r['pr_auc']:.4f} "
                  f"(chance {r['chance']:.5f}, {r['lift']:.1f}x)  "
                  f"rho {r['rho']:+.3f}  [{time.time()-t:.0f}s]", flush=True)
            pd.DataFrame(rows).to_csv(RES / "model_multiseed.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "model_multiseed.csv", index=False)
    print(f"\nwrote {RES / 'model_multiseed.csv'}\n")
    print("=" * 78)
    print("mean +/- 95% CI over seeds")
    for p, g in df.groupby("protocol", sort=False):
        print(f"  {p:<12} PR-AUC {g.pr_auc.mean():.4f} +/- {ci95(g.pr_auc):.4f}   "
              f"chance {g.chance.iloc[0]:.5f}   "
              f"rho {g.rho.mean():+.3f} +/- {ci95(g.rho):.3f}   "
              f"(n={len(g)})")
    print(f"\ninference: {df.inf_ms_per_tx.mean():.4f} ms/tx on {DEV.type}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--protocols", nargs="+",
                    default=["random", "temporal", "lobo_ronin", "lobo_nomad"])
    a = ap.parse_args()
    main(a.seeds, a.protocols)
