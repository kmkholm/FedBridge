"""
E3/E4 — federated training prototype (CPU reference).

Clients = bridge x time-half partitions of the random_train split
(4 clients, naturally non-IID: attack supervision is concentrated in the
nomad clients; ronin-late holds the 2 Ronin exploit txs).

E3: FedAvg vs centralized reference on identical data.
E4: one client poisoned (label flip), aggregation mean vs coordinate-wise
    median vs 20% trimmed mean.

Usage: python -u train_federated.py            # E3 clean run, all aggregators
       python -u train_federated.py --poison   # E4, poison client 0
Appends to results/federated.csv.
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from model import BridgeMambaKAN

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"
RES = PROJECT / "results"

SEED = 42
ROUNDS = 8
LOCAL_EPOCHS = 2
BATCH = 512
LR = 2e-3
MAX_NEG_PER_CLIENT = 8_000
POISON = "--poison" in sys.argv

torch.manual_seed(SEED)
np.random.seed(SEED)

d = np.load(OUT / "histories.npz", allow_pickle=False)
X, lengths, y = d["X"], d["lengths"], d["y"].astype(np.float32)
tx_hash, bridge_id, ts = d["tx_hash"], d["bridge_id"], d["timestamp"]
splits = dict(np.load(OUT / "splits.npz", allow_pickle=False).items())
ctx = pd.read_parquet(OUT / "context.parquet").set_index("tx_hash")
CTX = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)
usd = pd.read_parquet(OUT / "usd.parquet").set_index("tx_hash")["log_usd"]
CTX = np.hstack([CTX, usd.reindex(tx_hash).fillna(0)
                 .to_numpy(dtype=np.float32)[:, None]])
account = d["account"]

L = X.shape[1]
MASK = np.arange(L)[None, :] >= (L - lengths[:, None])

# ---- clients: bridge x time-half of random_train
tr = np.flatnonzero(splits["random_train"])
te = np.flatnonzero(splits["random_test"])
rng = np.random.default_rng(SEED)

clients = []
for b in (0, 1):
    idx = tr[bridge_id[tr] == b]
    parity = np.array([hash(a) % 2 for a in account[idx]])
    for half, sel in (("A", parity == 0), ("B", parity == 1)):
        cidx = idx[sel]
        neg = cidx[y[cidx] == 0]
        pos = cidx[y[cidx] == 1]
        if len(neg) > MAX_NEG_PER_CLIENT:
            neg = rng.choice(neg, size=MAX_NEG_PER_CLIENT, replace=False)
        clients.append({"name": f"{'ronin' if b == 0 else 'nomad'}-{half}",
                        "idx": np.concatenate([pos, neg])})

# global normalization from union of client data (server-side public stats)
all_tr = np.concatenate([c["idx"] for c in clients])
mu_x, sd_x = X[all_tr].mean((0, 1)), X[all_tr].std((0, 1)) + 1e-6
mu_c, sd_c = CTX[all_tr].mean(0), CTX[all_tr].std(0) + 1e-6

def tens(idx, labels=None):
    yy = labels if labels is not None else y[idx]
    return (torch.from_numpy((X[idx] - mu_x) / sd_x),
            torch.from_numpy(MASK[idx]),
            torch.from_numpy((CTX[idx] - mu_c) / sd_c),
            torch.from_numpy(yy))

for c in clients:
    labels = y[c["idx"]].copy()
    if POISON and c is clients[0]:
        labels = 1.0 - labels                       # label-flip poisoning
        c["name"] += "(POISONED)"
    c["tensors"] = tens(c["idx"], labels)
    print(f"client {c['name']:22s} n={len(c['idx']):6,} "
          f"pos={int(labels.sum()):4d}", flush=True)

xe, me, ce, ye = tens(te)


def local_train(net, tensors):
    xt, mt, ct, yt = tensors
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    pw = torch.tensor(min((yt == 0).sum().item() / max(yt.sum().item(), 1), 300.0))
    n = len(xt)
    for _ in range(LOCAL_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(xt[b], mt[b], ct[b]),
                                                      yt[b], pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net.state_dict()


def aggregate(states, sizes, how):
    out = {}
    w = np.array(sizes, dtype=np.float64)
    w /= w.sum()
    for k in states[0]:
        stack = torch.stack([s[k].float() for s in states])
        if how == "mean":
            out[k] = (stack * torch.tensor(w, dtype=torch.float32)
                      .view(-1, *[1] * (stack.dim() - 1))).sum(0)
        elif how == "median":
            out[k] = stack.median(0).values
        elif how == "trimmed":
            k_trim = max(1, int(0.2 * len(states)))
            srt = stack.sort(0).values
            out[k] = srt[k_trim:len(states) - k_trim].mean(0) \
                if len(states) > 2 * k_trim else stack.mean(0)
    return out


@torch.no_grad()
def evaluate(net):
    net.eval()
    outs = []
    for i in range(0, len(xe), 2048):
        outs.append(torch.sigmoid(net(xe[i:i + 2048], me[i:i + 2048],
                                      ce[i:i + 2048])))
    s = torch.cat(outs).numpy()
    return average_precision_score(ye.numpy(), s), roc_auc_score(ye.numpy(), s)


rows = []
for how in ("mean", "median", "trimmed"):
    torch.manual_seed(SEED)
    global_net = BridgeMambaKAN(d_feat=X.shape[2], d_ctx=CTX.shape[1], d_model=32)
    for rnd in range(1, ROUNDS + 1):
        t0 = time.time()
        states, sizes = [], []
        for c in clients:
            local = copy.deepcopy(global_net)
            states.append(local_train(local, c["tensors"]))
            sizes.append(len(c["idx"]))
        global_net.load_state_dict(aggregate(states, sizes, how))
        ap, roc = evaluate(global_net)
        print(f"[{'E4-poison' if POISON else 'E3-clean'}|{how}] round {rnd}/{ROUNDS} "
              f"PR-AUC {ap:.4f} ROC {roc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    rows.append({"experiment": "E4-poison" if POISON else "E3-clean",
                 "aggregator": how, "rounds": ROUNDS,
                 "pr_auc": ap, "roc_auc": roc})

out = RES / "federated.csv"
df = pd.DataFrame(rows)
if out.exists():
    df = pd.concat([pd.read_csv(out), df], ignore_index=True)
df.to_csv(out, index=False)
print(f"wrote {out}", flush=True)
