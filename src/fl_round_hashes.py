"""Real mini-federation producing per-round SHA-256 digests of the global
model for ledger anchoring (appendix demo). Same clients/data as
train_federated.py, mean aggregation, 3 rounds, reduced negatives for speed.
Output: results/fl_round_hashes.txt (round, sha256, test PR-AUC)."""

import copy
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from model import BridgeMambaKAN

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"
RES = PROJECT / "results"
SEED, ROUNDS, BATCH, LR, MAXN = 42, 3, 512, 2e-3, 4000

torch.manual_seed(SEED)
np.random.seed(SEED)
d = np.load(OUT / "histories.npz", allow_pickle=False)
X, lengths, y = d["X"], d["lengths"], d["y"].astype(np.float32)
tx_hash, bridge_id = d["tx_hash"], d["bridge_id"]
account = d["account"]
splits = dict(np.load(OUT / "splits.npz", allow_pickle=False).items())
ctx = pd.read_parquet(OUT / "context.parquet").set_index("tx_hash")
CTX = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)
usd = pd.read_parquet(OUT / "usd.parquet").set_index("tx_hash")["log_usd"]
CTX = np.hstack([CTX, usd.reindex(tx_hash).fillna(0).to_numpy(np.float32)[:, None]])
L = X.shape[1]
MASK = np.arange(L)[None, :] >= (L - lengths[:, None])

tr = np.flatnonzero(splits["random_train"])
te = np.flatnonzero(splits["random_test"])
rng = np.random.default_rng(SEED)
clients = []
for b in (0, 1):
    idx = tr[bridge_id[tr] == b]
    par = np.array([hash(a) % 2 for a in account[idx]])
    for h, sel in (("A", par == 0), ("B", par == 1)):
        cidx = idx[sel]
        neg = cidx[y[cidx] == 0]
        pos = cidx[y[cidx] == 1]
        if len(neg) > MAXN:
            neg = rng.choice(neg, size=MAXN, replace=False)
        clients.append(np.concatenate([pos, neg]))

allc = np.concatenate(clients)
mu_x, sd_x = X[allc].mean((0, 1)), X[allc].std((0, 1)) + 1e-6
mu_c, sd_c = CTX[allc].mean(0), CTX[allc].std(0) + 1e-6

def tens(idx):
    return (torch.from_numpy((X[idx] - mu_x) / sd_x),
            torch.from_numpy(MASK[idx]),
            torch.from_numpy((CTX[idx] - mu_c) / sd_c),
            torch.from_numpy(y[idx]))

def local(net, idx):
    xt, mt, ct, yt = tens(idx)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    pw = torch.tensor(min((yt == 0).sum().item() / max(yt.sum().item(), 1), 300.0))
    for i in torch.randperm(len(xt)).split(BATCH):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(net(xt[i], mt[i], ct[i]), yt[i],
                                           pos_weight=pw).backward()
        opt.step()
    return net.state_dict()

def sha(state):
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in sorted(state.items())}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()

xe, me, ce, ye = tens(te)
net = BridgeMambaKAN(d_feat=X.shape[2], d_ctx=CTX.shape[1], d_model=32)
lines = []
for r in range(1, ROUNDS + 1):
    states = [local(copy.deepcopy(net), c) for c in clients]
    w = np.array([len(c) for c in clients], float)
    w /= w.sum()
    agg = {k: sum(wi * s[k].float() for wi, s in zip(w, states))
           for k in states[0]}
    net.load_state_dict(agg)
    with torch.no_grad():
        s = torch.cat([torch.sigmoid(net(xe[i:i+2048], me[i:i+2048], ce[i:i+2048]))
                       for i in range(0, len(xe), 2048)]).numpy()
    ap = average_precision_score(ye.numpy(), s)
    h = sha(agg)
    lines.append(f"round={r} sha256={h} test_pr_auc={ap:.4f}")
    print(lines[-1], flush=True)

(RES / "fl_round_hashes.txt").write_text("\n".join(lines) + "\n")
print("wrote results/fl_round_hashes.txt")
