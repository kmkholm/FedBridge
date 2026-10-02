"""E2b -- BridgeMamba-KAN as the learned triage ranker (GPU), seeds 0-9.

Same alert universe, directions, USD prior and fusion rules as e2_triage.py; the learned score
comes from BridgeMamba-KAN over the full 32-step history plus the ten context features and the
USD value. Two positives cannot support a validation split, so the model trains for a fixed
eight epochs (otherwise the v1 settings: batch 512, AdamW 2e-3, weight decay 1e-4, pos_weight
cap 300, gradient clip 1.0, d_model 48).

writes results/e2b_runs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score as ap, roc_auc_score as roc

DATA = Path(__file__).resolve().parents[1] / "processed"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import BridgeMambaKAN  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS, EPOCHS, BATCH, LR = [42] + list(range(9)), 8, 512, 2e-3

d = np.load(DATA / "histories.npz", allow_pickle=False)
X, lengths, y = d["X"], d["lengths"], d["y"].astype(np.float32)
susp, tx_hash, bridge_id, account = d["suspicious"].astype(bool), d["tx_hash"], d["bridge_id"], d["account"]
ctx = pd.read_parquet(DATA / "context.parquet").set_index("tx_hash")
USD = pd.read_parquet(DATA / "usd.parquet").set_index("tx_hash")["log_usd"].reindex(tx_hash).fillna(0) \
    .to_numpy(dtype=np.float32)
CTX = np.hstack([ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32), USD[:, None]])
MASK = np.arange(X.shape[1])[None, :] >= (X.shape[1] - lengths[:, None])
flagged = susp | (y == 1)
DIRS = {"ronin_to_nomad": (0, 1), "nomad_to_ronin": (1, 0)}


def pct(s):
    return pd.Series(s).rank(pct=True, method="average").to_numpy()


def metrics(yt, s, acc):
    r = np.sort(np.argsort(np.argsort(-s, kind="stable"), kind="stable")[yt == 1] + 1)
    a = pd.DataFrame({"a": acc, "y": yt, "s": s}).groupby("a").agg(y=("y", "max"), s=("s", "max"))
    return dict(ap=ap(yt, s), roc=roc(yt, s), first_rank=int(r[0]), hits_top8=int((r <= 8).sum()),
                hits_top20=int((r <= 20).sum()), hits_top100=int((r <= 100).sum()),
                ranks=" ".join(map(str, r[:8])), addr_ap=ap(a.y, a.s),
                addr_hits_top20=int(a.sort_values("s", ascending=False).y.head(20).sum()))


def train_score(tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    mu_x, sd_x = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-6
    mu_c, sd_c = CTX[tr].mean(0), CTX[tr].std(0) + 1e-6
    T = lambda i: tuple(torch.from_numpy(a).to(DEV) for a in ((X[i] - mu_x) / sd_x, MASK[i], (CTX[i] - mu_c) / sd_c))
    xt, mt, ct = T(tr)
    yt = torch.from_numpy(y[tr]).to(DEV)
    net = BridgeMambaKAN(d_feat=X.shape[2], d_ctx=CTX.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    pw = torch.tensor(min((yt == 0).sum().item() / max(yt.sum().item(), 1), 300.0), device=DEV)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(net(xt[b], mt[b], ct[b]), yt[b], pos_weight=pw).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    xe, me, ce = T(te)
    with torch.no_grad():
        return torch.cat([torch.sigmoid(net(xe[i:i + 4096], me[i:i + 4096], ce[i:i + 4096]))
                          for i in range(0, len(xe), 4096)]).cpu().numpy()


rows = []
for dname, (btr, bte) in DIRS.items():
    tr, te = np.flatnonzero(flagged & (bridge_id == btr)), np.flatnonzero(flagged & (bridge_id == bte))
    yt, acc, pu = y[te].astype(int), account[te], pct(USD[te])
    for seed in SEEDS:
        s = train_score(tr, te, seed)
        pm = pct(s)
        for name, sc in [("kan", s), ("kan_hybrid_a0.5", 0.5 * pm + 0.5 * pu), ("kan_hybrid_max", np.maximum(pm, pu))]:
            rows.append(dict(direction=dname, method=name, seed=seed, **metrics(yt, sc, acc)))
        print(f"{dname} seed {seed}: KAN AP {rows[-3]['ap']:.4f} ranks {rows[-3]['ranks']} | "
              f"max-fusion AP {rows[-1]['ap']:.4f} ranks {rows[-1]['ranks']}", flush=True)
        pd.DataFrame(rows).to_csv(RES / "e2b_runs.csv", index=False)

df = pd.DataFrame(rows)
print(df.groupby(["direction", "method"])[["ap", "roc", "first_rank", "hits_top8", "hits_top20", "addr_ap"]]
      .agg(["mean", "std"]).round(4).to_string())
