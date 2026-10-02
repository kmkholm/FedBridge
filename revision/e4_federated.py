"""E4 -- federated prototype repeated over seeds, with a stronger adversary (GPU).

Same design as src/train_federated.py: four clients = bridge x account half of
the random-protocol training data (8,000 benign rows per client cap), BridgeMamba-KAN with
d_model 32, 8 rounds x 2 local epochs, AdamW 2e-3, aggregators mean / coordinate median /
20% trimmed mean, evaluation on the random-protocol test set.

Changes for the revision:
  * seeds 0-9 (model init, local shuffling, benign subsampling)
  * account halves use a fixed MD5 hash; v1 used Python's per-process salted hash()
  * three conditions: clean; label flip on client 0; boosted model replacement on client 0
    (label-flipped local model, update scaled by the number of clients before aggregation)

writes results/e4_runs.csv (one row per seed x condition x aggregator x round)
"""
import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

DATA = Path(__file__).resolve().parents[1] / "processed"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import BridgeMambaKAN  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
SEEDS = [42] + list(range(9))   # 42 = seed of the original submission
CONDITIONS = ("clean", "label_flip", "model_replacement")
AGGS = ("mean", "median", "trimmed")
ROUNDS, LOCAL_EPOCHS, BATCH, LR, MAX_NEG, BOOST = 8, 2, 512, 2e-3, 8_000, 4.0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)   # several conditions run as parallel processes

d = np.load(DATA / "histories.npz", allow_pickle=False)
X, lengths, y = d["X"], d["lengths"], d["y"].astype(np.float32)
tx_hash, bridge_id, account = d["tx_hash"], d["bridge_id"], d["account"]
splits = dict(np.load(DATA / "splits.npz", allow_pickle=False).items())
ctx = pd.read_parquet(DATA / "context.parquet").set_index("tx_hash")
usd = pd.read_parquet(DATA / "usd.parquet").set_index("tx_hash")["log_usd"]
CTX = np.hstack([ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32),
                 usd.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)[:, None]])
MASK = np.arange(X.shape[1])[None, :] >= (X.shape[1] - lengths[:, None])
TR, TE = np.flatnonzero(splits["random_train"]), np.flatnonzero(splits["random_test"])
HALF = np.array([int(hashlib.md5(str(a).encode()).hexdigest(), 16) % 2 for a in account])


def make_clients(seed):
    rng = np.random.default_rng(seed)
    clients = []
    for b in (0, 1):
        idx = TR[bridge_id[TR] == b]
        for h in (0, 1):
            c = idx[HALF[idx] == h]
            neg, pos = c[y[c] == 0], c[y[c] == 1]
            if len(neg) > MAX_NEG:
                neg = rng.choice(neg, size=MAX_NEG, replace=False)
            clients.append({"name": f"{'ronin' if b == 0 else 'nomad'}-{'AB'[h]}", "idx": np.concatenate([pos, neg])})
    return clients


def tens(idx, mu, labels=None):
    mu_x, sd_x, mu_c, sd_c = mu
    yy = y[idx] if labels is None else labels
    return tuple(torch.from_numpy(a).to(DEV) for a in
                 ((X[idx] - mu_x) / sd_x, MASK[idx], (CTX[idx] - mu_c) / sd_c, yy))


def local_train(net, t):
    xt, mt, ct, yt = t
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    pw = torch.tensor(min((yt == 0).sum().item() / max(yt.sum().item(), 1), 300.0), device=DEV)
    for _ in range(LOCAL_EPOCHS):
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(net(xt[b], mt[b], ct[b]), yt[b], pos_weight=pw).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net.state_dict()


def aggregate(states, sizes, how):
    w = torch.tensor(np.asarray(sizes, float) / np.sum(sizes), dtype=torch.float32, device=DEV)
    out = {}
    for k in states[0]:
        st = torch.stack([s[k].float() for s in states])
        if how == "mean":
            out[k] = (st * w.view(-1, *[1] * (st.dim() - 1))).sum(0)
        elif how == "median":
            out[k] = st.median(0).values
        else:
            kt = max(1, int(0.2 * len(states)))
            out[k] = st.sort(0).values[kt:len(states) - kt].mean(0) if len(states) > 2 * kt else st.mean(0)
    return out


@torch.no_grad()
def evaluate(net, te):
    net.eval()
    xe, me, ce, ye = te
    s = torch.cat([torch.sigmoid(net(xe[i:i + 4096], me[i:i + 4096], ce[i:i + 4096])) for i in range(0, len(xe), 4096)])
    s, yn = s.cpu().numpy(), ye.cpu().numpy()
    return average_precision_score(yn, s), roc_auc_score(yn, s)


def main(conditions, out_name):
    out = RES / out_name
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    prev = [pd.read_csv(f) for f in RES.glob("e4_runs*.csv")]
    allr = pd.concat(prev) if prev else pd.DataFrame(columns=["seed", "condition", "aggregator", "round"])
    full = allr.groupby(["seed", "condition", "aggregator"]).size()
    done = {k for k, v in full.items() if v >= ROUNDS}
    print(f"device {DEV}", flush=True)
    for seed in SEEDS:
        clients = make_clients(seed)
        allc = np.concatenate([c["idx"] for c in clients])
        mu = (X[allc].mean((0, 1)), X[allc].std((0, 1)) + 1e-6, CTX[allc].mean(0), CTX[allc].std(0) + 1e-6)
        te = tens(TE, mu)
        for cond in conditions:
            ct = [tens(c["idx"], mu, (1.0 - y[c["idx"]]) if (cond != "clean" and k == 0) else None)
                  for k, c in enumerate(clients)]
            for how in AGGS:
                if (seed, cond, how) in done:
                    continue
                t0 = time.time()
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                g = BridgeMambaKAN(d_feat=X.shape[2], d_ctx=CTX.shape[1], d_model=32).to(DEV)
                for rnd in range(1, ROUNDS + 1):
                    states = []
                    gs = {k: v.detach().clone() for k, v in g.state_dict().items()}
                    for k, c in enumerate(clients):
                        st = local_train(copy.deepcopy(g), ct[k])
                        if cond == "model_replacement" and k == 0:
                            st = {n: (gs[n] + BOOST * (v - gs[n])) if v.is_floating_point() else v for n, v in st.items()}
                        states.append(st)
                    g.load_state_dict(aggregate(states, [len(c["idx"]) for c in clients], how))
                    pr, roc = evaluate(g, te)
                    rows.append(dict(seed=seed, condition=cond, aggregator=how, round=rnd, pr_auc=pr, roc_auc=roc))
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"seed {seed} {cond:17s} {how:7s} final PR-AUC {pr:.4f} ROC {roc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    fin = df[df["round"] == ROUNDS].groupby(["condition", "aggregator"]).pr_auc.agg(["mean", "std", "count"])
    print(fin.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--out", default="e4_runs.csv")
    a = ap.parse_args()
    main(a.conditions, a.out)
