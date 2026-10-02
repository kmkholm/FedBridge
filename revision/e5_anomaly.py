"""E5 -- benign-only anomaly detection repeated over seeds (GPU).

Imports SeqAE / data from src/train_anomaly.py and the latent scorers from
score_anomaly_v2.py (both unchanged), and reproduces their procedure per seed: train the
sequence autoencoder on up to 60k benign training rows (6 epochs, batch 512, AdamW 2e-3),
score test rows by reconstruction error, then by Mahalanobis and 10-NN distance in the
learned latent space, and by Mahalanobis distance on raw context features (CtxGaussian).
Nothing is written into the FedBridge tree.

writes results/e5_anomaly.csv
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

SRC = Path(__file__).resolve().parents[1] / "src"
RES = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(SRC))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ta, sa = load("train_anomaly"), load("score_anomaly_v2")
X, MASK, TARGET, CTX, y, splits = ta.X, ta.MASK, ta.TARGET, ta.CTX, ta.y, ta.splits
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS, PROTOCOLS = [42] + list(range(9)), ["temporal", "lobo_ronin", "lobo_nomad", "random"]


@torch.no_grad()
def latents(net, idx, mu_x, sd_x, batch=4096):
    zs = []
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        h = net.embed(torch.from_numpy((X[b] - mu_x) / sd_x).to(DEV))
        for blk in net.blocks:
            h = blk(h)
        h = net.norm(h)
        m = torch.from_numpy(MASK[b]).to(DEV).unsqueeze(-1).float()
        zs.append(torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), h[:, -1]], -1).cpu())
    return torch.cat(zs).numpy().astype(np.float64)


def run(proto, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    ben = np.flatnonzero(splits[f"{proto}_train"] & (y == 0))
    if len(ben) > ta.MAX_TRAIN:
        ben = rng.choice(ben, size=ta.MAX_TRAIN, replace=False)
    te = np.flatnonzero(splits[f"{proto}_test"])
    ye = y[te]
    mu_x, sd_x = X[ben].mean((0, 1)), X[ben].std((0, 1)) + 1e-6
    mu_t, sd_t = TARGET[ben].mean(0), TARGET[ben].std(0) + 1e-6
    T = lambda i: tuple(torch.from_numpy(a).to(DEV) for a in ((X[i] - mu_x) / sd_x, MASK[i], (TARGET[i] - mu_t) / sd_t))
    xt, mt, tt = T(ben)
    net = ta.SeqAE(X.shape[2], TARGET.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=ta.LR, weight_decay=1e-4)
    for _ in range(ta.EPOCHS):
        net.train()
        perm = torch.randperm(len(ben), device=DEV)
        for i in range(0, len(ben), ta.BATCH):
            b = perm[i:i + ta.BATCH]
            opt.zero_grad()
            F.mse_loss(net(xt[b], mt[b]), tt[b]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    xe, me, tte = T(te)
    with torch.no_grad():
        rec = torch.cat([((net(xe[i:i + 4096], me[i:i + 4096]) - tte[i:i + 4096]) ** 2).mean(-1)
                         for i in range(0, len(te), 4096)]).cpu().numpy()
    z_fit, z_te = latents(net, ben, mu_x, sd_x), latents(net, te, mu_x, sd_x)
    sub = z_fit[rng.choice(len(z_fit), size=min(sa.KNN_FIT, len(z_fit)), replace=False)]
    mu_c, sd_c = CTX[ben].mean(0), CTX[ben].std(0) + 1e-6
    scores = {"AE reconstruction": rec, "latent Mahalanobis": sa.mahal_scores(z_fit, z_te),
              "latent kNN": sa.knn_scores(sub, z_te),
              "context Mahalanobis": sa.mahal_scores((CTX[ben] - mu_c) / sd_c, (CTX[te] - mu_c) / sd_c)}
    return [dict(protocol=proto, seed=seed, score=k, pr_auc=average_precision_score(ye, s),
                 roc_auc=roc_auc_score(ye, s), recall_at_fpr=ta.recall_at_fpr(ye, s),
                 chance=float(ye.mean()), n_test=len(te), n_pos=int(ye.sum())) for k, s in scores.items()]


if __name__ == "__main__":
    out = RES / "e5_anomaly.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["protocol"], r["seed"]) for r in rows}
    for proto in PROTOCOLS:
        for seed in SEEDS:
            if (proto, seed) in done:
                continue
            rows += run(proto, seed)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{proto} seed {seed}: " + "  ".join(f"{r['score']} {r['pr_auc']:.3f}/{r['roc_auc']:.3f}" for r in rows[-4:]), flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["protocol", "score"])[["pr_auc", "roc_auc"]].agg(["mean", "std"]).round(4).to_string())
