"""
XAI figures from the trained BridgeMamba-KAN checkpoint (random protocol):
  left  — permutation feature importance (drop in PR-AUC when one feature is
          shuffled across the test set): 11 named context + 14 history channels
  right — learned KAN edge functions (spline response curves) for the top
          context features, extracted from the first KAN layer.
Output: figures/fig_xai.pdf/png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from model import BridgeMambaKAN

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"
RES = PROJECT / "results"
FIG = PROJECT / "figures"

plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})

HIST_NAMES = ["log value", "n events", "log amount", "hour", "log Δt",
              "sc_deposit", "sc_withdrawal", "sc_tok_dep", "sc_tok_wd",
              "tc_deposit", "tc_withdrawal", "tc_tok_dep", "tc_tok_wd",
              "erc20_transfer"]
CTX_NAMES = ["n_tx 1h", "n_wd 1h", "wd_ratio 1h", "amt 1h", "new_acct 1h",
             "n_tx 24h", "n_wd 24h", "wd_ratio 24h", "amt 24h", "new_acct 24h"]

d = np.load(OUT / "histories.npz", allow_pickle=False)
X, lengths, y = d["X"], d["lengths"], d["y"].astype(np.float32)
tx_hash = d["tx_hash"]
splits = dict(np.load(OUT / "splits.npz", allow_pickle=False).items())
ctx = pd.read_parquet(OUT / "context.parquet").set_index("tx_hash")
CTX = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)
L = X.shape[1]
MASK = np.arange(L)[None, :] >= (L - lengths[:, None])

ck = torch.load(RES / "bridgemambakan_random.pt", weights_only=False)
mu_x, sd_x, mu_c, sd_c = ck["norm"]
net = BridgeMambaKAN(d_feat=X.shape[2], d_ctx=CTX.shape[1])
net.load_state_dict(ck["state_dict"])
net.eval()

# test subsample: all attacks + 8k benign
rng = np.random.default_rng(0)
te = np.flatnonzero(splits["random_test"])
pos = te[y[te] == 1]
neg = rng.choice(te[y[te] == 0], size=8000, replace=False)
idx = np.concatenate([pos, neg])
ye = y[idx]


@torch.no_grad()
def score(Xa, Ca):
    outs = []
    for i in range(0, len(idx), 2048):
        outs.append(torch.sigmoid(net(
            torch.from_numpy((Xa[i:i + 2048] - mu_x) / sd_x),
            torch.from_numpy(MASK[idx][i:i + 2048]),
            torch.from_numpy((Ca[i:i + 2048] - mu_c) / sd_c))))
    return torch.cat(outs).numpy()


X0, C0 = X[idx].copy(), CTX[idx].copy()
base = average_precision_score(ye, score(X0, C0))
print(f"base PR-AUC on subsample: {base:.4f}")

imp = {}
for j, name in enumerate(CTX_NAMES):
    Cp = C0.copy()
    Cp[:, j] = Cp[rng.permutation(len(Cp)), j]
    imp[name] = base - average_precision_score(ye, score(X0, Cp))
    print(f"  ctx {name}: {imp[name]:+.4f}", flush=True)
for j, name in enumerate(HIST_NAMES):
    Xp = X0.copy()
    Xp[:, :, j] = Xp[rng.permutation(len(Xp)), :, j]
    imp[name] = base - average_precision_score(ye, score(Xp, C0))
    print(f"  hist {name}: {imp[name]:+.4f}", flush=True)

top = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:10]

# ---- KAN edge response curves for top context features
kan = net.kan1
xs = torch.linspace(-3, 3, 121)
def edge_response(input_dim):
    z = torch.zeros(121, kan.d_in)
    z[:, input_dim] = xs
    with torch.no_grad():
        r = kan(z) - kan(torch.zeros(1, kan.d_in))
    j = r.abs().mean(0).argmax().item()          # most responsive hidden unit
    return r[:, j].numpy()

ctx_offset = 2 * 48                              # ctx dims start after pooled seq
curve_feats = [(n, i) for i, n in enumerate(CTX_NAMES)
               if n in dict(top)][:4] or [(CTX_NAMES[2], 2), (CTX_NAMES[4], 4)]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                         gridspec_kw={"width_ratios": [1.3, 1]})
names = [t[0] for t in top][::-1]
vals = [t[1] for t in top][::-1]
colors = ["#C44E52" if n in CTX_NAMES else "#4878A8" for n in names]
axes[0].barh(names, vals, color=colors)
axes[0].set_xlabel("permutation importance (Δ PR-AUC)")
axes[0].set_title("(a) Feature importance", fontsize=9, loc="left")
axes[0].tick_params(labelsize=8)

for (name, i), c in zip(curve_feats, ["#C44E52", "#4878A8", "#55A868", "#8172B2"]):
    axes[1].plot(xs.numpy(), edge_response(ctx_offset + i), lw=1.5,
                 color=c, label=name)
axes[1].axhline(0, color="k", lw=0.6, ls=":")
axes[1].set_xlabel("standardized feature value")
axes[1].set_ylabel("KAN edge response")
axes[1].set_title("(b) Learned spline edge functions", fontsize=9, loc="left")
axes[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "fig_xai.pdf")
fig.savefig(FIG / "fig_xai.png", dpi=300)
print("saved fig_xai")
