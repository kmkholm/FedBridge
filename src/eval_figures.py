"""
Evaluation figures for BridgeMamba-KAN (random-protocol checkpoint):
  fig_roc_pr     ROC curve + PR curve side by side
  fig_confusion  confusion matrix at the 0.1%-FPR operating threshold
  fig_training   loss + validation PR-AUC per epoch (from the training log)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)

from model import BridgeMambaKAN

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"
RES = PROJECT / "results"
FIG = PROJECT / "figures"

plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})
C1, C2 = "#4878A8", "#C44E52"

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

te = np.flatnonzero(splits["random_test"])
ye = y[te]
scores = []
with torch.no_grad():
    for i in range(0, len(te), 2048):
        b = te[i:i + 2048]
        scores.append(torch.sigmoid(net(
            torch.from_numpy((X[b] - mu_x) / sd_x),
            torch.from_numpy(MASK[b]),
            torch.from_numpy((CTX[b] - mu_c) / sd_c))))
s = torch.cat(scores).numpy()
np.savez(RES / "scores_random.npz", y=ye, score=s, tx_hash=tx_hash[te])

# ---------------- ROC + PR
fpr, tpr, thr = roc_curve(ye, s)
prec, rec, _ = precision_recall_curve(ye, s)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
axes[0].plot(fpr, tpr, color=C1, lw=1.5,
             label=f"BridgeMamba-KAN (AUC={auc(fpr, tpr):.4f})")
axes[0].plot([0, 1], [0, 1], "k:", lw=0.8, label="chance")
axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
axes[0].set_title("ROC", fontsize=9); axes[0].legend(frameon=False, fontsize=7.5)
axes[1].plot(rec, prec, color=C1, lw=1.5,
             label=f"BridgeMamba-KAN (AP={average_precision_score(ye, s):.4f})")
axes[1].axhline(ye.mean(), color="k", ls=":", lw=0.8,
                label=f"chance (prev={ye.mean():.4f})")
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
axes[1].set_title("Precision–Recall", fontsize=9)
axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "fig_roc_pr.pdf"); fig.savefig(FIG / "fig_roc_pr.png", dpi=300)
plt.close(fig); print("saved fig_roc_pr")

# ---------------- confusion matrix at 0.1% FPR threshold
t = thr[np.searchsorted(fpr, 0.001, side="right") - 1]
pred = (s >= t).astype(int)
cm = confusion_matrix(ye, pred)
fig, ax = plt.subplots(figsize=(3.1, 2.7))
ax.imshow(np.log10(cm + 1), cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > cm.max() / 3 else "#1a3a5c")
ax.set_xticks([0, 1], ["benign", "attack"]); ax.set_yticks([0, 1], ["benign", "attack"])
ax.set_xlabel("predicted"); ax.set_ylabel("actual")
ax.set_title(f"threshold @ 0.1% FPR (t={t:.3f})", fontsize=8.5)
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_confusion.pdf"); fig.savefig(FIG / "fig_confusion.png", dpi=300)
plt.close(fig); print("saved fig_confusion")

# ---------------- training curves (from the run log)
loss = [0.3247, 0.0814, 0.0582, 0.0489, 0.0372, 0.0283, 0.0268, 0.0177]
vap = [0.9413, 0.9695, 0.9913, 0.9807, 0.9916, 0.9928, 0.9970, 0.9978]
ep = np.arange(1, 9)
fig, ax1 = plt.subplots(figsize=(4.4, 2.7))
ax1.plot(ep, loss, "-o", color=C2, ms=3.5, lw=1.3, label="training loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("weighted BCE loss", color=C2)
ax1.tick_params(axis="y", labelcolor=C2)
ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(ep, vap, "-s", color=C1, ms=3.5, lw=1.3, label="validation PR-AUC")
ax2.set_ylabel("validation PR-AUC", color=C1)
ax2.tick_params(axis="y", labelcolor=C1)
ax2.set_ylim(0.9, 1.005)
fig.tight_layout()
fig.savefig(FIG / "fig_training.pdf"); fig.savefig(FIG / "fig_training.png", dpi=300)
plt.close(fig); print("saved fig_training")
