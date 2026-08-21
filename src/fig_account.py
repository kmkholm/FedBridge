"""Figures for the exploiter-account detection experiment.
(a) precision-recall curves for all four models against the chance level
(b) ROC curves (log-x) showing behaviour in the low false-positive regime
(c) confusion matrix of the proposed model at the operating point
(d) score separation between exploiter and benign accounts
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

PROJECT = Path(__file__).resolve().parents[1]
EXT = PROJECT / "processed_ext"
FIG = PROJECT / "figures"
plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})

z = np.load(EXT / "acct_scores.npz")
y = z["y"]
models = [("BridgeMamba-KAN", "kan", "#C44E52"),
          ("Random forest", "rf", "#4878A8"),
          ("Gradient boosting", "hgb", "#55A868"),
          ("Logistic regression", "lr", "#8172B2")]

fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.0))

# (a) PR curves
for nm, k, c in models:
    p, r, _ = precision_recall_curve(y, z[k])
    ax[0].plot(r, p, lw=1.6, color=c,
               label=f"{nm} ({average_precision_score(y, z[k]):.3f})")
ax[0].axhline(y.mean(), color="k", ls=":", lw=1,
              label=f"chance ({y.mean():.4f})")
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
ax[0].set_yscale("log")
ax[0].set_title("(a) Precision–recall", loc="left", fontsize=9)
ax[0].legend(frameon=False, fontsize=7)

# (b) ROC, log x
for nm, k, c in models:
    fpr, tpr, _ = roc_curve(y, z[k])
    ax[1].plot(np.clip(fpr, 1e-5, 1), tpr, lw=1.6, color=c,
               label=f"{nm} ({roc_auc_score(y, z[k]):.4f})")
ax[1].set_xscale("log")
ax[1].set_xlabel("false positive rate (log)"); ax[1].set_ylabel("true positive rate")
ax[1].set_title("(b) ROC, low-FPR regime", loc="left", fontsize=9)
ax[1].legend(frameon=False, fontsize=7, loc="lower right")

# (c) confusion matrix of the proposed model
s = z["kan"]
thr = np.quantile(s, 1 - y.mean())
cm = confusion_matrix(y, (s >= thr).astype(int))
ax[2].imshow(np.log10(cm + 1), cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[2].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                   fontsize=11,
                   color="white" if cm[i, j] > cm.max() / 3 else "#1a3a5c")
ax[2].set_xticks([0, 1], ["benign", "exploiter"])
ax[2].set_yticks([0, 1], ["benign", "exploiter"])
ax[2].set_xlabel("predicted"); ax[2].set_ylabel("actual")
ax[2].set_title("(c) BridgeMamba-KAN confusion", loc="left", fontsize=9)
ax[2].spines[:].set_visible(False)

# (d) score separation
bins = np.linspace(0, 1, 41)
ax[3].hist(s[y == 0], bins=bins, density=True, alpha=0.65, color="#4878A8",
           label="benign accounts")
ax[3].hist(s[y == 1], bins=bins, density=True, alpha=0.65, color="#C44E52",
           label="exploiter accounts")
ax[3].set_yscale("log")
ax[3].set_xlabel("model score"); ax[3].set_ylabel("density (log)")
ax[3].set_title("(d) Score separation", loc="left", fontsize=9)
ax[3].legend(frameon=False, fontsize=7.5)

fig.tight_layout()
fig.savefig(FIG / "fig_account.pdf")
fig.savefig(FIG / "fig_account.png", dpi=300)
print("saved fig_account")
