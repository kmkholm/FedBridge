"""Corrected multi-bridge figure (replaces the earlier version built on the
invalid violation labels).

(a) corpus scale per bridge, annotated with which bridges carry credible
    paired-USD labels;
(b) the conservation-ratio distribution on Across, showing why the label is
    credible: mass just below 1.0 (fees) with a thin anomalous tail;
(c) precision-recall on the temporal test split for all four models.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

PROJECT = Path(__file__).resolve().parents[1]
EXT = PROJECT / "processed_ext"
FIG = PROJECT / "figures"

plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})

# ---- (a) corpus composition
BRIDGES = [("stargate_bus", 3_526_050, True), ("across", 3_864_421, True),
           ("stargate_oft", 3_291_711, False), ("wormhole", 699_474, False),
           ("cctp", 592_141, False), ("connext", 464_542, False),
           ("orbitChain", 458_055, False), ("allBridge", 146_210, False),
           ("ccip", 11_430, False)]
BRIDGES.sort(key=lambda x: x[1])

fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.1))
names = [b[0] for b in BRIDGES]
vals = [b[1] / 1e6 for b in BRIDGES]
cols = ["#4878A8" if b[2] else "#B0B7BE" for b in BRIDGES]
ax[0].barh(names, vals, color=cols)
ax[0].set_xlabel("transactions (millions)")
ax[0].set_title("(a) Extended corpus, 12.35M transactions", loc="left", fontsize=9)
ax[0].text(0.98, 0.06, "blue: paired USD amounts\n(labels computable)",
           transform=ax[0].transAxes, ha="right", fontsize=7.5, color="#4878A8")

# ---- (b) conservation ratio distribution
d = pd.read_parquet(EXT / "xcdg_across.parquet",
                    columns=["input_amount_usd", "output_amount_usd"])
i = pd.to_numeric(d.input_amount_usd, errors="coerce")
o = pd.to_numeric(d.output_amount_usd, errors="coerce")
m = (i > 0) & o.notna()
r = (o[m] / i[m]).clip(0.9, 1.1)
ax[1].hist(r, bins=200, color="#4878A8", log=True)
ax[1].axvline(1.001, color="#C44E52", ls="--", lw=1.2)
ax[1].text(1.0015, ax[1].get_ylim()[1] * 0.25, "violation\nthreshold",
           fontsize=7.5, color="#C44E52")
ax[1].set_xlabel("destination / source value ratio")
ax[1].set_ylabel("transactions (log)")
ax[1].set_title("(b) Conservation ratio, Across", loc="left", fontsize=9)

# ---- (c) PR curves on the temporal split
z = np.load(EXT / "v2_scores.npz")
y = z["y"]
series = [("BridgeMamba-KAN", z["kan"], "#C44E52"),
          ("Gradient boosting", z["hgb"], "#4878A8"),
          ("Logistic regression", z["lr"], "#55A868"),
          ("Deposit-size prior", z["amt"], "#8C8C8C")]
for nm, s, c in series:
    p, rc, _ = precision_recall_curve(y, s)
    from sklearn.metrics import average_precision_score
    ax[2].plot(rc, p, lw=1.5, color=c,
               label=f"{nm} (AP={average_precision_score(y, s):.4f})")
ax[2].axhline(y.mean(), color="k", ls=":", lw=0.9,
              label=f"chance ({y.mean():.5f})")
ax[2].set_xlabel("recall")
ax[2].set_ylabel("precision")
ax[2].set_yscale("log")
ax[2].set_title("(c) Temporal split, Across", loc="left", fontsize=9)
ax[2].legend(frameon=False, fontsize=7)

fig.tight_layout()
fig.savefig(FIG / "fig_multibridge.pdf")
fig.savefig(FIG / "fig_multibridge.png", dpi=300)
print("saved corrected fig_multibridge")
