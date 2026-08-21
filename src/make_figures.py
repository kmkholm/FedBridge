"""
Paper figures — regenerate ALL from processed data + results CSVs.
Every figure saved as vector PDF + 300dpi PNG into figures/.

  fig_timeline      per-bridge weekly tx volume, exploit windows shaded
  fig_onset         Nomad Aug-1-2022 hourly unmatched rate (onset validation)
  fig_context       benign-vs-attack distribution of wd_ratio_1h & new_acct_1h
  fig_protocols     the "evaluation illusion": best PR-AUC by protocol/model,
                    with chance level (prevalence) marked per protocol

Run after any experiment: python make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"
RES = PROJECT / "results"
FIG = PROJECT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})
C_BENIGN, C_ATTACK, C_SUSP = "#4878A8", "#C44E52", "#B8860B"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"saved figures/{name}.pdf + .png")


txs = pd.read_parquet(OUT / "transactions_refined.parquet")
txs["dt"] = pd.to_datetime(txs.timestamp, unit="s")

# ---------------------------------------------------------------- timeline
fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=False)
windows = {"ronin-bridge": [("2022-03-23", "2022-03-24", "Ronin exploit\n($611M)")],
           "nomad-bridge": [("2022-08-01", "2022-08-06", "Nomad exploit wave\n($190M)")]}
for ax, bridge in zip(axes, ["ronin-bridge", "nomad-bridge"]):
    g = txs[txs.bridge == bridge].set_index("dt").resample("W").size()
    ax.fill_between(g.index, g.values, color=C_BENIGN, alpha=0.75, lw=0)
    for lo, hi, lab in windows[bridge]:
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=C_ATTACK, alpha=0.35)
        ax.annotate(lab, xy=(pd.Timestamp(lo), g.max() * 0.72),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=8, color=C_ATTACK)
    ax.set_ylabel("txs / week")
    ax.set_title(bridge.replace("-bridge", " bridge").title(), fontsize=9, loc="left")
axes[1].set_xlabel("date")
fig.tight_layout()
save(fig, "fig_timeline")

# ------------------------------------------------------------------- onset
nomad = txs[(txs.bridge == "nomad-bridge")
            & (txs.dt.dt.date.astype(str) == "2022-08-01")].copy()
hourly = (nomad.assign(flagged=nomad.label.isin(["attack_unmatched", "anomaly"]))
          .groupby(nomad.dt.dt.hour)
          .agg(total=("tx_hash", "size"), flagged=("flagged", "sum")))
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.bar(hourly.index, hourly.total - hourly.flagged, color=C_BENIGN,
       label="matched / benign", width=0.8)
ax.bar(hourly.index, hourly.flagged, bottom=hourly.total - hourly.flagged,
       color=C_ATTACK, label="unmatched / anomalous", width=0.8)
ax.axvline(21, color="k", ls="--", lw=0.8)
ax.annotate("first exploit tx\n~21:32 UTC", xy=(21, hourly.total.max() * 0.82),
            xytext=(-68, 0), textcoords="offset points", fontsize=8)
ax.set_xlabel("hour of 2022-08-01 (UTC)")
ax.set_ylabel("transactions")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
save(fig, "fig_onset")

# ----------------------------------------------------------------- context
ctx = pd.read_parquet(OUT / "context.parquet").merge(
    txs[["tx_hash", "label_refined"]], on="tx_hash")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
for ax, col, lab in [(axes[0], "wd_ratio_1h", "withdrawal ratio (1h window)"),
                     (axes[1], "new_acct_1h", "new-account fraction (1h window)")]:
    bins = np.linspace(0, 1, 41)
    for lr, color, name in [("benign", C_BENIGN, "benign"),
                            ("attack", C_ATTACK, "attack")]:
        v = ctx.loc[ctx.label_refined == lr, col]
        ax.hist(v, bins=bins, density=True, alpha=0.6, color=color, label=name)
    ax.set_xlabel(lab)
    ax.set_ylabel("density")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, "fig_context")

# --------------------------------------------------------------- protocols
base = pd.read_csv(RES / "baselines.csv")
best_base = (base.sort_values("pr_auc").groupby("protocol").last()
             .rename(columns={"pr_auc": "score"}))
rows = {"Best classical (GB/LR)": best_base["score"]}
mr_path = RES / "model_results.csv"
if mr_path.exists():
    mr = pd.read_csv(mr_path)
    for model, g in mr.groupby("model"):
        rows[model] = g.set_index("protocol")["pr_auc"]
prevalence = {"random": 465 / 161655, "temporal": 463 / 786,
              "lobo_ronin": 2 / 129321, "lobo_nomad": 463 / 32334}
order = ["random", "temporal", "lobo_ronin", "lobo_nomad"]
labels = ["random\n(the illusion)", "temporal\n(zero-day)",
          "LOBO→ronin", "LOBO→nomad"]
fig, ax = plt.subplots(figsize=(5.4, 2.8))
width = 0.8 / len(rows)
palette = ["#888888", "#4878A8", "#C44E52", "#55A868", "#8172B2"]
for j, (name, series) in enumerate(rows.items()):
    vals = [series.get(p, np.nan) for p in order]
    xs = np.arange(len(order)) + j * width
    ax.bar(xs, vals, width=width * 0.92, label=name, color=palette[j % len(palette)])
for i, p in enumerate(order):
    ax.hlines(prevalence[p], i - 0.08, i + 0.8, color="k", ls=":", lw=1)
ax.hlines([], [], [], color="k", ls=":", lw=1, label="chance (prevalence)")
ax.set_xticks(np.arange(len(order)) + 0.4 - width / 2)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("PR-AUC")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout()
save(fig, "fig_protocols")

print("done")
