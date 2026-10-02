"""V2 result figures, all read from ../results (no typed numbers).

fig_protocols.pdf   PR-AUC by protocol and model family, mean +/- SD over ten seeds, chance marked
fig_triage.pdf      (a) fusion sweep over alpha, (b) Ronin exploit ranks per scoring rule,
                    (c) Nomad-side average precision per rule with address-cluster 95 % CI
fig_accounts.pdf    (a) precision-recall, (b) ROC low-FPR (seed 42), (c) precision at alert budget k
                    (mean +/- SD over seeds), (d) PR-AUC per seed for each model
fig_federated.pdf   test PR-AUC per round, mean +/- SD over seeds, three conditions x three aggregators
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import C, setup

setup()
RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parent


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig); print("saved", name)


def fig_protocols():
    b = pd.read_csv(RES / "e1a_baselines.csv")
    k = pd.read_csv(RES / "e1b_model.csv")
    a = pd.read_csv(RES / "e5_anomaly.csv") if (RES / "e5_anomaly.csv").exists() else None
    protos = ["random", "temporal", "lobo_ronin", "lobo_nomad"]
    names = ["random\n(in distribution)", "temporal\n(unseen Nomad wave)", "LOBO→Ronin", "LOBO→Nomad"]
    series = [("Logistic regression", C["grey"]), ("Gradient boosting", C["blue"]), ("BridgeMamba-KAN", C["red"])]
    if a is not None:
        series += [("AE reconstruction", C["gold"]), ("latent Mahalanobis", C["purple"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    w = 0.8 / len(series)
    for i, p in enumerate(protos):
        vals = []
        for m in ["logreg", "hgb"]:
            sub = b[(b.protocol == p) & (b.model == m)]
            best = sub.groupby("features").pr_auc.mean().idxmax()
            vals.append(sub[sub.features == best].pr_auc.values)
        vals.append(k[k.protocol == p].pr_auc.values)
        if a is not None:
            for sc in ["AE reconstruction", "latent Mahalanobis"]:
                vals.append(a[(a.protocol == p) & (a.score == sc)].pr_auc.values)
        chance = b[b.protocol == p].chance.iloc[0]
        for j, ((lab, col), v) in enumerate(zip(series, vals)):
            x = i - 0.4 + w * (j + 0.5)
            ax.bar(x, v.mean(), w * 0.92, color=col, label=lab if i == 0 else None, zorder=2)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, color="#222222", lw=0.8, capsize=2, zorder=3)
        ax.plot([i - 0.45, i + 0.45], [chance, chance], ls=":", color="black", lw=1.1, zorder=4,
                label="chance (prevalence)" if i == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(names)
    ax.set_ylabel("PR-AUC (mean ± SD, 10 seeds)"); ax.set_ylim(0, 1.04)
    ax.legend(ncol=3, loc="upper right", fontsize=7)
    save(fig, "fig_protocols")


def fig_triage():
    al = pd.read_csv(RES / "e2_alpha.csv")
    r = pd.read_csv(RES / "e2_runs.csv")
    s = pd.read_csv(RES / "e2_summary.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1.0, 1.15]))
    ax = axs[0]
    for d, col, lab in [("ronin_to_nomad", C["purple"], "Ronin→Nomad"), ("nomad_to_ronin", C["blue"], "Nomad→Ronin")]:
        g = al[al.direction == d].groupby("alpha").ap.agg(["mean", "std"])
        ax.plot(g.index, g["mean"], marker="o", ms=3, color=col, label=lab)
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=col, alpha=0.18, lw=0)
        mx = r[(r.direction == d) & (r.method == "hybrid_max")].ap.mean()
        ax.axhline(mx, color=col, ls="--", lw=0.9)
    ax.set_yscale("log"); ax.set_xlabel("α (weight on learned score)"); ax.set_ylabel("average precision")
    ax.set_title("(a) fusion sweep; dashed = $h_{\\max}$"); ax.legend(loc="center right", fontsize=7)
    ax = axs[1]
    meths = [("usd_prior", "USD prior"), ("raw_amount", "raw amounts"), ("learned", "learned (GB)"),
             ("hybrid_a0.5", "$h_{0.5}$"), ("hybrid_max", "$h_{\\max}$")]
    for i, (m, lab) in enumerate(meths):
        x = r[(r.direction == "nomad_to_ronin") & (r.method == m)]
        rk = [int(v) for v in x.ranks.iloc[0].split()[:2]]
        ax.scatter(rk, [i, i], s=22, color=C["red"], zorder=3)
        ax.plot(rk, [i, i], color="#bbbbbb", lw=0.8, zorder=2)
    ax.axvspan(1, 20, color=C["gold"], alpha=0.18, lw=0)
    ax.set_xscale("log"); ax.set_xlim(0.8, 2e4); ax.set_yticks(range(len(meths))); ax.set_yticklabels([l for _, l in meths])
    ax.invert_yaxis(); ax.set_xlabel("rank in Ronin queue (15,326)"); ax.set_title("(b) Ronin exploit ranks")
    ax = axs[2]
    meths2 = [("learned", "learned", C["blue"]), ("usd_prior", "USD", C["gold"]),
              ("hybrid_a0.5", "$h_{0.5}$", C["purple"]), ("hybrid_max", "$h_{\\max}$", C["red"])]
    for i, (m, lab, col) in enumerate(meths2):
        row = s[(s.direction == "ronin_to_nomad") & (s.method == m)].iloc[0]
        ax.bar(i, row.ap_mean, 0.65, color=col, zorder=2)
        ax.errorbar(i, row.ap_mean, yerr=[[row.ap_mean - row.ap_ci_lo], [row.ap_ci_hi - row.ap_mean]],
                    color="#222222", capsize=2, lw=0.8, zorder=3)
    lo = s[(s.direction == "ronin_to_nomad") & s.method.str.startswith("learned_only")]
    for _, row in lo.iterrows():
        ax.scatter(0, row.ap_mean, marker="x", color="#222222", s=18, zorder=4)
    chance = 463 / 1070
    ax.axhline(chance, ls=":", color="black", lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _ in meths2], fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("AP, Nomad queue"); ax.set_title("(c) Ronin→Nomad, 95 % CI; × = one positive")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_triage")


def fig_accounts():
    r = pd.read_csv(RES / "e3_runs.csv")
    z = np.load(RES / "e3_scores" / "seed42.npz")
    y = z["y"]
    models = [("BridgeMamba_KAN", "BridgeMamba-KAN", C["red"]), ("RandomForest", "Random forest", C["blue"]),
              ("HistGradientBoosting", "Gradient boosting", C["green"]), ("LogisticRegression", "Logistic regression", C["purple"])]
    fig, axs = plt.subplots(1, 4, figsize=(9.4, 2.4))
    for key, lab, col in models:
        p, rc, _ = precision_recall_curve(y, z[key]); axs[0].plot(rc, p, color=col, lw=1.2, label=lab)
        fpr, tpr, _ = roc_curve(y, z[key]); axs[1].plot(fpr, tpr, color=col, lw=1.2)
    axs[0].axhline(y.mean(), ls=":", color="black", lw=1, label=f"chance ({y.mean():.4f})")
    axs[0].set_yscale("log"); axs[0].set_xlabel("recall"); axs[0].set_ylabel("precision"); axs[0].set_title("(a) precision–recall (seed 42)")
    axs[0].legend(fontsize=6, loc="lower left")
    axs[1].set_xscale("log"); axs[1].set_xlim(1e-5, 1); axs[1].set_xlabel("false positive rate (log)")
    axs[1].set_ylabel("true positive rate"); axs[1].set_title("(b) ROC, low-FPR regime")
    ks = np.array([25, 50, 100, 150, 200, 250, 350, 500])
    for key, lab, col in models:
        P = []
        for sd in sorted(r.seed.unique()):
            zz = np.load(RES / "e3_scores" / f"seed{sd}.npz")
            o = np.argsort(-zz[key], kind="stable")
            P.append([zz["y"][o[:k]].mean() for k in ks])
        P = np.array(P)
        axs[2].plot(ks, P.mean(0), marker="o", ms=2.5, color=col, lw=1.1)
        axs[2].fill_between(ks, P.mean(0) - P.std(0, ddof=1), P.mean(0) + P.std(0, ddof=1), color=col, alpha=0.15, lw=0)
    for k in (100, 250, 500):
        axs[2].axvline(k, color="#bbbbbb", lw=0.7, ls="--")
    axs[2].set_xlabel("alert budget k (accounts)"); axs[2].set_ylabel("precision@k"); axs[2].set_title("(c) label-free operating points")
    for i, (key, lab, col) in enumerate(models):
        v = r[r.model == lab.replace("Random forest", "RandomForest").replace("Gradient boosting", "HistGradientBoosting")
              .replace("Logistic regression", "LogisticRegression")].pr_auc.values
        axs[3].scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=10, color=col, zorder=3)
        axs[3].plot([i - 0.25, i + 0.25], [v.mean()] * 2, color="#222222", lw=1.2)
    axs[3].set_xticks(range(4)); axs[3].set_xticklabels(["KAN", "RF", "GB", "LR"])
    axs[3].set_ylabel("PR-AUC per seed"); axs[3].set_title("(d) ten seeds")
    fig.tight_layout(w_pad=0.7)
    save(fig, "fig_accounts")


def fig_federated():
    r = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("e4_runs*.csv"))]).drop_duplicates(["seed", "condition", "aggregator", "round"])
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4), sharey=True)
    cols = {"mean": C["blue"], "median": C["red"], "trimmed": C["green"]}
    for ax, (c, title) in zip(axs, [("clean", "clean federation"), ("label_flip", "label flip, one client"),
                                     ("model_replacement", "model replacement (×4)")]):
        for agg, col in cols.items():
            g = r[(r.condition == c) & (r.aggregator == agg)].groupby("round").pr_auc.agg(["mean", "std"])
            ax.plot(g.index, g["mean"], marker="o", ms=2.5, color=col, label={"mean": "FedAvg", "median": "coord. median",
                                                                               "trimmed": "trimmed mean 20 %"}[agg])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=col, alpha=0.15, lw=0)
        ax.set_title(title); ax.set_xlabel("round")
    axs[0].set_ylabel("test PR-AUC (mean ± SD)"); axs[0].legend(fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_federated")


if __name__ == "__main__":
    for fn in sys.argv[1:] or ["fig_protocols", "fig_triage", "fig_accounts", "fig_federated"]:
        try:
            globals()[fn]()
        except FileNotFoundError as e:
            print("skip", fn, e)
