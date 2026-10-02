"""Build the V2 LaTeX tables and a numbers digest from the result CSVs.

Every number that enters the manuscript is printed here first, so the text can be checked
against one file (results/numbers.txt). Tables go to results/tex/*.tex.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "results"
TEX = RES / "tex"
TEX.mkdir(exist_ok=True)
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.append(s)


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}" if len(v) > 1 and v.std(ddof=1) > 1e-12 else f"{v.mean():.{d}f}"


def ci95(v):
    v = np.asarray(v, float)
    h = 2.262 * v.std(ddof=1) / np.sqrt(len(v))      # t(0.975, 9)
    return v.mean() - h, v.mean() + h


def sci(x):
    if x >= 1e-3:
        return f"{x:.3f}"
    e = int(np.floor(np.log10(x)))
    return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"


# ------------------------------------------------------------------ Table 3: protocols
def table3():
    b = pd.read_csv(RES / "e1a_baselines.csv")
    k = pd.read_csv(RES / "e1b_model.csv") if (RES / "e1b_model.csv").exists() else pd.DataFrame()
    names = {"random": "Random", "temporal": "Temporal", "lobo_ronin": "LOBO$\\to$Ronin", "lobo_nomad": "LOBO$\\to$Nomad"}
    lines = []
    for p in ["random", "temporal", "lobo_ronin", "lobo_nomad"]:
        sub = b[b.protocol == p]
        chance = sub.chance.iloc[0]
        g = sub.groupby(["model", "features"]).pr_auc.mean()
        rows = []
        for m, lab in [("logreg", "LogReg"), ("hgb", "Gradient boosting")]:
            feat = g[m].idxmax()
            r = sub[(sub.model == m) & (sub.features == feat)]
            rows.append((f"{lab} ({feat})", r))
        if len(k) and (k.protocol == p).any():
            rows.append(("BridgeMamba-KAN", k[k.protocol == p].rename(columns={"recall_at_fpr": "recall_at_fpr"})))
        for j, (lab, r) in enumerate(rows):
            n = len(r)
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{names[p]} ({sci(chance)})}}" if j == 0 else ""
            pr = ms(r.pr_auc, 4 if p.startswith("lobo") else 3) if not p == "lobo_ronin" else \
                f"{sci(r.pr_auc.mean())}" + (f" $\\pm$ {sci(r.pr_auc.std(ddof=1))}" if r.pr_auc.std(ddof=1) > 0 else "")
            lines.append(f"{first} & {lab} & {pr} & {ms(r.roc_auc)} & {ms(r.recall_at_fpr)} & {n} \\\\")
            say("T3", p, lab, "PR", r.pr_auc.mean().round(4), "sd", round(r.pr_auc.std(ddof=1), 4),
                "ROC", r.roc_auc.mean().round(4), "R@.1%", r.recall_at_fpr.mean().round(3),
                "seed42", r[r.seed == 42].pr_auc.round(4).tolist())
        lines.append("\\midrule")
    lines = lines[:-1]
    (TEX / "table3_rows.tex").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ Table 4: triage
def table4():
    r = pd.read_csv(RES / "e2_runs.csv")
    s = pd.read_csv(RES / "e2_summary.csv").set_index(["direction", "method"])
    kb = pd.read_csv(RES / "e2b_runs.csv") if (RES / "e2b_runs.csv").exists() else pd.DataFrame()

    def ranks(d, m):
        x = r[(r.direction == d) & (r.method == m)]
        if d == "ronin_to_nomad":
            n8 = int((x.hits_top8 == 8).sum())
            if n8 == len(x):
                return f"1--8 of 1{{,}}070 ({n8}/{len(x)} seeds)" if len(x) > 1 and x.ap.std() > 0 else "1--8 of 1{,}070"
            return f"{int(x.hits_top8.mean())} of first 8; first at rank {int(x.first_rank.min())}"
        rr = x.ranks.iloc[0].split()
        return ", ".join(f"{int(v):,}".replace(",", "{,}") for v in rr[:2]) + " of 15{,}326"

    def ap(d, m, kan=False):
        x = (kb if kan else r)
        x = x[(x.direction == d) & (x.method == m)]
        fmt = lambda u: f"{u:.3f}" if u >= 0.01 else sci(u)
        v = ms(x.ap) if len(x) > 1 and x.ap.std() > 1e-12 and x.ap.mean() >= 0.01 else fmt(x.ap.mean())
        key = (d, m)
        if not kan and key in s.index and not np.isnan(s.loc[key, "ap_ci_lo"]):
            v += f" [{s.loc[key, 'ap_ci_lo']:.2f}, {s.loc[key, 'ap_ci_hi']:.2f}]"
        say("T4", d, m, "kan" if kan else "", "AP", round(x.ap.mean(), 4), "sd", round(x.ap.std(ddof=1), 4) if len(x) > 1 else "",
            "first", int(x.first_rank.max()), "top8", x.hits_top8.mean(), "top20", x.hits_top20.mean(), "ranks", x.ranks.iloc[0])
        return v

    def kranks(d, m):
        x = kb[(kb.direction == d) & (kb.method == m)]
        if d == "nomad_to_ronin":
            rr = x.ranks.str.split(expand=True).astype(int)
            f = lambda lo, hi: (f"{lo:,}" if lo == hi else f"{lo:,}--{hi:,}").replace(",", "{,}")
            return f"{f(rr[0].min(), rr[0].max())} and {f(rr[1].min(), rr[1].max())} of 15{{,}}326"
        if d == "ronin_to_nomad":
            if (x.hits_top8 == 8).all():
                return f"1--8 of 1{{,}}070 ({len(x)}/{len(x)} seeds)"
            return f"{x.hits_top8.mean():.1f} of first 8 (mean); first at rank {int(x.first_rank.min())}--{int(x.first_rank.max())}"
        return f"first exploit at rank {int(x.first_rank.min()):,}--{int(x.first_rank.max()):,}".replace(",", "{,}")

    L = []
    L.append("\\multicolumn{3}{l}{\\emph{Ronin$\\to$Nomad: rank Nomad's 1{,}070 alerts (463 exploits, 272 addresses)}}\\\\")
    L.append(f"Learned, gradient boosting (2 positives) & {ap('ronin_to_nomad', 'learned')} & {ranks('ronin_to_nomad', 'learned')} \\\\")
    if len(kb):
        L.append(f"Learned, BridgeMamba-KAN (2 positives) & {ap('ronin_to_nomad', 'kan', True)} & {kranks('ronin_to_nomad', 'kan')} \\\\")
    L.append(f"\\quad trained on Ronin exploit 1 only & {ap('ronin_to_nomad', 'learned_only_pos0_noES')} & {ranks('ronin_to_nomad', 'learned_only_pos0_noES')} \\\\")
    L.append(f"\\quad trained on Ronin exploit 2 only & {ap('ronin_to_nomad', 'learned_only_pos1_noES')} & {ranks('ronin_to_nomad', 'learned_only_pos1_noES')} \\\\")
    L.append(f"USD prior (no training) & {ap('ronin_to_nomad', 'usd_prior')} & {ranks('ronin_to_nomad', 'usd_prior')} \\\\")
    L.append(f"Fusion $h_{{0.5}}$ (Eq.~\\ref{{eq:fusion}}) & {ap('ronin_to_nomad', 'hybrid_a0.5')} & {ranks('ronin_to_nomad', 'hybrid_a0.5')} \\\\")
    L.append(f"Fusion $h_{{\\max}}$ (Eq.~\\ref{{eq:fusion}}) & {ap('ronin_to_nomad', 'hybrid_max')} & {ranks('ronin_to_nomad', 'hybrid_max')} \\\\")
    if len(kb):
        L.append(f"Fusion $h_{{0.5}}$, BridgeMamba-KAN source & {ap('ronin_to_nomad', 'kan_hybrid_a0.5', True)} & {kranks('ronin_to_nomad', 'kan_hybrid_a0.5')} \\\\")
        L.append(f"Fusion $h_{{\\max}}$, BridgeMamba-KAN source & {ap('ronin_to_nomad', 'kan_hybrid_max', True)} & {kranks('ronin_to_nomad', 'kan_hybrid_max')} \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{3}{l}{\\emph{Nomad$\\to$Ronin: rank Ronin's 15{,}326 alerts (2 exploits, 1 address)}}\\\\")
    L.append(f"Learned, gradient boosting (463 positives) & {ap('nomad_to_ronin', 'learned')} & {ranks('nomad_to_ronin', 'learned')} \\\\")
    if len(kb):
        L.append(f"Learned, BridgeMamba-KAN (463 positives) & {ap('nomad_to_ronin', 'kan', True)} & {kranks('nomad_to_ronin', 'kan')} \\\\")
    L.append(f"USD prior (no training) & {ap('nomad_to_ronin', 'usd_prior')} & {ranks('nomad_to_ronin', 'usd_prior')} \\\\")
    L.append(f"Raw amounts (ablation) & {ap('nomad_to_ronin', 'raw_amount')} & 4, $\\sim$9{{,}}500 of 15{{,}}326 \\\\")
    L.append(f"Fusion $h_{{0.5}}$ & {ap('nomad_to_ronin', 'hybrid_a0.5')} & first exploit at rank {int(r[(r.direction == 'nomad_to_ronin') & (r.method == 'hybrid_a0.5')].first_rank.max())} \\\\")
    L.append(f"Fusion $h_{{\\max}}$ & {ap('nomad_to_ronin', 'hybrid_max')} & {ranks('nomad_to_ronin', 'hybrid_max')} \\\\")
    if len(kb):
        L.append(f"Fusion $h_{{0.5}}$, BridgeMamba-KAN source & {ap('nomad_to_ronin', 'kan_hybrid_a0.5', True)} & {kranks('nomad_to_ronin', 'kan_hybrid_a0.5')} \\\\")
        L.append(f"Fusion $h_{{\\max}}$, BridgeMamba-KAN source & {ap('nomad_to_ronin', 'kan_hybrid_max', True)} & {kranks('nomad_to_ronin', 'kan_hybrid_max')} \\\\")
    (TEX / "table4_rows.tex").write_text("\n".join(L), encoding="utf-8")
    a = pd.read_csv(RES / "e2_alpha.csv").groupby(["direction", "alpha"]).ap.agg(["mean", "std"])
    say("ALPHA sweep\n" + a.round(4).to_string())


# ------------------------------------------------------------------ Table 5: accounts
def table5():
    if not (RES / "e3_runs.csv").exists():
        return
    r = pd.read_csv(RES / "e3_runs.csv")
    L = []
    for m in ["BridgeMamba-KAN", "RandomForest", "HistGradientBoosting", "LogisticRegression"]:
        x = r[r.model == m]
        lab = {"RandomForest": "Random forest", "HistGradientBoosting": "Gradient boosting",
               "LogisticRegression": "Logistic regression"}.get(m, m)
        L.append(f"{lab} & {ms(x.pr_auc)} & {ms(x.roc_auc, 4)} & {ms(x['p@100'], 2)} & {ms(x['p@250'], 2)} & "
                 f"{ms(x['r@250'], 2)} & {ms(x['r@500'], 2)} & {ms(x.f1_matched, 3)} \\\\")
        say("T5", m, "PR", round(x.pr_auc.mean(), 4), "sd", round(x.pr_auc.std(ddof=1), 4), "CI", np.round(ci95(x.pr_auc), 4),
            "ROC", round(x.roc_auc.mean(), 4), "P@100", round(x['p@100'].mean(), 3), "P@250", round(x['p@250'].mean(), 3),
            "R@250", round(x['r@250'].mean(), 3), "R@500", round(x['r@500'].mean(), 3), "matched", round(x.f1_matched.mean(), 3),
            "seed42", x[x.seed == 42].pr_auc.round(4).tolist(), "n", len(x))
    (TEX / "table5_rows.tex").write_text("\n".join(L), encoding="utf-8")
    if (RES / "e3_paired.csv").exists():
        say("PAIRED\n" + pd.read_csv(RES / "e3_paired.csv").round(4).to_string())


# ------------------------------------------------------------------ federated, anomaly
def federated():
    if not list(RES.glob("e4_runs*.csv")):
        return
    r = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("e4_runs*.csv"))]).drop_duplicates(["seed", "condition", "aggregator", "round"])
    fin = r[r["round"] == r["round"].max()]
    g = fin.groupby(["condition", "aggregator"]).pr_auc.agg(["mean", "std", "count"])
    say("FED final round\n" + g.round(4).to_string())
    L = []
    cond = {"clean": "Clean", "label_flip": "Label flip (client 1)", "model_replacement": "Model replacement ($\\times$4)"}
    for c in ["clean", "label_flip", "model_replacement"]:
        cells = [ms(fin[(fin.condition == c) & (fin.aggregator == a)].pr_auc) for a in ["mean", "median", "trimmed"]]
        L.append(f"{cond[c]} & " + " & ".join(cells) + " \\\\")
    (TEX / "table_fed_rows.tex").write_text("\n".join(L), encoding="utf-8")


def anomaly():
    if not (RES / "e5_anomaly.csv").exists():
        return
    r = pd.read_csv(RES / "e5_anomaly.csv")
    g = r.groupby(["protocol", "score"])[["pr_auc", "roc_auc"]].agg(["mean", "std", "count"])
    say("ANOMALY\n" + g.round(4).to_string())


if __name__ == "__main__":
    for fn in (table3, table4, table5, federated, anomaly):
        try:
            fn()
        except FileNotFoundError as e:
            say("missing", e)
    (RES / "numbers.txt").write_text("\n".join(log), encoding="utf-8")
