"""Write LaTeX number macros (results/tex/numbers_macros.tex) used in the V2 text.
Missing results produce a visible placeholder (\\textbf{??}) so nothing silently goes stale."""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "results"
M = {}


def pm(v, d=3):
    v = np.asarray(v, float)
    return f"${v.mean():.{d}f}\\pm{v.std(ddof=1):.{d}f}$" if v.std(ddof=1) > 1e-12 else f"{v.mean():.{d}f}"


def mean(v, d=3):
    return f"{np.asarray(v, float).mean():.{d}f}"


def put(k, fn):
    try:
        M[k] = fn()
    except Exception:
        M[k] = r"\textbf{??}"


b = pd.read_csv(RES / "e1a_baselines.csv") if (RES / "e1a_baselines.csv").exists() else None
k = pd.read_csv(RES / "e1b_model.csv") if (RES / "e1b_model.csv").exists() else None
sel = lambda p, m, f: b[(b.protocol == p) & (b.model == m) & (b.features == f)]
kp = lambda p: k[k.protocol == p]

put("GBrandom", lambda: pm(sel("random", "hgb", "tx+ctx+h").pr_auc))
put("GBrandomRecall", lambda: mean(sel("random", "hgb", "tx+ctx+h").recall_at_fpr, 2))
put("GBtemporal", lambda: pm(sel("temporal", "hgb", "tx+ctx").pr_auc))
put("GBtemporalRecall", lambda: mean(sel("temporal", "hgb", "tx+ctx").recall_at_fpr, 2))
put("LRtemporal", lambda: mean(sel("temporal", "logreg", "tx+ctx").pr_auc))
put("GBlobonomad", lambda: pm(sel("lobo_nomad", "hgb", "tx+ctx").pr_auc))
put("KANrandom", lambda: pm(kp("random").pr_auc))
put("KANrandomROC", lambda: mean(kp("random").roc_auc, 4))
put("KANrandomRecall", lambda: mean(kp("random").recall_at_fpr, 3))
put("KANtemporal", lambda: pm(kp("temporal").pr_auc))
put("KANtemporalMin", lambda: f"{kp('temporal').pr_auc.min():.2f}")
put("KANtemporalMax", lambda: f"{kp('temporal').pr_auc.max():.2f}")
put("KANloboronin", lambda: (lambda v: (_ for _ in ()).throw(ValueError()) if not np.isfinite(v) else f"${v / 10 ** np.floor(np.log10(v)):.1f}\\times10^{{{int(np.floor(np.log10(v)))}}}$")(kp("lobo_ronin").pr_auc.mean()))
put("KANlobonomad", lambda: pm(kp("lobo_nomad").pr_auc))

if (RES / "e5_anomaly.csv").exists():
    a = pd.read_csv(RES / "e5_anomaly.csv")
    ap_ = lambda p, s_: a[(a.protocol == p) & (a.score == s_)]
    put("AEtemporalROC", lambda: pm(ap_("temporal", "AE reconstruction").roc_auc, 2))
    put("AElobonomadROC", lambda: pm(ap_("lobo_nomad", "AE reconstruction").roc_auc, 2))
    put("MahalTemporal", lambda: pm(ap_("temporal", "latent Mahalanobis").pr_auc, 2))
    put("CtxTemporal", lambda: mean(ap_("temporal", "context Mahalanobis").pr_auc, 2))
    put("KNNtemporal", lambda: pm(ap_("temporal", "latent kNN").pr_auc, 2))

if (RES / "e3_runs.csv").exists():
    r = pd.read_csv(RES / "e3_runs.csv")
    for key, name in [("KAN", "BridgeMamba-KAN"), ("RF", "RandomForest"), ("GB", "HistGradientBoosting"), ("LR", "LogisticRegression")]:
        x = r[r.model == name]
        put(f"{key}acct", lambda x=x: pm(x.pr_auc))
        put(f"{key}acctROC", lambda x=x: mean(x.roc_auc, 4))
        put(f"{key}acctPhundred", lambda x=x: mean(x["p@100"], 2))
        put(f"{key}acctRtwofifty", lambda x=x: mean(x["r@250"], 2))
        put(f"{key}acctRfivehundred", lambda x=x: mean(x["r@500"], 2))
    if (RES / "e3_paired.csv").exists():
        pr = pd.read_csv(RES / "e3_paired.csv").set_index("comparison")
        for key, name in [("RF", "RandomForest"), ("GB", "HistGradientBoosting"), ("LR", "LogisticRegression")]:
            row = pr.loc[f"BridgeMamba-KAN vs {name}"]
            put(f"pairWins{key}", lambda row=row: f"{int(row.wins)}")
            put(f"pairP{key}", lambda row=row: f"{row.wilcoxon_p:.3f}" if row.wilcoxon_p >= 0.001 else "<0.001")
            put(f"pairLo{key}", lambda row=row: f"{row.boot_lo:.3f}")
            put(f"pairHi{key}", lambda row=row: f"{row.boot_hi:.3f}")

if (RES / "e2b_runs.csv").exists():
    kb = pd.read_csv(RES / "e2b_runs.csv")
    q = lambda d, m: kb[(kb.direction == d) & (kb.method == m)]
    put("KANtriageNomad", lambda: pm(q("ronin_to_nomad", "kan").ap))
    put("KANtriageNomadTop", lambda: mean(q("ronin_to_nomad", "kan").hits_top20, 1))
    put("KANtriageRoninFirst", lambda: f"{int(q('nomad_to_ronin', 'kan').first_rank.min()):,}".replace(",", "{,}"))
    put("KANtriageMaxNomad", lambda: pm(q("ronin_to_nomad", "kan_hybrid_max").ap))
    put("KANtriageMaxRonin", lambda: f"{q('nomad_to_ronin', 'kan_hybrid_max').ranks.iloc[0]}")

if list(RES.glob("e4_runs*.csv")):
    f = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("e4_runs*.csv"))]).drop_duplicates(["seed", "condition", "aggregator", "round"])
    fin = f[f["round"] == f["round"].max()]
    for c, ck in [("clean", "Clean"), ("label_flip", "Flip"), ("model_replacement", "Repl")]:
        for g, gk in [("mean", "Mean"), ("median", "Median"), ("trimmed", "Trim")]:
            put(f"FL{ck}{gk}", lambda c=c, g=g: pm(fin[(fin.condition == c) & (fin.aggregator == g)].pr_auc))

EXPECTED = ["AEtemporalROC", "AElobonomadROC", "MahalTemporal", "CtxTemporal", "KNNtemporal", "KANtriageNomad", "KANtriageNomadTop", "KANtriageRoninFirst", "KANtriageMaxNomad", "KANtriageMaxRonin"] + [f"FL{c}{g}" for c in ("Clean", "Flip", "Repl") for g in ("Mean", "Median", "Trim")]
for k_ in EXPECTED:
    M.setdefault(k_, r"\textbf{??}")

lines = [f"\\newcommand{{\\{k_}}}{{{v}}}" for k_, v in M.items()]
(RES / "tex").mkdir(exist_ok=True)
(RES / "tex" / "numbers_macros.tex").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
