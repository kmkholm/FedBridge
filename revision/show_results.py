"""Print the headline ten-seed results of the revision, read from results/*.csv."""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
k = pd.read_csv(R / "e1b_model.csv")
t = pd.read_csv(R / "e2_runs.csv")
kb = pd.read_csv(R / "e2b_runs.csv")
a = pd.read_csv(R / "e3_runs.csv")
f = lambda v, d=3: f"{v.mean():.{d}f} +/- {v.std(ddof=1):.{d}f}"

rnd = k[k.protocol == "random"]
seeds = sorted(int(x) for x in rnd.seed.unique())
print(f"BridgeMamba-KAN, {len(seeds)} seeds (" + ("42 and 0-8" if seeds == list(range(9)) + [42] else ", ".join(map(str, seeds))) + ")")
print(f"  random split            PR-AUC {f(rnd.pr_auc)}   ROC {rnd.roc_auc.mean():.4f}")
h = kb[(kb.direction == "ronin_to_nomad") & (kb.method == "kan_hybrid_a0.5")]
print(f"  zero-day Ronin->Nomad   AP     {f(h.ap)}   exploits at ranks 1-8 in {(h.hits_top8 == 8).sum()}/{len(h)} seeds")
m = t[(t.direction == "nomad_to_ronin") & (t.method == "hybrid_max")]
print(f"  zero-day Nomad->Ronin   both exploits at ranks {m.ranks.iloc[0].replace(' ', ' and ')} of 15,326 alerts (h_max)")
row = lambda n: a[a.model == n].pr_auc
print(f"  exploiter accounts      PR-AUC {f(row('BridgeMamba-KAN'))}   "
      f"(RF {row('RandomForest').mean():.3f}, GB {row('HistGradientBoosting').mean():.3f}, LR {row('LogisticRegression').mean():.3f})")
