"""E2 -- Table 4 zero-day alert triage, repeated over seeds, with the analyses the
reviewers asked for.

Alert universe and features are those of src/triage_experiment.py:
rule-engine alerts (attack or suspicious), features = final-step history + bridge
context + log USD value, learned ranker = HistGradientBoosting(class_weight="balanced").

Adds:
  * seeds 0-9 for the learned ranker (the USD and raw-amount priors are deterministic)
  * raw-amount ablation (sum of undecoded token amounts, no decimals or prices)
  * hybrid rank fusion  h = a*pct(model) + (1-a)*pct(USD),  a = 0.5 fixed a priori,
    plus the full sweep a in {0, 0.1, ..., 1}; and the max rule h = max(pct(model), pct(USD))
  * exploiter-address level evaluation (score of an address = max over its alerts)
  * address-cluster bootstrap 95% CI for the Nomad-side average precision
  * leave-one-positive-out: Ronin->Nomad trained on only one of the two Ronin exploits

writes results/e2_runs.csv, e2_summary.csv, e2_alpha.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score as ap, roc_auc_score as roc

DATA = Path(os.environ.get("FB_DATA", Path(__file__).resolve().parents[1] / "processed")).expanduser()
RES = Path(__file__).resolve().parent / "results"
SEEDS = [42] + list(range(9))   # 42 = seed of the original submission
ALPHAS = np.round(np.linspace(0, 1, 11), 1)
rng_boot = np.random.default_rng(2026)

d = np.load(DATA / "histories.npz", allow_pickle=False)
X, y, susp = d["X"], d["y"].astype(int), d["suspicious"].astype(bool)
tx_hash, bridge_id, account = d["tx_hash"], d["bridge_id"], d["account"]
ctx = pd.read_parquet(DATA / "context.parquet").set_index("tx_hash")
CTX = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)
USD = pd.read_parquet(DATA / "usd.parquet").set_index("tx_hash")["log_usd"] \
    .reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)
ev = pd.read_parquet(DATA / "events.parquet", columns=["tx_hash", "amount_f"])
RAW = np.log10(ev.groupby("tx_hash")["amount_f"].sum().clip(lower=0)
               .reindex(tx_hash).fillna(0).to_numpy() + 1)
F = np.hstack([X[:, -1, :], CTX, USD[:, None]])
flagged = susp | (y == 1)
print(f"alert universe {flagged.sum():,} (exploits {y[flagged].sum()})", flush=True)

DIRS = {"ronin_to_nomad": (0, 1), "nomad_to_ronin": (1, 0)}   # (train bridge, test bridge)


def pct(s):
    return pd.Series(s).rank(pct=True, method="average").to_numpy()


def ranks(yt, s):
    r = np.argsort(np.argsort(-s, kind="stable"), kind="stable")[yt == 1] + 1
    return np.sort(r)


def metrics(yt, s, acc):
    r = ranks(yt, s)
    a = pd.DataFrame({"a": acc, "y": yt, "s": s}).groupby("a").agg(y=("y", "max"), s=("s", "max"))
    return dict(ap=ap(yt, s), roc=roc(yt, s), first_rank=int(r[0]),
                hits_top8=int((r <= 8).sum()), hits_top20=int((r <= 20).sum()),
                hits_top100=int((r <= 100).sum()), ranks=" ".join(map(str, r[:8])),
                addr_ap=ap(a.y, a.s), addr_hits_top20=int(a.sort_values("s", ascending=False).y.head(20).sum()))


def cluster_boot_ap(yt, s, acc, n=2000):
    """95% CI of AP resampling exploiter addresses (and benign alerts) as clusters."""
    groups = pd.Series(np.arange(len(yt))).groupby(acc).apply(list).tolist()
    vals = []
    for _ in range(n):
        pick = rng_boot.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        if yt[idx].sum() == 0 or yt[idx].sum() == len(idx):
            continue
        vals.append(ap(yt[idx], s[idx]))
    return np.percentile(vals, [2.5, 97.5])


runs, alpha_rows = [], []
for dname, (btr, bte) in DIRS.items():
    tr, te = flagged & (bridge_id == btr), flagged & (bridge_id == bte)
    yt, acc = y[te], account[te]
    base = dict(direction=dname, n_alerts=int(te.sum()), n_exploits=int(yt.sum()),
                n_exploit_addr=int(len(np.unique(acc[yt == 1]))))

    for name, s in [("usd_prior", USD[te]), ("raw_amount", RAW[te])]:
        runs.append({**base, "method": name, "seed": -1, "train_pos": 0, **metrics(yt, s, acc)})

    # (train indices, early_stopping). HGB's automatic early stopping needs >= 2 positives
    # for its stratified validation split, so the one-positive variants run without it and
    # are compared against an all-positives reference that also runs without it.
    variants = {"all": (np.flatnonzero(tr), "auto")}
    if dname == "ronin_to_nomad":        # leave-one-positive-out over the two Ronin exploits
        variants["all_noES"] = (np.flatnonzero(tr), False)
        pos = np.flatnonzero(tr & (y == 1))
        for k, p in enumerate(pos):
            variants[f"only_pos{k}_noES"] = (np.setdiff1d(np.flatnonzero(tr), np.delete(pos, k)), False)

    for vname, (idx, es) in variants.items():
        for seed in SEEDS:
            clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, early_stopping=es)
            s = clf.fit(F[idx], y[idx]).predict_proba(F[te])[:, 1]
            runs.append({**base, "method": "learned" if vname == "all" else f"learned_{vname}",
                         "seed": seed, "train_pos": int(y[idx].sum()), **metrics(yt, s, acc)})
            if vname == "all":
                pm, pu = pct(s), pct(USD[te])
                for a in ALPHAS:
                    h = a * pm + (1 - a) * pu
                    alpha_rows.append(dict(direction=dname, seed=seed, alpha=a, **metrics(yt, h, acc)))
                runs.append({**base, "method": "hybrid_a0.5", "seed": seed, "train_pos": int(y[idx].sum()),
                             **metrics(yt, 0.5 * pm + 0.5 * pu, acc)})
                # "either source" escalation: an alert ranks by the stronger of its two percentiles
                runs.append({**base, "method": "hybrid_max", "seed": seed, "train_pos": int(y[idx].sum()),
                             **metrics(yt, np.maximum(pm, pu), acc)})
        print(f"{dname} {vname} done", flush=True)

runs = pd.DataFrame(runs)
runs.to_csv(RES / "e2_runs.csv", index=False)
pd.DataFrame(alpha_rows).to_csv(RES / "e2_alpha.csv", index=False)

num = ["ap", "roc", "first_rank", "hits_top8", "hits_top20", "hits_top100", "addr_ap", "addr_hits_top20"]
summ = runs.groupby(["direction", "method"])[num].agg(["mean", "std", "min", "max"])
summ.columns = ["_".join(c) for c in summ.columns]
summ = summ.reset_index()

# address-cluster bootstrap CI on the Nomad side, deterministic prior and the seed-42 learned ranker
te = flagged & (bridge_id == 1)
yt, acc = y[te], account[te]
tr = flagged & (bridge_id == 0)
s0 = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEEDS[0]).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
ci = {("ronin_to_nomad", "usd_prior"): cluster_boot_ap(yt, USD[te], acc),
      ("ronin_to_nomad", "learned"): cluster_boot_ap(yt, s0, acc),
      ("ronin_to_nomad", "hybrid_a0.5"): cluster_boot_ap(yt, 0.5 * pct(s0) + 0.5 * pct(USD[te]), acc),
      ("ronin_to_nomad", "hybrid_max"): cluster_boot_ap(yt, np.maximum(pct(s0), pct(USD[te])), acc)}
summ["ap_ci_lo"] = [ci.get((r.direction, r.method), [np.nan, np.nan])[0] for r in summ.itertuples()]
summ["ap_ci_hi"] = [ci.get((r.direction, r.method), [np.nan, np.nan])[1] for r in summ.itertuples()]
summ.to_csv(RES / "e2_summary.csv", index=False)
pd.set_option("display.width", 250)
print(summ[["direction", "method", "ap_mean", "ap_std", "roc_mean", "first_rank_max", "hits_top8_mean",
            "hits_top20_mean", "addr_ap_mean", "ap_ci_lo", "ap_ci_hi"]].round(4).to_string(index=False))
