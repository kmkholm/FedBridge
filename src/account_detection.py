"""Account-level exploiter detection on the Across corpus.

Reframing rationale. Predicting an individual mispriced transfer from
deposit-side data is close to predicting a defect in a destination-side price
registry from the depositor's conduct, and performs accordingly. The
operationally meaningful question is whether an *account* is systematically
harvesting the defect: 1,284 value-conservation failures originate from 318
depositors, 138 of whom repeat, and those repeat accounts show a 15.2%
violation rate across their activity. An account is a sequence of transactions,
which is the object the selective state-space encoder is designed to classify.

Task: given an account's ordered transaction history, classify it as an
exploiter (>= MIN_VIOL conservation failures) or not.
Protocol: temporal -- accounts are assigned by the time of their first
transaction, so test accounts are entirely unseen during training.
Models: BridgeMamba-KAN over the account sequence, versus gradient boosting,
logistic regression and a random-forest baseline over pooled account features.

Outputs: processed_ext/acct_results.csv, acct_scores.npz
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import StandardScaler

from model import BridgeMambaKAN

PROJECT = Path(__file__).resolve().parents[1]
EXT = PROJECT / "processed_ext"
SEQ_LEN, SEED = 24, 42
MIN_VIOL = 1          # an account is an exploiter if it has >= this many
MIN_TXS = 3           # accounts must have at least this many transactions
torch.manual_seed(SEED)
np.random.seed(SEED)
num = lambda s: pd.to_numeric(s, errors="coerce")


def load():
    d = pd.read_parquet(EXT / "across_full.parquet")
    i, o = num(d.input_amount_usd), num(d.output_amount_usd)
    fee = num(d.src_fee_usd).fillna(0)
    ts = pd.to_datetime(num(d.src_timestamp), unit="s", errors="coerce")
    ok = i.notna() & o.notna() & (i > 0) & ts.notna()
    f = pd.DataFrame({
        "t": ts[ok], "acct": d.depositor[ok].astype(str),
        "in_usd": i[ok], "fee": fee[ok],
        "src": d.src_blockchain[ok].astype(str),
        "dst": d.dst_blockchain[ok].astype(str),
        "viol": ((fee[ok] + i[ok] - o[ok]) < 0).astype(int),
    })
    for c, name in [("quote_timestamp", "quote_ts"), ("fill_deadline", "fill_dl"),
                    ("exclusivity_deadline", "excl_dl")]:
        if c in d.columns:
            f[name] = num(d[c])[ok].to_numpy()
    return f.sort_values(["acct", "t"]).reset_index(drop=True)


def per_tx_features(d):
    """Deposit-side per-transaction features. The destination amount, which
    defines a violation, is never used."""
    d = d.copy()
    epoch = d.t.astype("int64") // 10**9
    d["log_in"] = np.log10(d.in_usd + 1)
    d["fee_ratio"] = (d.fee / d.in_usd.clip(lower=1e-9)).clip(0, 1)
    d["hour"] = d.t.dt.hour / 23.0
    g = d.groupby("acct", sort=False)
    d["log_dt"] = np.log10(g.t.diff().dt.total_seconds().fillna(0).clip(lower=0) + 1)
    route = d.src + ">" + d.dst
    d["route_freq"] = np.log10(route.map(route.value_counts(normalize=True)) + 1e-6)
    if "quote_ts" in d.columns:
        d["quote_stale"] = np.log10((epoch - d.quote_ts).clip(0, 86400) + 1)
    if "fill_dl" in d.columns:
        d["fill_window"] = np.log10((d.fill_dl - epoch).clip(0, 86400) + 1)
    if "excl_dl" in d.columns:
        d["excl_window"] = np.log10((d.excl_dl - epoch).clip(0, 86400) + 1)
    cols = [c for c in ["log_in", "fee_ratio", "hour", "log_dt", "route_freq",
                        "quote_stale", "fill_window", "excl_window"] if c in d.columns]
    d[cols] = d[cols].replace([np.inf, -np.inf], 0).fillna(0)
    return d, cols


def build_accounts(d, cols):
    """One row per account: its label, its first-seen time, a fixed-length
    sequence of its most recent transactions, and pooled summary features."""
    g = d.groupby("acct", sort=False)
    keep = g.size()[g.size() >= MIN_TXS].index
    d = d[d.acct.isin(keep)].copy()
    g = d.groupby("acct", sort=False)

    meta = g.agg(n_tx=("in_usd", "size"), first_t=("t", "min"), last_t=("t", "max"),
                 viol=("viol", "sum"), mean_in=("in_usd", "mean"),
                 max_in=("in_usd", "max"), std_in=("in_usd", "std"),
                 mean_fee=("fee", "mean"), n_src=("src", "nunique"),
                 n_dst=("dst", "nunique"))
    meta["y"] = (meta.viol >= MIN_VIOL).astype(np.float32)
    meta["span_h"] = (meta.last_t - meta.first_t).dt.total_seconds() / 3600
    meta["rate_per_h"] = meta.n_tx / meta.span_h.clip(lower=0.5)
    meta["log_mean_in"] = np.log10(meta.mean_in + 1)
    meta["log_max_in"] = np.log10(meta.max_in + 1)
    meta["cv_in"] = (meta.std_in / meta.mean_in.clip(lower=1e-9)).fillna(0).clip(0, 10)
    meta["log_n_tx"] = np.log10(meta.n_tx)
    meta["fee_share"] = (meta.mean_fee / meta.mean_in.clip(lower=1e-9)).clip(0, 1)

    pooled_cols = ["log_n_tx", "log_mean_in", "log_max_in", "cv_in", "fee_share",
                   "n_src", "n_dst", "rate_per_h"]
    # append per-transaction feature means/max for the account
    for c in cols:
        meta[f"m_{c}"] = g[c].mean()
        meta[f"x_{c}"] = g[c].max()
        pooled_cols += [f"m_{c}", f"x_{c}"]

    # sequences: last SEQ_LEN transactions, right aligned
    order = {a: k for k, a in enumerate(meta.index)}
    X = np.zeros((len(meta), SEQ_LEN, len(cols)), np.float32)
    M = np.zeros((len(meta), SEQ_LEN), bool)
    feats = d[cols].to_numpy(np.float32)
    acct = d.acct.to_numpy()
    bounds = {}
    start = 0
    for k in range(len(d)):
        if k and acct[k] != acct[k - 1]:
            bounds[acct[k - 1]] = (start, k)
            start = k
    bounds[acct[-1]] = (start, len(d))
    for a, (s, e) in bounds.items():
        k = order[a]
        w = feats[max(s, e - SEQ_LEN):e]
        X[k, -len(w):] = w
        M[k, -len(w):] = True
    return meta, X, M, meta[pooled_cols].replace([np.inf, -np.inf], 0)\
        .fillna(0).to_numpy(np.float32), pooled_cols


def rec_at_fpr(y, s, f=0.001):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def train_kan(Xtr, Mtr, Ctr, ytr, Xte, Mte, Cte, epochs=10, bs=512):
    net = BridgeMambaKAN(d_feat=Xtr.shape[2], d_ctx=Ctr.shape[1], d_model=48)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    pw = torch.tensor(float(min((ytr == 0).sum() / max(ytr.sum(), 1), 100.0)))
    xt, mt, ct, yt = map(torch.from_numpy, (Xtr, Mtr, Ctr, ytr))
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(xt[b], mt[b], ct[b]),
                                                      yt[b], pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        print(f"    epoch {ep+1}/{epochs}: loss {tot/len(xt):.4f}", flush=True)
    net.eval()
    o = []
    with torch.no_grad():
        for i in range(0, len(Xte), 2048):
            o.append(torch.sigmoid(net(torch.from_numpy(Xte[i:i+2048]),
                                       torch.from_numpy(Mte[i:i+2048]),
                                       torch.from_numpy(Cte[i:i+2048]))))
    return torch.cat(o).numpy()


if __name__ == "__main__":
    d = load()
    d, cols = per_tx_features(d)
    print(f"per-transaction features ({len(cols)}): {cols}", flush=True)
    meta, X, M, C, pooled = build_accounts(d, cols)
    y = meta.y.to_numpy(np.float32)
    print(f"accounts with >= {MIN_TXS} txs: {len(meta):,} | "
          f"exploiters: {int(y.sum()):,} ({100*y.mean():.3f}%)", flush=True)

    cut = meta.first_t.quantile(0.7)
    tr = (meta.first_t <= cut).to_numpy()
    te = ~tr
    print(f"train accounts {tr.sum():,} ({int(y[tr].sum())} exploiters) | "
          f"test {te.sum():,} ({int(y[te].sum())} exploiters, "
          f"{100*y[te].mean():.3f}%)", flush=True)

    sc = StandardScaler().fit(C[tr])
    Cs = sc.transform(C).astype(np.float32)
    mu, sd = np.nanmean(X[tr].reshape(-1, X.shape[2]), 0), \
             np.nanstd(X[tr].reshape(-1, X.shape[2]), 0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)

    s_kan = train_kan(Xs[tr], M[tr], Cs[tr], y[tr], Xs[te], M[te], Cs[te])
    s_hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)\
        .fit(Cs[tr], y[tr]).predict_proba(Cs[te])[:, 1]
    s_rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED)\
        .fit(Cs[tr], y[tr]).predict_proba(Cs[te])[:, 1]
    s_lr = LogisticRegression(max_iter=2000, class_weight="balanced")\
        .fit(Cs[tr], y[tr]).predict_proba(Cs[te])[:, 1]

    yte = y[te]
    rows = []
    for nm, s in [("BridgeMamba-KAN", s_kan), ("HistGradientBoosting", s_hgb),
                  ("RandomForest", s_rf), ("LogisticRegression", s_lr)]:
        thr = np.quantile(s, 1 - yte.mean())        # flag as many as are positive
        pred = (s >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="binary",
                                                      zero_division=0)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        rows.append(dict(model=nm, pr_auc=average_precision_score(yte, s),
                         roc_auc=roc_auc_score(yte, s),
                         precision=p, recall=r, f1=f1,
                         recall_at_fpr=rec_at_fpr(yte, s),
                         tp=tp, fp=fp, fn=fn, tn=tn,
                         n_test=len(yte), test_pos=int(yte.sum()),
                         prevalence=float(yte.mean())))
        print(f"  {nm:<22} PR {rows[-1]['pr_auc']:.4f}  ROC {rows[-1]['roc_auc']:.4f}"
              f"  P {p:.3f} R {r:.3f} F1 {f1:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(EXT / "acct_results.csv", index=False)
    np.savez(EXT / "acct_scores.npz", y=yte, kan=s_kan, hgb=s_hgb, rf=s_rf, lr=s_lr)
    print(f"\nwrote {EXT / 'acct_results.csv'}", flush=True)
