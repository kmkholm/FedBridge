"""
Stage 2c — bridge-level context features (label-free, real-time computable).

Motivation: exploit waves are unmissable at the bridge level even when each
attacking wallet is brand new (Nomad copycats: fresh EOAs, median account
history = 2). For every transaction we compute, over trailing windows of
1 hour and 24 hours on the SAME bridge:

  n_tx_w        log1p count of bridge txs in window
  n_wd_w        log1p count of withdrawal-type txs in window
  wd_ratio_w    withdrawal txs / all txs in window
  amt_w         log10 total traced amount in window
  new_acct_w    fraction of window txs from never-seen-before accounts

10 features (5 x 2 windows) -> processed/context.parquet keyed by tx_hash.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"

WINDOWS = {"1h": 3600, "24h": 86400}
WITHDRAW_TYPES = {"sc_withdrawal", "sc_token_withdrew", "tc_withdrawal", "tc_token_withdrew"}

txs = pd.read_parquet(OUT / "transactions_refined.parquet")
events = pd.read_parquet(OUT / "events.parquet", columns=["tx_hash", "event_type", "amount_f"])

is_wd = (events[events.event_type.isin(WITHDRAW_TYPES)]
         .groupby("tx_hash").size().rename("n_wd_events"))
amt = events.groupby("tx_hash")["amount_f"].sum().rename("trace_amount")

df = txs[["tx_hash", "bridge", "timestamp", "from"]].merge(
    is_wd, on="tx_hash", how="left").merge(amt, on="tx_hash", how="left")
df["is_wd"] = (df["n_wd_events"].fillna(0) > 0).astype(np.float64)
df["amt"] = df["trace_amount"].fillna(0).clip(lower=0)
df = df.sort_values(["bridge", "timestamp"], kind="stable").reset_index(drop=True)

# first-seen flag per account (within bridge)
df["new_acct"] = (~df.duplicated(subset=["bridge", "from"])).astype(np.float64)

out_cols = {}
for wname, wsec in WINDOWS.items():
    for col in ["n_tx", "n_wd", "wd_ratio", "amt", "new_acct"]:
        out_cols[f"{col}_{wname}"] = np.zeros(len(df), dtype=np.float32)

for bridge, g in df.groupby("bridge", sort=False):
    ts = g.timestamp.to_numpy(dtype=np.float64)
    idx = g.index.to_numpy()
    ones = np.ones(len(g))
    series = {"n_tx": ones, "n_wd": g.is_wd.to_numpy(),
              "amt": g.amt.to_numpy(), "new_acct": g.new_acct.to_numpy()}
    cum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in series.items()}
    for wname, wsec in WINDOWS.items():
        lo = np.searchsorted(ts, ts - wsec, side="left")
        hi = np.arange(1, len(g) + 1)  # inclusive of current tx
        n_tx = cum["n_tx"][hi] - cum["n_tx"][lo]
        n_wd = cum["n_wd"][hi] - cum["n_wd"][lo]
        w_amt = cum["amt"][hi] - cum["amt"][lo]
        n_new = cum["new_acct"][hi] - cum["new_acct"][lo]
        out_cols[f"n_tx_{wname}"][idx] = np.log1p(n_tx)
        out_cols[f"n_wd_{wname}"][idx] = np.log1p(n_wd)
        out_cols[f"wd_ratio_{wname}"][idx] = n_wd / np.maximum(n_tx, 1)
        out_cols[f"amt_{wname}"][idx] = np.log10(w_amt + 1)
        out_cols[f"new_acct_{wname}"][idx] = n_new / np.maximum(n_tx, 1)

ctx = pd.DataFrame({"tx_hash": df.tx_hash, **out_cols})
ctx.to_parquet(OUT / "context.parquet", index=False)

# sanity: context at the exploit moments vs global mean
chk = ctx.merge(txs[["tx_hash", "label_refined", "bridge"]], on="tx_hash")
print("mean wd_ratio_1h / new_acct_1h / n_wd_1h by refined label:")
print(chk.groupby("label_refined")[["wd_ratio_1h", "new_acct_1h", "n_wd_1h"]]
         .mean().round(3).to_string())
print(f"\nwrote {OUT / 'context.parquet'} ({len(ctx):,} rows)")
