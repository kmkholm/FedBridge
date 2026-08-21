"""
Stage 2b — account-level history sequences.

For every labeled transaction, build the ordered sequence of the SAME
account's bridge transactions up to and including it (max HIST_LEN steps).
Per-step features (label-free, computable in real time at inference):

  0  log10(tx value + 1)
  1  n_events in the decoded trace
  2  log10(total traced amount + 1)
  3  hour of day / 23
  4  log10(seconds since account's previous tx + 1)   (0 for first)
  5..13  event-type histogram (9 types) of the trace

Account key: tx sender (`from`). Sample label = label of the final step.
Output: processed/histories.npz aligned with transactions_refined order
        (benign/attack/suspicious only).
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"

HIST_LEN = 32
EVENT_TYPES = [
    "sc_deposit", "sc_withdrawal", "sc_token_deposited", "sc_token_withdrew",
    "tc_deposit", "tc_withdrawal", "tc_token_deposited", "tc_token_withdrew",
    "erc20_transfer",
]
N_FEAT = 5 + len(EVENT_TYPES)

txs = pd.read_parquet(OUT / "transactions_refined.parquet")
events = pd.read_parquet(OUT / "events.parquet",
                         columns=["tx_hash", "event_type", "amount_f"])

# per-tx trace summary: event-type histogram + total amount
hist = (events.assign(one=1)
        .pivot_table(index="tx_hash", columns="event_type", values="one",
                     aggfunc="sum", fill_value=0)
        .reindex(columns=EVENT_TYPES, fill_value=0))
amt = events.groupby("tx_hash")["amount_f"].sum().rename("trace_amount")

keep = txs[txs.label_refined.isin(["benign", "attack", "suspicious"])].copy()
keep = keep.merge(hist, on="tx_hash", how="left").merge(amt, on="tx_hash", how="left")
keep[EVENT_TYPES] = keep[EVENT_TYPES].fillna(0)
keep["trace_amount"] = keep["trace_amount"].fillna(0)

keep["log_value"] = np.log10(keep.value_f.fillna(0).clip(lower=0) + 1)
keep["log_trace_amount"] = np.log10(keep.trace_amount.clip(lower=0) + 1)
keep["hour"] = pd.to_datetime(keep.timestamp, unit="s").dt.hour / 23.0
keep["n_ev"] = keep["n_events"].fillna(0)

keep = keep.sort_values(["from", "timestamp"], kind="stable").reset_index(drop=True)
keep["dt_prev"] = keep.groupby("from")["timestamp"].diff()
keep["log_dt"] = np.log10(keep["dt_prev"].fillna(0).clip(lower=0) + 1)

step_feats = keep[["log_value", "n_ev", "log_trace_amount", "hour", "log_dt"]
                  + EVENT_TYPES].to_numpy(dtype=np.float32)

n = len(keep)
X = np.zeros((n, HIST_LEN, N_FEAT), dtype=np.float32)
L = np.zeros(n, dtype=np.int16)

start = 0
from_vals = keep["from"].to_numpy()
for i in range(n):
    if i > 0 and from_vals[i] != from_vals[i - 1]:
        start = i
    lo = max(start, i - HIST_LEN + 1)
    w = step_feats[lo:i + 1]
    X[i, -len(w):] = w          # right-aligned so "now" is always the last step
    L[i] = len(w)

y = (keep.label_refined == "attack").to_numpy(np.int8)
susp = (keep.label_refined == "suspicious").to_numpy(np.int8)

np.savez_compressed(
    OUT / "histories.npz",
    X=X, lengths=L, y=y, suspicious=susp,
    bridge_id=(keep.bridge == "nomad-bridge").to_numpy(np.int8),
    timestamp=keep.timestamp.to_numpy(),
    tx_hash=keep.tx_hash.to_numpy().astype("U66"),
    account=keep["from"].to_numpy().astype("U42"),
)

trainable = ~susp.astype(bool)
print(f"samples: {n:,} | trainable {trainable.sum():,} "
      f"(attack {int(y[trainable].sum()):,})")
print(f"history length: mean {L.mean():.1f}, median {np.median(L):.0f}, "
      f"full({HIST_LEN}) {(L == HIST_LEN).mean() * 100:.1f}%")
atk = L[y == 1]
print(f"attack-sample history length: mean {atk.mean():.1f}, median {np.median(atk):.0f}")
print(f"wrote {OUT / 'histories.npz'}")
