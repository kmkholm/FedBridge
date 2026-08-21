"""
Stage 2a — tokenize per-transaction event traces for sequence models.

Each transaction becomes a fixed-length sequence of events (sorted by
event_index). Per-event features:
  token id     event_type from a small vocabulary (+PAD/+OOV)
  log_amount   log10(amount+1), amounts are uint256 -> float64 (lossy, fine)
  contract id  token/asset contract bucketed into top-K vocab (+OOV)
  addl flag    1 if the event came from additional (RPC-recovered) data

Per-transaction context features: log tx value, status, hour-of-day, and
bridge/chain ids (bridge id is also the federated client id).

Output: processed/sequences.npz
  train classes only (benign=0, attack=1) + a parallel "suspicious" array
  kept aside for the label-noise analysis (E5/E6 discussion).
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"

MAX_LEN = 64          # covers >99% of traces; exploit txs are short
TOP_K_CONTRACTS = 255 # contract vocab size (0 = OOV/none)

EVENT_TYPES = [
    "sc_deposit", "sc_withdrawal", "sc_token_deposited", "sc_token_withdrew",
    "tc_deposit", "tc_withdrawal", "tc_token_deposited", "tc_token_withdrew",
    "erc20_transfer",
]
EVENT_VOCAB = {t: i + 1 for i, t in enumerate(EVENT_TYPES)}  # 0 = PAD

events = pd.read_parquet(OUT / "events.parquet",
                         columns=["tx_hash", "event_index", "event_type", "amount_f",
                                  "contract", "dst_token", "from_additional_data", "bridge"])
txs = pd.read_parquet(OUT / "transactions_refined.parquet")

# asset contract: erc20_transfer has `contract`; typed bridge events carry dst_token
events["asset"] = events["contract"].fillna(events["dst_token"]).fillna("none")
top = events["asset"].value_counts().head(TOP_K_CONTRACTS).index
asset_vocab = {a: i + 1 for i, a in enumerate(top)}  # 0 = OOV/none

events["tok"] = events["event_type"].map(EVENT_VOCAB).fillna(0).astype(np.int16)
events["asset_id"] = events["asset"].map(asset_vocab).fillna(0).astype(np.int16)
events["log_amount"] = np.log10(events["amount_f"].fillna(0).clip(lower=0) + 1).astype(np.float32)
events["addl"] = events["from_additional_data"].astype(np.float32)

keep = txs[txs.label_refined.isin(["benign", "attack", "suspicious"])].copy()
keep["y"] = (keep.label_refined == "attack").astype(np.int8)
keep["is_suspicious"] = (keep.label_refined == "suspicious").astype(np.int8)
keep["bridge_id"] = (keep.bridge == "nomad-bridge").astype(np.int8)  # 0=ronin, 1=nomad

events = events[events.tx_hash.isin(set(keep.tx_hash))]
events = events.sort_values(["tx_hash", "event_index"])

grouped = events.groupby("tx_hash", sort=False)
order = {h: i for i, h in enumerate(keep.tx_hash)}
n = len(keep)

tok = np.zeros((n, MAX_LEN), dtype=np.int16)
asset = np.zeros((n, MAX_LEN), dtype=np.int16)
feat = np.zeros((n, MAX_LEN, 2), dtype=np.float32)   # log_amount, addl
length = np.zeros(n, dtype=np.int16)

for h, g in grouped:
    i = order[h]
    L = min(len(g), MAX_LEN)
    length[i] = L
    tok[i, :L] = g["tok"].values[:L]
    asset[i, :L] = g["asset_id"].values[:L]
    feat[i, :L, 0] = g["log_amount"].values[:L]
    feat[i, :L, 1] = g["addl"].values[:L]

ts = pd.to_datetime(keep.timestamp, unit="s")
ctx = np.stack([
    np.log10(keep.value_f.fillna(0).clip(lower=0) + 1),
    keep.status.fillna(1),
    ts.dt.hour / 23.0,
    ts.dt.dayofweek / 6.0,
], axis=1).astype(np.float32)

np.savez_compressed(
    OUT / "sequences.npz",
    tokens=tok, assets=asset, feats=feat, lengths=length, ctx=ctx,
    y=keep.y.values, suspicious=keep.is_suspicious.values,
    bridge_id=keep.bridge_id.values, timestamp=keep.timestamp.values,
    tx_hash=keep.tx_hash.values.astype("U66"),
)

trainable = keep[keep.is_suspicious == 0]
print(f"sequences: {n:,} total | trainable {len(trainable):,} "
      f"(benign {sum(trainable.y == 0):,}, attack {sum(trainable.y == 1):,}) "
      f"| suspicious held aside {int(keep.is_suspicious.sum()):,}")
print(f"seq length: mean {length.mean():.1f}, p99 {np.percentile(length, 99):.0f}, "
      f"truncated {(length == MAX_LEN).mean() * 100:.2f}%")
print(f"wrote {OUT / 'sequences.npz'}")
