"""Reproduce the originating study's Table 1 descriptive statistics on our own
extraction, using their aggregation convention (value = row-wise sum over the
three per-bridge amount columns), and compare against their published figures.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
EXT = PROJECT / "processed_ext"

# published Table 1 values (Augusto et al., XChainDataGen)
PUBLISHED = {
    "cctp": (592_141, 8.57e9), "ccip": (11_430, 427.71e6),
    "stargate_oft": (3_291_711, 10.41e9), "stargate_bus": (3_526_050, 3.28e9),
    "across": (3_864_421, 5.51e9),
}
AMT_COLS = ["amount_usd", "amount_received_ld_usd", "output_amount_usd",
            "input_amount_usd"]


def fmt(v):
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"


rows = []
for b in ["cctp", "ccip", "stargate_oft", "stargate_bus", "across"]:
    f = EXT / f"xcdg_{b}.parquet"
    if not f.exists():
        continue
    d = pd.read_parquet(f)
    present = [c for c in AMT_COLS if c in d.columns]
    # their convention: row-wise sum across amount columns, then aggregate.
    # for Across the destination value is the transferred amount; for the
    # others only one column is populated, so the sum selects it.
    if b == "across":
        val = pd.to_numeric(d["output_amount_usd"], errors="coerce").fillna(0)
    elif b == "stargate_bus":
        val = pd.to_numeric(d["output_amount_usd"], errors="coerce").fillna(0)
    else:
        val = sum(pd.to_numeric(d[c], errors="coerce").fillna(0) for c in present)
    n_pub, v_pub = PUBLISHED[b]
    rows.append({
        "bridge": b,
        "txs_ours": len(d), "txs_published": n_pub,
        "txs_match": "yes" if len(d) == n_pub else f"diff {len(d)-n_pub:+,}",
        "value_ours": fmt(val.sum()), "value_published": fmt(v_pub),
        "value_ratio": round(val.sum() / v_pub, 3) if v_pub else None,
        "median_tx": fmt(val[val > 0].median()) if (val > 0).any() else "-",
        "max_tx": fmt(val.max()),
        "chains": d.src_blockchain.nunique() if "src_blockchain" in d else None,
        "depositors": d.depositor.nunique() if "depositor" in d else None,
        "recipients": d.recipient.nunique() if "recipient" in d else None,
    })

t = pd.DataFrame(rows)
print("=== Our extraction vs. published Table 1 (XChainDataGen) ===")
print(t[["bridge", "txs_ours", "txs_published", "txs_match"]].to_string(index=False))
print()
print(t[["bridge", "value_ours", "value_published", "value_ratio"]].to_string(index=False))
print()
print(t[["bridge", "chains", "depositors", "recipients", "median_tx",
         "max_tx"]].to_string(index=False))
t.to_csv(EXT / "corpus_stats.csv", index=False)
print(f"\ntotal transactions: {t.txs_ours.sum():,} "
      f"(published 11,285,753)")
print(f"wrote {EXT / 'corpus_stats.csv'}")
