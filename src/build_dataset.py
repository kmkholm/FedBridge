"""
FedBridgeMamba-KAN — Stage 1 preprocessing.

Parses the XChainWatcher Datalog facts (decoded bridge events) into
per-transaction event sequences and labels every transaction using the
Datalog rule verdicts:

  benign          — leg of a valid matched cross-chain transaction
                    (CCTX_Deposit / CCTX_Withdrawal / *MatchWithAdditionalData)
  attack_unmatched— withdrawal/deposit with NO matching leg on the other chain
                    (contains the Ronin & Nomad exploit transactions)
  anomaly         — flagged by *_Anomalies or *_FinalityBreak rules
  unlabeled       — appears in facts but in no result relation
                    (background traffic: reverted txs, random transfers, ...)

Outputs (to <project>/processed/):
  events.parquet        one row per decoded event
  transactions.parquet  one row per transaction: bridge, chain, ts, label, seq stats
  label_report.txt      counts per bridge/label for sanity checking
"""

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATALOG = PROJECT / "datasets" / "XChainWatcher" / "cross-chain-rules-validator" / "datalog"
OUT = PROJECT / "processed"
OUT.mkdir(exist_ok=True)

BRIDGES = ["ronin-bridge", "nomad-bridge"]

# ---------------------------------------------------------------- fact schemas
# from datalog/lib/declarations.dl — Souffle facts are TSV, no header.
FACT_SCHEMAS = {
    "sc_deposit": ["tx_hash", "event_index", "sender", "to", "amount"],
    "sc_withdrawal": ["tx_hash", "event_index", "from", "beneficiary", "amount"],
    "sc_token_deposited": ["tx_hash", "event_index", "deposit_id", "beneficiary",
                           "dst_token", "origin_token", "dst_chain_id", "standard", "amount"],
    "sc_token_withdrew": ["tx_hash", "event_index", "withdrawal_id", "beneficiary",
                          "dst_token", "amount"],
    "tc_deposit": ["tx_hash", "event_index", "from", "beneficiary", "amount"],
    "tc_withdrawal": ["tx_hash", "event_index", "sender", "to", "amount"],
    "tc_token_deposited": ["tx_hash", "event_index", "deposit_id", "beneficiary",
                           "dst_token", "amount"],
    "tc_token_withdrew": ["tx_hash", "event_index", "withdrawal_id", "beneficiary",
                          "origin_token", "dst_token", "dst_chain_id", "standard", "amount"],
    "erc20_transfer": ["tx_hash", "chain_id", "event_index", "contract", "from", "to", "amount"],
}
# additional_* files share the schema of their base relation
for base in list(FACT_SCHEMAS):
    FACT_SCHEMAS[f"additional_{base}"] = FACT_SCHEMAS[base]

TX_SCHEMA = ["timestamp", "chain_id", "tx_hash", "tx_index", "from", "to", "value", "status", "fee"]

# ------------------------------------------------------------- result schemas
# from acceptance-rules.dl / additional-rules.dl / misc-rules.dl.
# map: result file -> (label, [columns holding tx hashes to label])
RESULTS = {
    # valid matched cctx: both legs benign
    "CCTX_Deposit": ("benign", [2, 5]),
    "CCTX_Withdrawal": ("benign", [2, 5]),
    "TC_WithdrawalsMatchWithAdditionalData": ("benign", [2, 5]),
    "SC_WithdrawalsMatchWithAdditionalData": ("benign", [2, 5]),
    # unmatched legs — attack candidates (Ronin/Nomad exploits live here)
    "TC_WithdrawalsWithoutMatch": ("attack_unmatched", [2]),
    "SC_WithdrawalsWithoutMatch": ("attack_unmatched", [2]),
    "TC_WithdrawalsWithoutMatchWithAdditionalData": ("attack_unmatched", [2]),
    "SC_WithdrawalsWithoutMatchWithAdditionalData": ("attack_unmatched", [2]),
    "SC_DepositsWithoutMatch": ("attack_unmatched", [2]),
    "TC_DepositsWithoutMatch": ("attack_unmatched", [2]),
    # rule anomalies + finality breaks
    "SC_ValidNativeTokenDeposit_Anomalies": ("anomaly", [0]),
    "SC_ValidERC20TokenDeposit_Anomalies": ("anomaly", [0]),
    "TC_ValidERC20TokenDeposit_Anomalies": ("anomaly", [0]),
    "TC_ValidNativeTokenWithdrawal_Anomalies": ("anomaly", [0]),
    "TC_ValidERC20TokenWithdrawal_Anomalies": ("anomaly", [0]),
    "SC_ValidERC20TokenWithdrawal_Anomalies": ("anomaly", [0]),
    "CCTX_Deposit_FinalityBreak": ("anomaly", [2, 5]),
    "CCTX_Withdrawal_FinalityBreak": ("anomaly", [2, 5]),
}
# label priority when a tx appears in several relations (higher wins)
PRIORITY = {"unlabeled": 0, "benign": 1, "anomaly": 2, "attack_unmatched": 3}


def read_facts(path: Path, cols: list[str]) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path, sep="\t", header=None, names=cols, dtype=str,
                     on_bad_lines="warn", quoting=3)
    return df


def load_bridge(bridge: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    facts_dir = DATALOG / bridge / "facts"
    results_dir = DATALOG / bridge / "results"

    # ---- events
    frames = []
    for rel, cols in FACT_SCHEMAS.items():
        df = read_facts(facts_dir / f"{rel}.facts", cols)
        if df is None:
            continue
        df["event_type"] = rel.removeprefix("additional_")
        df["from_additional_data"] = rel.startswith("additional_")
        frames.append(df)
    events = pd.concat(frames, ignore_index=True)
    events["bridge"] = bridge

    # ---- transactions (base + additional)
    tx_frames = []
    for name in ["transaction", "additional_transaction"]:
        df = read_facts(facts_dir / f"{name}.facts", TX_SCHEMA)
        if df is not None:
            tx_frames.append(df)
    txs = pd.concat(tx_frames, ignore_index=True).drop_duplicates(subset=["tx_hash"])
    txs["bridge"] = bridge

    # ---- labels from result relations
    label = {}
    for rel, (lab, hash_cols) in RESULTS.items():
        p = results_dir / f"{rel}.csv"
        if not p.exists() or p.stat().st_size == 0:
            continue
        df = pd.read_csv(p, sep="\t", header=None, dtype=str, quoting=3)
        for c in hash_cols:
            if c >= df.shape[1]:
                continue
            for h in df[c].dropna():
                if PRIORITY[lab] > PRIORITY.get(label.get(h, "unlabeled"), 0):
                    label[h] = lab
    txs["label"] = txs["tx_hash"].map(label).fillna("unlabeled")
    return events, txs


def main() -> None:
    all_events, all_txs = [], []
    for bridge in BRIDGES:
        ev, tx = load_bridge(bridge)
        all_events.append(ev)
        all_txs.append(tx)
        print(f"{bridge}: {len(ev):,} events, {len(tx):,} transactions")

    events = pd.concat(all_events, ignore_index=True)
    txs = pd.concat(all_txs, ignore_index=True)

    # numeric casts (amounts are uint256 — keep string + float64 view)
    events["event_index"] = pd.to_numeric(events["event_index"], errors="coerce")
    events["amount_f"] = pd.to_numeric(events["amount"], errors="coerce")
    txs["timestamp"] = pd.to_numeric(txs["timestamp"], errors="coerce")
    txs["value_f"] = pd.to_numeric(txs["value"], errors="coerce")
    txs["status"] = pd.to_numeric(txs["status"], errors="coerce")

    # sequence stats per tx
    seq = (events.sort_values(["tx_hash", "event_index"])
                 .groupby("tx_hash")
                 .agg(n_events=("event_type", "size"),
                      n_event_types=("event_type", "nunique")))
    txs = txs.merge(seq, on="tx_hash", how="left")
    txs["n_events"] = txs["n_events"].fillna(0).astype(int)

    events.to_parquet(OUT / "events.parquet", index=False)
    txs.to_parquet(OUT / "transactions.parquet", index=False)

    report = txs.groupby(["bridge", "label"]).size().unstack(fill_value=0)
    with open(OUT / "label_report.txt", "w") as f:
        f.write(report.to_string() + "\n")
    print("\n=== label report (transactions) ===")
    print(report.to_string())
    print(f"\nwrote {OUT / 'events.parquet'} ({len(events):,} rows)")
    print(f"wrote {OUT / 'transactions.parquet'} ({len(txs):,} rows)")


if __name__ == "__main__":
    main()
