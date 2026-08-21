"""
Stage 1b — refine Datalog rule verdicts into training labels.

Rule verdicts are honest but noisy: "unmatched" includes benign withdrawals
whose deposit leg predates the collection window (dominant on Ronin, whose
crawl starts 2022-01-01 while the bridge is older). Refined taxonomy:

  attack      confirmed exploit txs:
                ronin — the two documented exploit txs (2022-03-23)
                nomad — unmatched/anomalous txs during the exploit wave
                        (>= 2022-08-01 00:00 UTC, the initial exploit +
                         copycat wave documented in incident post-mortems)
  suspicious  unmatched/anomalous OUTSIDE the exploit windows
              (mostly window-boundary artifacts; EXCLUDED from training,
               reported in the paper's label-noise discussion)
  benign      legs of valid matched cross-chain transactions
  unlabeled   background traffic in no result relation (excluded)

Output: processed/transactions_refined.parquet (adds `label_refined`)
"""

from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"

RONIN_EXPLOIT_TXS = {
    "0xc28fad5e8d5e0ce6a2eaf67b6687be5d58113e16be590824d6cfa1a94467d0b7",  # 173,600 ETH
    "0xed2c72ef1a552ddaec6dd1f5cddf0b59a8f37f82bdda5257d9c7c37db7bb9b08",  # 25.5M USDC
}
# onset validated against the data itself: unmatched-withdrawal rate jumps from
# ~2-8/hour background to 23/161/34 per hour at 21:00-23:59 UTC on Aug 1 2022,
# matching the documented first exploit tx (~21:32 UTC); copycat wave tapers by Aug 6
NOMAD_EXPLOIT_START = pd.Timestamp("2022-08-01 21:00", tz="UTC").timestamp()
NOMAD_EXPLOIT_END = pd.Timestamp("2022-08-06", tz="UTC").timestamp()

txs = pd.read_parquet(OUT / "transactions.parquet")

flagged = txs.label.isin(["attack_unmatched", "anomaly"])
is_ronin_attack = txs.tx_hash.isin(RONIN_EXPLOIT_TXS)
is_nomad_attack = (
    (txs.bridge == "nomad-bridge") & flagged
    & (txs.timestamp >= NOMAD_EXPLOIT_START) & (txs.timestamp < NOMAD_EXPLOIT_END)
)

txs["label_refined"] = "unlabeled"
txs.loc[txs.label == "benign", "label_refined"] = "benign"
txs.loc[flagged, "label_refined"] = "suspicious"
txs.loc[is_ronin_attack | is_nomad_attack, "label_refined"] = "attack"

txs.to_parquet(OUT / "transactions_refined.parquet", index=False)

report = txs.groupby(["bridge", "label_refined"]).size().unstack(fill_value=0)
print(report.to_string())
attacks = txs[txs.label_refined == "attack"]
print(f"\nattack txs: {len(attacks)} "
      f"(ronin {sum(attacks.bridge == 'ronin-bridge')}, "
      f"nomad {sum(attacks.bridge == 'nomad-bridge')})")
print("nomad attack time range:",
      pd.to_datetime(attacks[attacks.bridge == 'nomad-bridge'].timestamp.min(), unit='s'),
      "->",
      pd.to_datetime(attacks[attacks.bridge == 'nomad-bridge'].timestamp.max(), unit='s'))
