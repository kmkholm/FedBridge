"""
E1c — alert-triage reformulation: rank true exploits among rule-engine alerts.

Universe = flagged txs only (label_refined in {attack, suspicious}).
Protocols:
  lobo_ronin  train on nomad alerts (463 atk / 607 benign-alert),
              test on ronin alerts (2 atk / 15,324 benign-alert)
  lobo_nomad  reverse direction
  unsup_amount  no training at all: rank alerts by traced amount (sanity ref)
Features: final-step history feats + context (same as baselines).
Model: HistGradientBoosting. Metric: PR-AUC + rank of each true exploit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"

d = np.load(OUT / "histories.npz", allow_pickle=False)
X, y, susp = d["X"], d["y"], d["suspicious"].astype(bool)
tx_hash, bridge_id = d["tx_hash"], d["bridge_id"]
ctx = pd.read_parquet(OUT / "context.parquet").set_index("tx_hash")
CTX = ctx.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)
usd = pd.read_parquet(OUT / "usd.parquet").set_index("tx_hash")["log_usd"]
USD = usd.reindex(tx_hash).fillna(0).to_numpy(dtype=np.float32)[:, None]
F = np.hstack([X[:, -1, :], CTX, USD])

flagged = susp | (y == 1)          # the rule-engine alert stream
print(f"alert universe: {flagged.sum():,} "
      f"(true exploits {int(y[flagged].sum())})")

for name, train_b, test_b in [("lobo_ronin (train nomad)", 1, 0),
                              ("lobo_nomad (train ronin)", 0, 1)]:
    tr = flagged & (bridge_id == train_b)
    te = flagged & (bridge_id == test_b)
    if y[tr].sum() == 0:
        print(f"{name}: no train positives ({int(y[tr].sum())}) — skipped")
        continue
    clf = HistGradientBoostingClassifier(class_weight="balanced",
                                         random_state=42).fit(F[tr], y[tr])
    s = clf.predict_proba(F[te])[:, 1]
    ranks = (np.argsort(np.argsort(-s))[y[te] == 1] + 1)
    print(f"{name}: PR-AUC {average_precision_score(y[te], s):.4f} "
          f"ROC {roc_auc_score(y[te], s):.4f} | "
          f"exploit ranks {sorted(ranks.tolist())[:8]} of {int(te.sum()):,}")

# unsupervised amount ranking reference (USD-normalized)
amt = USD[:, 0]
for name, b in [("ronin", 0), ("nomad", 1)]:
    te = flagged & (bridge_id == b)
    s = amt[te]
    ranks = (np.argsort(np.argsort(-s))[y[te] == 1] + 1)
    print(f"amount-only [{name}]: PR-AUC "
          f"{average_precision_score(y[te], s):.4f} "
          f"ROC {roc_auc_score(y[te], s):.4f} | "
          f"exploit ranks {sorted(ranks.tolist())[:8]} of {int(te.sum()):,}")
