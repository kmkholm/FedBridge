"""
Stage 3a — frozen evaluation splits (every experiment uses these masks).

Protocols (masks stored in processed/splits.npz, aligned with histories.npz):

  random   stratified 80/20 (seed 42) — the optimistic protocol everyone
           else reports; we report it AND show why it overstates.
  temporal train on everything before 2022-08-01 21:00 UTC (Nomad exploit
           onset), test on everything after — the exploit is a true zero-day
           for the model. Ronin's 2 attack txs (Mar 2022) fall in TRAIN,
           giving supervision; the entire Nomad wave is TEST.
  lobo_ronin / lobo_nomad — leave-one-bridge-out: train on the other bridge,
           test on the held-out one (zero-day bridge generalization).

Suspicious-labeled samples are excluded from all train sets AND all test
metrics (kept for the label-noise discussion only).
"""

from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"

d = np.load(OUT / "histories.npz", allow_pickle=False)
y, susp = d["y"], d["suspicious"].astype(bool)
bridge_id, ts = d["bridge_id"], d["timestamp"].astype(np.int64)
n = len(y)
usable = ~susp

rng = np.random.default_rng(42)
test_random = np.zeros(n, dtype=bool)
for cls in (0, 1):
    idx = np.flatnonzero(usable & (y == cls))
    test_random[rng.choice(idx, size=int(0.2 * len(idx)), replace=False)] = True

NOMAD_ONSET = 1659387600  # 2022-08-01 21:00 UTC
train_temporal = usable & (ts < NOMAD_ONSET)
test_temporal = usable & (ts >= NOMAD_ONSET)

splits = {
    "random_train": usable & ~test_random,
    "random_test": usable & test_random,
    "temporal_train": train_temporal,
    "temporal_test": test_temporal,
    "lobo_ronin_train": usable & (bridge_id == 1),   # train nomad, test ronin
    "lobo_ronin_test": usable & (bridge_id == 0),
    "lobo_nomad_train": usable & (bridge_id == 0),   # train ronin, test nomad
    "lobo_nomad_test": usable & (bridge_id == 1),
}
np.savez_compressed(OUT / "splits.npz", **splits)

for k, m in splits.items():
    print(f"{k:18s} n={m.sum():7,}  attack={int(y[m].sum()):5,}")
