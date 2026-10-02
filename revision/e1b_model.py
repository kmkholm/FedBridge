"""E1b -- BridgeMamba-KAN over the four frozen protocols, repeated over seeds (GPU).

Reuses run() from src/train_model_multiseed.py, a CUDA port of
src/train_model.py with identical hyper-parameters (8 epochs, batch 512, AdamW 2e-3,
pos_weight cap 300, 10% stratified validation, patience 2, 50k benign cap). Nothing is
written into the FedBridge tree: rows and test scores go to results/ here.

usage: python e1b_model.py --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1] / "src"
RES = Path(__file__).resolve().parent / "results"
SCORES = RES / "e1b_scores"
SCORES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location("tmm", CODE / "train_model_multiseed.py")
tmm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tmm)


def main(seeds, protocols):
    out = RES / "e1b_model.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["protocol"], r["seed"]) for r in rows}
    print(f"device={tmm.DEV}", flush=True)
    for protocol in protocols:
        for seed in seeds:
            if (protocol, seed) in done:
                continue
            r = tmm.run(protocol, seed, save_scores=SCORES / f"{protocol}_s{seed}.npz")
            if r is None:
                continue
            rows.append(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{protocol:11s} seed {seed}  PR-AUC {r['pr_auc']:.4f}  ROC {r['roc_auc']:.4f}  "
                  f"R@0.1%FPR {r['recall_at_fpr']:.3f}  ({r['train_s']}s)", flush=True)
        v = [r["pr_auc"] for r in rows if r["protocol"] == protocol]
        if v:
            print(f"== {protocol}: PR-AUC {np.mean(v):.4f} +/- {np.std(v, ddof=1) if len(v) > 1 else 0:.4f} (n={len(v)})", flush=True)
    print(f"wrote results/{out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42] + list(range(9)))
    ap.add_argument("--protocols", nargs="+", default=["random", "temporal", "lobo_ronin", "lobo_nomad"])
    a = ap.parse_args()
    main(a.seeds, a.protocols)
