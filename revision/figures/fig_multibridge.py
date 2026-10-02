"""Figure: extended multi-bridge corpus. (a) transactions per bridge in the 12,354,560-transaction
corpus (counts from processed_ext/ext_model.log); (b) Across value ratio output / (input + fee), whose
values above 1 are exactly the conservation failures fee + input - output < 0 used in the paper."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import C, setup

setup()
EXT = Path(__file__).resolve().parents[2] / "processed_ext"
COUNTS = {"Across": 3864421, "Stargate (Bus)": 3526050, "Stargate (OFT)": 3291711, "CCTP": 592141,
          "Connext": 464542, "Orbit Chain": 458055, "Allbridge": 146210, "CCIP": 11430}
assert sum(COUNTS.values()) == 12354560

d = pd.read_parquet(EXT / "across_full.parquet",
                    columns=["input_amount_usd", "output_amount_usd", "src_fee_usd", "src_timestamp"])
num = lambda s: pd.to_numeric(s, errors="coerce")
i, o, fee = num(d.input_amount_usd), num(d.output_amount_usd), num(d.src_fee_usd).fillna(0)
ts = pd.to_datetime(num(d.src_timestamp), unit="s", errors="coerce")
ok = i.notna() & o.notna() & (i > 0) & ts.notna()
i, o, fee = i[ok], o[ok], fee[ok]
viol = int(((fee + i - o) < 0).sum())
ratio = (o / (i + fee)).clip(0.9, 1.1)
print("usable", int(ok.sum()), "violations", viol)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
names = list(COUNTS)[::-1]
axs[0].barh(names, [COUNTS[n] / 1e6 for n in names],
            color=[C["blue"] if n == "Across" else "#b7bec8" for n in names])
axs[0].set_xlabel("transactions (millions)")
axs[0].set_title("(a) 12.35 M transactions, eight bridges")
axs[1].hist(ratio, bins=200, color=C["blue"], log=True)
axs[1].axvline(1.0, color=C["red"], ls="--", lw=1.2)
axs[1].text(1.003, axs[1].get_ylim()[1] * 0.2, f"{viol:,} failures\n(fee + in − out < 0)", color=C["red"], fontsize=7.5)
axs[1].set_xlabel("output / (input + fee), Across")
axs[1].set_ylabel("transactions (log)")
axs[1].set_title(f"(b) value conservation, {int(ok.sum()):,} transactions")
fig.tight_layout(w_pad=1.0)
out = Path(__file__).resolve().parent
fig.savefig(out / "fig_multibridge.pdf"); fig.savefig(out / "fig_multibridge.png", dpi=200)
print("saved fig_multibridge")
