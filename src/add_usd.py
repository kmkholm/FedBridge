"""
Stage 2d — USD-normalized transaction amounts (fixes cross-token units).

Static exploit-era prices (early/mid 2022) + token decimals for the dominant
contracts; unknown tokens default to decimals=18, price=$1. This is a
deliberate approximation: ranking needs magnitudes, not cents.
Output: processed/usd.parquet (tx_hash, log_usd).
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "processed"

# contract -> (decimals, usd price during 2022 exploit era)
TOKENS = {
    "0xc99a6a985ed2cac1ef41640596c5a5f9f4e19ef5": (18, 3000.0),  # WETH (Ronin)
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": (18, 3000.0),  # WETH mainnet
    "0x30d2a9f5fdf90ace8c17952cbb4ee48a55d916a7": (18, 1700.0),  # WETH (Nomad/moonbeam)
    "0xbb0e17ef65f82ab018d8edd776e8dd940327b28b": (18, 50.0),    # AXS mainnet
    "0x97a9107c1793bc407d6f527b77e7fff4d812bece": (18, 50.0),    # AXS (Ronin)
    "0xcc8fa225d80b9c7d42f96e9570156c65d6caaa25": (6, 1.0),      # USDC (Ronin)
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": (6, 1.0),      # USDC mainnet
    "0xdac17f958d2ee523a2206206994597c13d831ec7": (6, 1.0),      # USDT
    "0xa8754b9fa15fc18bb59458815510e40a12cd2014": (0, 0.02),     # SLP mainnet
    "0x0b7007c13325c48911f73a2dad5fa5dcbf808adc": (0, 0.02),     # SLP (Ronin)
    "0x8f552a71efe5eefc207bf75485b356a0b3f01ec9": (18, 0.5),     # WGLMR
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": (8, 30000.0),  # WBTC
}
DEFAULT = (18, 0.0)   # unknown tokens: no reliable price -> contribute $0

ev = pd.read_parquet(OUT / "events.parquet",
                     columns=["tx_hash", "contract", "dst_token", "amount_f"])
ev["asset"] = ev["contract"].fillna(ev["dst_token"]).str.lower()
dec = ev["asset"].map({k: v[0] for k, v in TOKENS.items()}).fillna(DEFAULT[0])
price = ev["asset"].map({k: v[1] for k, v in TOKENS.items()}).fillna(DEFAULT[1])
ev["usd"] = ev["amount_f"].fillna(0).clip(lower=0) / (10.0 ** dec) * price

usd = ev.groupby("tx_hash")["usd"].sum().rename("usd").reset_index()
usd["log_usd"] = np.log10(usd["usd"] + 1)
usd.to_parquet(OUT / "usd.parquet", index=False)

top = usd.nlargest(6, "usd")
print(top.to_string(index=False))
print(f"wrote {OUT / 'usd.parquet'} ({len(usd):,} rows)")
