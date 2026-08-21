"""Re-extract Across with the fee columns, so the originating study's own cost
definition can be applied verbatim:

    user_cost = src_fee_usd + (input_amount_usd - output_amount_usd)

A negative user cost means the user received more value than they committed,
which is the criterion behind the 1,285 'profitable' transactions reported by
the dataset authors. Our earlier label ignored fees and is therefore looser.
"""

import io
import os
import zipfile
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ZIP = PROJECT / "jun2024-dec2024.zip"
OUT = PROJECT / "processed_ext"
CHUNK = 300_000
WANT = ["src_blockchain", "dst_blockchain", "src_timestamp", "dst_timestamp",
        "depositor", "recipient", "src_transaction_hash", "src_fee", "src_fee_usd",
        "dst_fee", "dst_fee_usd", "input_amount", "output_amount",
        "input_amount_usd", "output_amount_usd", "input_token", "output_token",
        "quote_timestamp", "fill_deadline", "exclusivity_deadline",
        "exclusive_relayer", "relayer", "src_contract_address"]


def main():
    with zipfile.ZipFile(ZIP) as zf:
        member = next(m for m in zf.namelist() if m.endswith("across.dump"))
        with zf.open(member) as raw:
            st = io.TextIOWrapper(io.BufferedReader(raw, 1 << 22),
                                  encoding="utf-8", errors="replace")
            cols, buf, part, total, parts = None, [], 0, 0, []
            for line in st:
                if cols is None:
                    if line.startswith("COPY public.") and "_cross_chain_transactions (" in line:
                        cols = [c.strip().strip('"') for c in
                                line.split("(", 1)[1].rsplit(")", 1)[0].split(",")]
                        print("columns:", cols, flush=True)
                    continue
                if line.startswith("\\."):
                    break
                buf.append(line.rstrip("\n").split("\t"))
                if len(buf) >= CHUNK:
                    df = pd.DataFrame(buf, columns=cols)
                    df = df[[c for c in WANT if c in df.columns]]
                    p = OUT / f"tmp_af_{part:04d}.parquet"
                    df.to_parquet(p, index=False)
                    parts.append(p)
                    total += len(buf)
                    part += 1
                    buf = []
                    print(f"  {total:,} rows", flush=True)
            if buf:
                df = pd.DataFrame(buf, columns=cols)
                df = df[[c for c in WANT if c in df.columns]]
                p = OUT / f"tmp_af_{part:04d}.parquet"
                df.to_parquet(p, index=False)
                parts.append(p)
                total += len(buf)
    d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for p in parts:
        os.remove(p)
    d.to_parquet(OUT / "across_full.parquet", index=False)
    print(f"across_full: {len(d):,} rows | {list(d.columns)}", flush=True)

    # apply the originating study's cost definition
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    inp, out = num("input_amount_usd"), num("output_amount_usd")
    fee = num("src_fee_usd").fillna(0)
    user_cost = fee + (inp - out)
    ok = inp.notna() & out.notna() & (inp > 0)
    print(f"\nusable rows: {int(ok.sum()):,}", flush=True)
    print(f"user_cost < 0 (their 'profitable' criterion): "
          f"{int((user_cost[ok] < 0).sum()):,}", flush=True)
    print(f"our earlier criterion (out > in*1.001, fees ignored): "
          f"{int((out[ok] > inp[ok] * 1.001).sum()):,}", flush=True)
    print(f"strictly out > in (no margin, no fees): "
          f"{int((out[ok] > inp[ok]).sum()):,}", flush=True)
    d.loc[ok, "user_cost"] = user_cost[ok]
    d.loc[ok, "profitable"] = (user_cost[ok] < 0).astype(int)
    prof = d[ok & (user_cost < 0)]
    print("\nprofitable by source chain:", flush=True)
    print(prof.src_blockchain.value_counts().head(6).to_string(), flush=True)
    d.to_parquet(OUT / "across_full.parquet", index=False)
    print(f"\nwrote {OUT / 'across_full.parquet'}", flush=True)


if __name__ == "__main__":
    main()
