"""Re-parse Stargate's two cross-chain tables (Bus and OFT/Taxi modes) with
their actual column names, normalising to the common schema used by the other
bridges. Single pass over the 25 GB dump, bounded memory."""

import io
import os
import zipfile
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ZIP = PROJECT / "jun2024-dec2024.zip"
OUT = PROJECT / "processed_ext"
CHUNK = 250_000

TARGETS = {
    "stargate_bus_cross_chain_transactions": "stargate_bus",
    "stargate_oft_cross_chain_transactions": "stargate_oft",
}
# common name -> candidate source columns, first match wins
MAP = {
    "src_blockchain": ["src_blockchain"],
    "dst_blockchain": ["dst_blockchain"],
    "src_timestamp": ["user_timestamp", "src_timestamp", "bus_timestamp"],
    "dst_timestamp": ["dst_timestamp"],
    "depositor": ["user_from_address", "passenger", "depositor", "src_from_address"],
    "recipient": ["dst_to_address", "user_to_address", "recipient"],
    "src_transaction_hash": ["user_transaction_hash", "src_transaction_hash"],
    "input_amount_usd": ["amount_sent_ld_usd", "input_amount_usd", "amount_usd"],
    "output_amount_usd": ["amount_received_ld_usd", "output_amount_usd"],
}


def flush(buf, cols, name, part, parts):
    if not buf:
        return 0
    df = pd.DataFrame(buf, columns=cols)
    out = {}
    for tgt, cands in MAP.items():
        src = next((c for c in cands if c in df.columns), None)
        if src:
            out[tgt] = df[src]
    d = pd.DataFrame(out)
    p = OUT / f"tmp_{name}_{part:04d}.parquet"
    d.to_parquet(p, index=False)
    parts.append(p)
    return len(buf)


def main():
    with zipfile.ZipFile(ZIP) as zf:
        member = next(m for m in zf.namelist() if m.endswith("stargate.dump"))
        with zf.open(member) as raw:
            stream = io.TextIOWrapper(io.BufferedReader(raw, 1 << 22),
                                      encoding="utf-8", errors="replace")
            cur = None
            cols = None
            buf, part, total = [], 0, 0
            parts = []
            done = {}
            for line in stream:
                if cur is None:
                    if line.startswith("COPY public."):
                        tbl = line.split("COPY public.", 1)[1].split(" ", 1)[0]
                        if tbl in TARGETS and TARGETS[tbl] not in done:
                            cur = TARGETS[tbl]
                            cols = [c.strip().strip('"') for c in
                                    line.split("(", 1)[1].rsplit(")", 1)[0].split(",")]
                            buf, part, total, parts = [], 0, 0, []
                            print(f"reading {tbl} -> {cur}", flush=True)
                    continue
                if line.startswith("\\."):
                    total += flush(buf, cols, cur, part, parts)
                    df = pd.concat([pd.read_parquet(p) for p in parts],
                                   ignore_index=True)
                    df["bridge_name"] = cur
                    df.to_parquet(OUT / f"xcdg_{cur}.parquet", index=False)
                    for p in parts:
                        os.remove(p)
                    print(f"{cur}: DONE {total:,} cross-chain transactions | "
                          f"{list(df.columns)}", flush=True)
                    done[cur] = total
                    cur, cols = None, None
                    if len(done) == len(TARGETS):
                        break
                    continue
                buf.append(line.rstrip("\n").split("\t"))
                if len(buf) >= CHUNK:
                    total += flush(buf, cols, cur, part, parts)
                    part += 1
                    buf = []
                    print(f"  {cur}: {total:,} rows", flush=True)
    print("STARGATE_DONE", flush=True)


if __name__ == "__main__":
    main()
