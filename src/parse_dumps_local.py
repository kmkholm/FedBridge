"""Stream-parse the XChainDataGen Postgres dumps on Windows (no DB, no full
extraction, bounded memory). Reads the *_cross_chain_transactions COPY block
from each dump inside the zip and writes one parquet per bridge."""

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

ZIP = Path(r"D:\Drive D\Ajloun Papers\FedBridge 2026\jun2024-dec2024.zip")
OUT = Path(r"D:\Drive D\Ajloun Papers\FedBridge 2026\processed_ext")
OUT.mkdir(exist_ok=True)
CHUNK = 250_000
KEEP = ["src_blockchain", "dst_blockchain", "src_timestamp", "dst_timestamp",
        "depositor", "recipient", "src_transaction_hash", "amount", "amount_usd",
        "input_amount", "output_amount", "input_amount_usd", "output_amount_usd"]


def parse(zf: zipfile.ZipFile, member: str, name: str) -> int:
    cols, buf, part, total = None, [], 0, 0
    parts = []

    def flush():
        nonlocal buf, part, total
        if not buf:
            return
        df = pd.DataFrame(buf, columns=cols)
        keep = [c for c in KEEP if c in df.columns]
        df = df[keep]
        p = OUT / f"tmp_{name}_{part:04d}.parquet"
        df.to_parquet(p, index=False)
        parts.append(p)
        total += len(buf)
        part += 1
        buf = []
        print(f"  {name}: {total:,} rows", flush=True)

    with zf.open(member) as raw:
        stream = io.TextIOWrapper(io.BufferedReader(raw, 1 << 22),
                                  encoding="utf-8", errors="replace")
        for line in stream:
            if cols is None:
                if line.startswith("COPY public.") and "_cross_chain_transactions (" in line:
                    cols = [c.strip().strip('"') for c in
                            line.split("(", 1)[1].rsplit(")", 1)[0].split(",")]
                continue
            if line.startswith("\\."):
                break
            buf.append(line.rstrip("\n").split("\t"))
            if len(buf) >= CHUNK:
                flush()
    flush()

    if not parts:
        print(f"{name}: no cross-chain table found", flush=True)
        return 0
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["bridge_name"] = name
    df.to_parquet(OUT / f"xcdg_{name}.parquet", index=False)
    for p in parts:
        os.remove(p)
    print(f"{name}: DONE {total:,} cross-chain transactions | {list(df.columns)}",
          flush=True)
    return total


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["ccip", "cctp", "across", "stargate"]
    with zipfile.ZipFile(ZIP) as zf:
        members = {m.split("/")[-1].replace(".dump", ""): m
                   for m in zf.namelist() if m.endswith(".dump")}
        for name in wanted:
            if name not in members:
                print(f"{name}: not in archive", flush=True)
                continue
            if (OUT / f"xcdg_{name}.parquet").exists():
                print(f"{name}: already parsed, skipping", flush=True)
                continue
            parse(zf, members[name], name)
    print("LOCAL_PARSE_DONE", flush=True)
