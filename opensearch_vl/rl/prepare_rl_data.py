#!/usr/bin/env python3
"""Build RL train / dev parquets from audited question lists + candidate parquets.

Each list is JSON: [{"data_id", "gold", "aliases", ...}] (train_clean_*.json, dev_clean.json).
The audited gold replaces the extracted answer; aliases go into an "aliases" column that
rl/train_grpo.py passes to the judge and to the retrieval-hit check.

    python rl/prepare_rl_data.py --list train_clean_b12.json --cands c1.parquet c2.parquet --out rl_train.parquet
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", nargs="+", required=True)
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = {}
    for p in args.list:
        for it in json.load(open(p, encoding="utf-8")):
            items[it["data_id"]] = it
    df = pd.concat([pd.read_parquet(p) for p in args.cands]).drop_duplicates("data_id")
    df = df[df.data_id.isin(items)].copy()
    missing = set(items) - set(df.data_id)
    if missing:
        raise SystemExit(f"{len(missing)} listed ids not in candidates, e.g. {sorted(missing)[:5]}")
    df["answer"] = [items[d]["gold"] for d in df.data_id]
    df["aliases"] = [[str(a) for a in (items[d].get("aliases") or []) if str(a).strip()] for d in df.data_id]
    df = df[["data_id", "id", "category", "data_source", "prompt", "images", "answer", "aliases"]].reset_index(drop=True)
    df.to_parquet(args.out)
    print(args.out, len(df), df.category.value_counts().to_dict())


if __name__ == "__main__":
    main()
