#!/usr/bin/env python3
"""Drop candidate questions that the blind gold audit (check_rl_gold.py verdicts) flags.

A row is dropped when its verdict is missing or failed, question_ok == "no", or gold_correct == "no".

    python apply_gold_audit.py --cands in.parquet --verdicts /root/eval_audit/x --out out.parquet
"""
import argparse, collections, glob, json

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cands", required=True)
ap.add_argument("--verdicts", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
V = {}
for f in glob.glob(f"{a.verdicts}/verdicts_*.jsonl"):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        V[r["data_id"]] = r
d = pd.read_parquet(a.cands)
bad = lambda r: (not r) or "judge_failed" in r or r.get("question_ok") == "no" or r.get("gold_correct") == "no"
mask = [not bad(V.get(q)) for q in d.data_id]
keep = d[mask].reset_index(drop=True)
print(f"audited {len(V)}, kept {len(keep)} of {len(d)}")
if "v2k" in d.columns:
    print("kept/total by v2k:", {k: f"{int(sum(m for m, x in zip(mask, d.v2k) if x == k))}/{int((d.v2k == k).sum())}"
                                 for k in sorted(d.v2k.unique())})
keep.to_parquet(a.out)
