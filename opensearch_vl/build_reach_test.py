#!/usr/bin/env python3
"""Reachability test (2026-11-19, DIAGNOSIS ONLY - these trajectories never become training data).

For eval questions that v2 + v3d never solved (16/16 wrong) plus easy controls, build two case files:
  reachA.parquet  the question as is                        -> teacher without the answer (can it be imitated?)
  reachB.parquet  the question + the reference answer and   -> teacher must FIND EVIDENCE for the answer with our
                  an evidence-finding instruction              tools without ever searching for the answer itself
                                                               (is the answer reachable in our tool environment?)
    python opensearch_vl/build_reach_test.py --ids reach_ids.json --eval dev1.parquet dev2.parquet --out-dir DIR
reach_ids.json: {"case_id": "zero16" | "control", ...}
"""
import argparse, json
from pathlib import Path

import pandas as pd

HINT = """

[PRIVILEGED REFERENCE ANSWER - for this run only]: {gold}
You are told the reference answer so that you know what to look for. Your task is to FIND EVIDENCE for it with
the tools, the way an agent who did not know the answer would have to: work out what the image shows (image_search
on the image or on crops, reading text in it), then search and follow intermediate entities until a tool output
contains the answer. Hard rules:
- Never put the reference answer, any part of it, or a translation / spelling variant of it into a search query or
  any other tool argument. Queries must be built only from the question, the image and earlier tool outputs.
- Do not stop at your own knowledge: the answer must appear in a tool output of this conversation.
- If a query fails, change it substantially (aliases, official names, another hop, a date / place qualifier).
- In the final <response>, give the answer and state which tool result (turn and passage) contains it. If after a
  thorough investigation no tool output contains it, say EVIDENCE_NOT_FOUND."""

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True)
ap.add_argument("--eval", nargs="+", required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
ids = json.load(open(a.ids))
df = pd.concat([pd.read_parquet(p) for p in a.eval]).drop_duplicates("data_id")
df = df[df.data_id.isin(ids)].reset_index(drop=True)
missing = sorted(set(ids) - set(df.data_id))
assert not missing, f"{len(missing)} ids not in eval parquets, e.g. {missing[:3]}"
df["category"] = [f"{ids[q]}:{c}" for q, c in zip(df.data_id, df.category)]
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / "reachA.parquet")
b = df.copy()
b["prompt"] = [[{**m[0], "content": m[0]["content"] + HINT.format(gold=g)}] + list(m[1:])
               for m, g in zip(b.prompt, b.answer)]
b.to_parquet(out / "reachB.parquet")
print(f"{len(df)} cases ({sum(v == 'zero16' for v in ids.values())} zero16, "
      f"{sum(v == 'control' for v in ids.values())} control) -> {out}")
