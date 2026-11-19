#!/usr/bin/env python3
"""Reachability test (2026-11-19, DIAGNOSIS ONLY - these trajectories never become training data).

For eval questions that v2 + v3d never solved (16/16 wrong) plus easy controls, build two case files:
  reachA.parquet  the question as is                        -> teacher without the answer (can it be imitated?)
  reachB.parquet  the question + the reference answer and   -> teacher must FIND EVIDENCE for the answer with our
                  an evidence-finding instruction              tools; it may search the answer only after a tool
                                                               output has surfaced it
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
- Do not search for the reference answer (or any part, translation or spelling variant of it) BEFORE a tool output
  of this conversation has surfaced it. Until then, build queries only from the question, the image and earlier
  tool outputs. Once a tool output has shown the answer (e.g. image_search names the entity), use it freely: search
  it to verify it or to continue to the next hop.
- Do not stop at your own knowledge: the answer must appear in a tool output of this conversation.
- If a query fails, change it substantially (aliases, official names, another hop, a date / place qualifier).
- In the final <response>, give the answer and state which tool result (turn and passage) contains it. If after a
  thorough investigation no tool output contains it, say EVIDENCE_NOT_FOUND."""

HINT_C = """

[HINT - for this run only]: the key entity shown in the image is: {entity}. The answer is NOT given; find it with
the tools. The final answer must be supported by a tool output of this conversation."""

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True)
ap.add_argument("--eval", nargs="+", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--hints", help="group C: {case_id: {entity, reveals_answer}} -> reachC.parquet")
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
if a.hints:  # group C: only the key entity of the image is given (no answer); questions whose entity is the answer drop out
    h = json.load(open(a.hints))
    keep = [bool(h.get(q, {}).get("entity")) and not h[q]["reveals_answer"] for q in df.data_id]
    c = df[keep].copy()
    c["prompt"] = [[{**m[0], "content": m[0]["content"] + HINT_C.format(entity=h[q]["entity"])}] + list(m[1:])
                   for m, q in zip(c.prompt, c.data_id)]
    c.to_parquet(out / "reachC.parquet")
    print(f"group C: {len(c)} of {len(df)} cases have a usable entity hint")
print(f"{len(df)} cases ({sum(v == 'zero16' for v in ids.values())} zero16, "
      f"{sum(v == 'control' for v in ids.values())} control) -> {out}")
