#!/usr/bin/env python3
"""Compare teacher runs on the same questions: accuracy, per-question coverage, and SFT yield.

Each --arm is name=cases.parquet:dir1,dir2 (sample dirs with *_trajectory.json and
eval_report_vdr_strict_details.jsonl). Only questions present in every arm are compared.
Yield = trajectories passing teacher_to_sft filters (correct, ended with response, think on tool turns,
no rule leak, gold in a tool output). --tags maps case_id -> category label (json) for a breakdown.
"""
import argparse, collections, glob, json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher_to_sft as T

ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append", required=True)
ap.add_argument("--tags", default=None)
a = ap.parse_args()
tags = json.load(open(a.tags)) if a.tags else {}
arms = {}
for spec in a.arm:
    name, rest = spec.split("=", 1)
    cases, dirs = rest.split(":", 1)
    c = pd.read_parquet(cases, columns=[x for x in ("data_id", "answer", "aliases") if x in pd.read_parquet(cases).columns])
    golds = {r["data_id"]: [r["answer"]] + ([str(x) for x in r["aliases"]] if "aliases" in c.columns and r["aliases"] is not None else [])
             for _, r in c.iterrows()}
    res = collections.defaultdict(list)
    for d in dirs.split(","):
        J = T.judged(d)
        for p in glob.glob(f"{d}/*_trajectory.json"):
            t = json.load(open(p)); q = t["case_id"]
            if q not in golds or q not in J:
                continue
            T.GOLDS = golds
            ok = J[q]
            kept = ok and T.convert(t, "x")[0] is not None
            calls = sum(1 for x in t["turns"] if x.get("tool_call"))
            res[q].append((ok, kept, calls, t.get("stop_reason")))
    arms[name] = res
common = set.intersection(*(set(r) for r in arms.values()))
print(f"common questions: {len(common)}")
groups = sorted({tags.get(q, "all") for q in common})
for name, res in arms.items():
    for g in groups + (["ALL"] if len(groups) > 1 else []):
        qs = [q for q in common if g in ("ALL", tags.get(q, "all"))]
        traj = [x for q in qs for x in res[q]]
        n = len(traj)
        print(f"{name:8s} {g:14s} q={len(qs):3d} traj={n:3d} acc={100*sum(x[0] for x in traj)/n:5.1f} "
              f"q_any_correct={sum(any(x[0] for x in res[q]) for q in qs):3d} kept_traj={sum(x[1] for x in traj):3d} "
              f"q_any_kept={sum(any(x[1] for x in res[q]) for q in qs):3d} calls={sum(x[2] for x in traj)/n:.2f} "
              f"broken={sum(x[3] in ('no_tool_call','max_tokens') for x in traj)}")
