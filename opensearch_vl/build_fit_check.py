#!/usr/bin/env python3
"""Fit check (2026-11-19): did v3d learn its own training questions?

Turns SFT training samples back into eval cases (question + question image(s); gold = the teacher's final
\\boxed{} answer, which passed the vision audits) in the paper7 parquet format:
  fitnew__*  --n-new samples of the v3d new data (seen x2 by v3d, never by v2)       -> v3d vs v2 = learning
  fitv2__*   --n-v2 samples of the v2 SFT subset (seen once by v2 and by v3e)      -> how well SFT fits at all

    python opensearch_vl/build_fit_check.py --new-ids /root/rl_data/v3d_final_ids.txt \
        --out /root/autodl-fs/datasets/imagentrl/eval/fit_check.parquet
"""
import argparse, importlib.util, json, random
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "select_sft_subset", Path(__file__).resolve().parents[1] / "SFT/scripts/select_sft_subset.py")
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

ap = argparse.ArgumentParser()
ap.add_argument("--new-ids", required=True)
ap.add_argument("--cand-dir", default="/root/sft-audit/v3d/cand")
ap.add_argument("--v2-dir", default="/root/datasets/sft_subset_q3vl_v2")
ap.add_argument("--n-new", type=int, default=120)
ap.add_argument("--n-v2", type=int, default=60)
ap.add_argument("--seed", type=int, default=20261119)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = random.Random(a.seed)


def load(d):
    data = json.load(open(f"{d}/sft_subset.json", encoding="utf-8"))
    man = [json.loads(l) for l in open(f"{d}/manifest.jsonl", encoding="utf-8")]
    assert len(data) == len(man)
    return {m["id"]: (s, m) for s, m in zip(data, man)}


def case(prefix, sid, s, m):
    conv = s["conversations"]
    first = conv[0]["value"]
    k = first.count("<image>")
    gpts = [c["value"] for c in conv if c["from"] == "gpt"]
    box = [b for b in sel.boxed_spans(gpts[-1]) if b and b.strip()] if gpts else []
    if not k or not box:
        return None
    cid = f"{prefix}__{sid.replace(':', '_')}"
    return {"data_id": cid, "id": cid, "category": prefix, "data_source": m["source"],
            "prompt": [{"role": "user", "content": first.replace("<image>", "").strip()}],
            "images": [{"bytes": open(p, "rb").read()} for p in s["images"][:k]],
            "answer": box[-1].strip(), "n_calls": m["n_calls"]}


rows, rep = [], {}
for prefix, pool, n in (("fitnew", None, a.n_new), ("fitv2", None, a.n_v2)):
    if prefix == "fitnew":
        by = load(a.cand_dir)
        ids = [l.strip() for l in open(a.new_ids) if l.strip()]
        assert all(i in by for i in ids)
    else:
        by = load(a.v2_dir)
        ids = sorted(by)
    rng.shuffle(ids)
    got = []
    for sid in ids:
        if len(got) == n:
            break
        r = case(prefix, sid, *by[sid])
        if r:
            got.append(r)
    assert len(got) == n, (prefix, len(got))
    rows += got
    rep[prefix] = {"n": n, "mean_calls": sum(r["n_calls"] for r in got) / n,
                   "sources": {s: sum(r["data_source"] == s for r in got) for s in sorted({r["data_source"] for r in got})}}
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(out)
rep["ids"] = [r["id"] for r in rows]
json.dump(rep, open(out.with_suffix(".report.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "ids"}), f"-> {out}")
