#!/usr/bin/env python3
"""Mix new teacher SFT samples with a random replay subset of the v2 SFT data (LLaMA-Factory dataset dir).

    python build_sft_mix.py --new /root/datasets/sft_v3a_new/sft_new.json \
        --replay /root/datasets/sft_subset_q3vl_v2/sft_subset.json --n-replay 1000 \
        --out /root/datasets/sft_v3a_mix --name imagentrl_sft_v3a
"""
import argparse, json, os, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--new", required=True, nargs="+")
ap.add_argument("--replay", required=True)
ap.add_argument("--n-replay", type=int, default=1000)
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--seed", type=int, default=20261116)
a = ap.parse_args()
new = [x for f in a.new for x in json.load(open(f, encoding="utf-8"))]
old = json.load(open(a.replay, encoding="utf-8"))
rep = random.Random(a.seed).sample(old, min(a.n_replay, len(old)))
data = new + rep
random.Random(a.seed + 1).shuffle(data)
missing = sum(1 for s in data for p in s["images"] if not os.path.exists(p))
tags = sum(1 for s in data if sum(c["value"].count("<image>") for c in s["conversations"]) != len(s["images"]))
assert missing == 0, f"{missing} image files missing"
assert tags == 0, f"{tags} samples with <image> tag / image count mismatch"
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
json.dump(data, open(out / "sft_mix.json", "w", encoding="utf-8"), ensure_ascii=False)
info = {a.name: {"file_name": "sft_mix.json", "formatting": "sharegpt",
                 "columns": {"messages": "conversations", "images": "images", "system": "system", "tools": "tools"},
                 "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                          "assistant_tag": "gpt", "observation_tag": "observation"}}}
json.dump(info, open(out / "dataset_info.json", "w"), indent=1)
print(f"new {len(new)} + replay {len(rep)} = {len(data)} samples -> {out}")
