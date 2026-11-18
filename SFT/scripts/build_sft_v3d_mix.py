#!/usr/bin/env python3
"""SFT v3d training mix (2026-11-18): gap-targeted SFT-36K trajectories x2 + a larger v2 replay seen once.

- new: the audited v3d trajectories (ids in --final-ids), taken from the candidate subset written by
  build_sft_v3d_cand.sh (sharegpt, absolute image paths), each included --new-copies times;
- replay: --n-replay samples of the v2 SFT subset; every v2 sample with >= --long-calls tool calls is
  included (long-chain behaviour), the rest is filled stratified by (source, tool category);
- trained for 1 epoch, so new data is seen --new-copies times and each replay sample once.

    python build_sft_v3d_mix.py --final-ids /root/rl_data/v3d_final_ids.txt --out /root/datasets/sft_v3d_mix
"""
import argparse, collections, json, os, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--final-ids", required=True)
ap.add_argument("--cand-dir", default="/root/sft-audit/v3d/cand")
ap.add_argument("--v2-dir", default="/root/datasets/sft_subset_q3vl_v2")
ap.add_argument("--n-replay", type=int, default=2700)
ap.add_argument("--long-calls", type=int, default=5)
ap.add_argument("--new-copies", type=int, default=2)
ap.add_argument("--out", required=True)
ap.add_argument("--name", default="imagentrl_sft_v3d")
ap.add_argument("--seed", type=int, default=20261118)
a = ap.parse_args()
rng = random.Random(a.seed)


def load(d, data_file):
    data = json.load(open(f"{d}/{data_file}", encoding="utf-8"))
    man = [json.loads(l) for l in open(f"{d}/manifest.jsonl", encoding="utf-8")]
    assert len(data) == len(man), (d, len(data), len(man))
    return data, man


want = [l.strip() for l in open(a.final_ids) if l.strip()]
cdata, cman = load(a.cand_dir, "sft_subset.json")
by_id = {m["id"]: s for s, m in zip(cdata, cman)}
missing = [i for i in want if i not in by_id]
assert not missing, f"{len(missing)} final ids not in candidate subset, e.g. {missing[:3]}"
new = [by_id[i] for i in want]

vdata, vman = load(a.v2_dir, "sft_subset.json")
idx = list(range(len(vdata)))
long_idx = [i for i in idx if vman[i]["n_calls"] >= a.long_calls]
rest = [i for i in idx if vman[i]["n_calls"] < a.long_calls]
need = max(0, a.n_replay - len(long_idx))
strata = collections.defaultdict(list)
for i in rest:
    strata[(vman[i]["source"], vman[i]["cat"])].append(i)
fill = []
for k, lst in strata.items():
    rng.shuffle(lst)
    fill += lst[: round(need * len(lst) / len(rest))]
rng.shuffle(fill)
replay_idx = long_idx + fill[:need]
replay = [vdata[i] for i in replay_idx]

data = new * a.new_copies + replay
rng.shuffle(data)
bad_img = sum(1 for s in data for p in s["images"] if not os.path.exists(p))
bad_tag = sum(1 for s in data if sum(c["value"].count("<image>") for c in s["conversations"]) != len(s["images"]))
assert bad_img == 0 and bad_tag == 0, (bad_img, bad_tag)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
json.dump(data, open(out / "sft_mix.json", "w", encoding="utf-8"), ensure_ascii=False)
info = {a.name: {"file_name": "sft_mix.json", "formatting": "sharegpt",
                 "columns": {"messages": "conversations", "images": "images", "system": "system", "tools": "tools"},
                 "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                          "assistant_tag": "gpt", "observation_tag": "observation"}}}
json.dump(info, open(out / "dataset_info.json", "w"), indent=1)
calls = lambda ms: dict(sorted(collections.Counter(min(m["n_calls"], 6) for m in ms).items()))
cm = {m["id"]: m for m in cman}
rep = {"new": len(new), "new_copies": a.new_copies, "replay": len(replay), "replay_long": len(long_idx),
       "total_samples": len(data),
       "new_calls": calls([cm[i] for i in want]), "replay_calls": calls([vman[i] for i in replay_idx]),
       "replay_by_source": dict(collections.Counter(vman[i]["source"] for i in replay_idx)),
       "replay_by_cat": dict(collections.Counter(vman[i]["cat"] for i in replay_idx))}
json.dump(rep, open(out / "report.json", "w"), indent=1)
print(json.dumps(rep))
